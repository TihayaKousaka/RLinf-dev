# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = MagicMock()
if "rlinf.envs.wrappers" not in sys.modules:
    sys.modules["rlinf.envs.wrappers"] = MagicMock()

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.envs.realworld.common.wrappers import critical_phase as critical_phase_module
from rlinf.envs.realworld.common.wrappers.critical_phase import CriticalPhaseWrapper
from rlinf.models.embodiment.openpi_rlt.rollout import (
    COLLECTION_PHASE_ONLINE,
    COLLECTION_PHASE_WARMUP,
    RLTActionRouteInputs,
    RLTStage2RolloutRouteConfig,
    TransitionSource,
    route_rlt_stage2_actions,
    route_rlt_stage2_rollout,
)
from rlinf.models.embodiment.openpi_rlt.trajectory_adapter import (
    RLTStage2TrajectoryReplayAdapter,
)


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class DummyRealWorldEnv:
    def __init__(self):
        self.config = AttrDict(is_dummy=False)
        self.unwrapped = self
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        return {}, {}

    def step(self, action):
        self.step_count += 1
        return {}, 0.0, False, False, {}


class FakeKeyboardListener:
    def __init__(self):
        self.press_batches: list[list[str]] = [["v"]]
        self.drained_on_reset = False

    def pop_pressed_keys(self) -> list[str]:
        if self.press_batches:
            return self.press_batches.pop(0)
        return []


class FakeStage2Student:
    def predict_action_batch(self, **_kwargs):
        base_flat = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        actions = (base_flat + 100.0).reshape(1, 2, 2)
        return actions, {
            "prev_logprobs": torch.zeros((1, 1), dtype=torch.float32),
            "prev_values": torch.zeros((1, 1), dtype=torch.float32),
            "forward_inputs": {
                "x": torch.zeros((1, 2), dtype=torch.float32),
                "a_tilde": base_flat,
            },
        }


def _replay_cfg() -> AttrDict:
    return AttrDict(
        actor=AttrDict(
            model=AttrDict(
                num_action_chunks=2,
                action_dim=2,
                rlt_stage2=AttrDict(),
            ),
        ),
        env=AttrDict(train=AttrDict(auto_reset=True, env_type="realworld")),
    )


def _route_chunk(
    *,
    ready_for_online: bool,
    in_critical_phase: bool,
    record_transition: bool,
    source_base: float,
    human_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    chunk_len = 2
    action_dim = 2
    base_flat = torch.tensor(
        [[source_base, source_base + 1.0, source_base + 2.0, source_base + 3.0]],
        dtype=torch.float32,
    )
    student_actions = (base_flat + 100.0).reshape(1, chunk_len, action_dim)
    expert_takeover = torch.zeros((1,), dtype=torch.bool)
    human_action = None
    human_flags = None
    if human_step is not None:
        human_action = base_flat + 1000.0
        human_flags = torch.zeros((1, chunk_len), dtype=torch.bool)
        human_flags[:, human_step] = True

    route = route_rlt_stage2_actions(
        RLTActionRouteInputs(
            student_actions=student_actions,
            base_flat=base_flat,
            expert_actions=None,
            expert_takeover=expert_takeover,
            requested_expert_takeover=expert_takeover.clone(),
            intervention_phase=torch.zeros((1,), dtype=torch.float32),
            in_critical_phase=torch.full((1,), in_critical_phase, dtype=torch.bool),
            record_transition=torch.full((1,), record_transition, dtype=torch.bool),
            ready_for_online=ready_for_online,
            online_gate_step=1000,
            chunk_length=chunk_len,
            action_dim=action_dim,
        )
    )
    forward_inputs = {
        "x": torch.tensor([[source_base, source_base + 0.5]], dtype=torch.float32),
        "a_tilde": route.base_flat.detach(),
        **route.to_forward_input_updates(),
    }
    return route.action_flat.detach(), forward_inputs, human_action, human_flags


def _append_routed_chunk(
    rollout: EmbodiedRolloutResult,
    *,
    ready_for_online: bool,
    in_critical_phase: bool,
    record_transition: bool,
    source_base: float,
    done: bool = False,
    human_step: int | None = None,
) -> None:
    action, forward_inputs, human_action, human_flags = _route_chunk(
        ready_for_online=ready_for_online,
        in_critical_phase=in_critical_phase,
        record_transition=record_transition,
        source_base=source_base,
        human_step=human_step,
    )
    rollout.append_step_result(
        ChunkStepResult(
            actions=action,
            rewards=torch.tensor([[0.0, float(done)]], dtype=torch.float32),
            dones=torch.tensor([[False, done]], dtype=torch.bool),
            terminations=torch.tensor([[False, done]], dtype=torch.bool),
            truncations=torch.zeros((1, 2), dtype=torch.bool),
            forward_inputs=forward_inputs,
            versions=torch.zeros((1, 1), dtype=torch.float32),
        )
    )
    if human_action is not None and human_flags is not None:
        rollout.update_last_actions(
            intervene_actions=human_action,
            intervene_flags=human_flags,
        )


def test_critical_phase_key_uses_edge_press_queue(monkeypatch):
    listeners: list[FakeKeyboardListener] = []

    def _make_listener():
        listener = FakeKeyboardListener()
        listeners.append(listener)
        return listener

    monkeypatch.setattr(critical_phase_module, "KeyboardListener", _make_listener)

    env = CriticalPhaseWrapper(
        DummyRealWorldEnv(),
        task_mode="full_task",
        critical_phase_key="v",
        record_prefix_before_critical_phase=False,
    )
    _obs, info = env.reset()
    assert info["record_transition"] is False
    assert listeners[0].press_batches == []

    listeners[0].press_batches.append(["v"])
    _obs, _reward, _terminated, _truncated, info = env.step(None)

    assert info["in_critical_phase"] is True
    assert info["record_transition"] is True


def test_realworld_boundary_chunk_waits_until_next_rollout_to_enter_replay():
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    base_action = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    human_action = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
    rollout.append_step_result(
        ChunkStepResult(
            actions=base_action,
            rewards=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            dones=torch.tensor([[False, True]], dtype=torch.bool),
            terminations=torch.tensor([[False, True]], dtype=torch.bool),
            truncations=torch.zeros((1, 2), dtype=torch.bool),
            forward_inputs={
                "x": torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32),
                "a_tilde": base_action.clone(),
                "action": base_action.clone(),
                "action_chunk": base_action.clone(),
                "intervention_flags": torch.zeros((1, 2), dtype=torch.bool),
                "source_chunk": torch.full(
                    (1, 2),
                    int(TransitionSource.BASE),
                    dtype=torch.uint8,
                ),
                "source": torch.full(
                    (1, 1),
                    int(TransitionSource.BASE),
                    dtype=torch.uint8,
                ),
                "collection_phase_id": torch.tensor(
                    [[COLLECTION_PHASE_ONLINE]],
                    dtype=torch.uint8,
                ),
                "record_transition": torch.zeros((1, 1), dtype=torch.bool),
            },
        )
    )

    rollout.update_last_actions(
        intervene_actions=human_action,
        intervene_flags=torch.tensor([[False, True]], dtype=torch.bool),
    )

    adapter = RLTStage2TrajectoryReplayAdapter(_replay_cfg())
    replay_before_boundary, _ = adapter.build_replay_trajectories(
        rollout.to_trajectory()
    )
    assert replay_before_boundary == []

    expected_executed_action = torch.tensor(
        [[1.0, 2.0, 30.0, 40.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(rollout.actions[-1], expected_executed_action)
    torch.testing.assert_close(
        rollout.forward_inputs[-1]["action"],
        expected_executed_action,
    )
    torch.testing.assert_close(
        rollout.forward_inputs[-1]["action_chunk"],
        base_action,
    )
    torch.testing.assert_close(
        rollout.forward_inputs[-1]["intervention_flags"],
        torch.zeros((1, 2), dtype=torch.bool),
    )
    torch.testing.assert_close(
        rollout.forward_inputs[-1]["source_chunk"],
        torch.tensor(
            [[int(TransitionSource.BASE), int(TransitionSource.BASE)]],
            dtype=torch.uint8,
        ),
    )
    assert bool(rollout.forward_inputs[-1]["record_transition"].item()) is False

    replay_trajectories, completed_episodes = adapter.build_replay_trajectories(
        rollout.to_trajectory()
    )

    assert completed_episodes == 0
    assert replay_trajectories == []


def test_realworld_stage2_phase_gate_and_replay_sources():
    rollout = EmbodiedRolloutResult(max_episode_length=5)

    _append_routed_chunk(
        rollout,
        ready_for_online=False,
        in_critical_phase=False,
        record_transition=False,
        source_base=0.0,
    )
    _append_routed_chunk(
        rollout,
        ready_for_online=False,
        in_critical_phase=False,
        record_transition=False,
        source_base=10.0,
        human_step=1,
    )
    _append_routed_chunk(
        rollout,
        ready_for_online=False,
        in_critical_phase=True,
        record_transition=True,
        source_base=20.0,
    )
    _append_routed_chunk(
        rollout,
        ready_for_online=True,
        in_critical_phase=False,
        record_transition=False,
        source_base=30.0,
    )
    _append_routed_chunk(
        rollout,
        ready_for_online=True,
        in_critical_phase=True,
        record_transition=True,
        source_base=40.0,
        done=True,
        human_step=0,
    )

    source_chunks = [
        forward_inputs["source_chunk"].reshape(-1).tolist()
        for forward_inputs in rollout.forward_inputs
    ]
    assert source_chunks == [
        [int(TransitionSource.BASE), int(TransitionSource.BASE)],
        [int(TransitionSource.BASE), int(TransitionSource.BASE)],
        [int(TransitionSource.BASE), int(TransitionSource.BASE)],
        [int(TransitionSource.BASE), int(TransitionSource.BASE)],
        [int(TransitionSource.RL), int(TransitionSource.RL)],
    ]
    assert [bool(fi["record_transition"].item()) for fi in rollout.forward_inputs] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert [bool(fi["student_control"].item()) for fi in rollout.forward_inputs] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert [int(fi["collection_phase_id"].item()) for fi in rollout.forward_inputs] == [
        COLLECTION_PHASE_WARMUP,
        COLLECTION_PHASE_WARMUP,
        COLLECTION_PHASE_WARMUP,
        COLLECTION_PHASE_ONLINE,
        COLLECTION_PHASE_ONLINE,
    ]

    replay_trajectories, completed_episodes = (
        RLTStage2TrajectoryReplayAdapter(_replay_cfg()).build_replay_trajectories(
            rollout.to_trajectory()
        )
    )

    assert completed_episodes == 1
    assert len(replay_trajectories) == 2
    replay_sources = [
        int(traj.forward_inputs["source"].item()) for traj in replay_trajectories
    ]
    assert replay_sources == [
        int(TransitionSource.BASE),
        int(TransitionSource.HUMAN),
    ]
    replay_source_chunks = [
        traj.forward_inputs["source_chunk"].reshape(-1).tolist()
        for traj in replay_trajectories
    ]
    assert replay_source_chunks == [
        [int(TransitionSource.BASE), int(TransitionSource.BASE)],
        [int(TransitionSource.HUMAN), int(TransitionSource.HUMAN)],
    ]
    torch.testing.assert_close(
        replay_trajectories[0].forward_inputs["action_chunk"].reshape(-1),
        torch.tensor([20.0, 21.0, 22.0, 23.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        replay_trajectories[1].forward_inputs["action_chunk"].reshape(-1),
        torch.tensor([1040.0, 1041.0, 142.0, 143.0], dtype=torch.float32),
    )


def test_realworld_missing_critical_metadata_defaults_to_base_control():
    student = FakeStage2Student()

    realworld_route = route_rlt_stage2_rollout(
        env_obs={},
        policy_info=None,
        student_model=student,
        expert_model_getter=lambda: None,
        model_kwargs={},
        cfg=RLTStage2RolloutRouteConfig(
            ready_for_online=True,
            online_gate_step=1000,
            allow_expert=False,
            chunk_length=2,
            action_dim=2,
            in_critical_phase_default=False,
            record_transition_default=False,
        ),
    )
    torch.testing.assert_close(
        realworld_route.actions.reshape(-1),
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    )
    assert realworld_route.result["forward_inputs"]["source_chunk"].reshape(-1).tolist() == [
        int(TransitionSource.BASE),
        int(TransitionSource.BASE),
    ]
    assert bool(realworld_route.result["forward_inputs"]["student_control"].item()) is False
    assert bool(realworld_route.result["forward_inputs"]["record_transition"].item()) is False

    default_route = route_rlt_stage2_rollout(
        env_obs={},
        policy_info=None,
        student_model=student,
        expert_model_getter=lambda: None,
        model_kwargs={},
        cfg=RLTStage2RolloutRouteConfig(
            ready_for_online=True,
            online_gate_step=1000,
            allow_expert=False,
            chunk_length=2,
            action_dim=2,
        ),
    )
    torch.testing.assert_close(
        default_route.actions.reshape(-1),
        torch.tensor([101.0, 102.0, 103.0, 104.0], dtype=torch.float32),
    )
    assert default_route.result["forward_inputs"]["source_chunk"].reshape(-1).tolist() == [
        int(TransitionSource.RL),
        int(TransitionSource.RL),
    ]
    assert bool(default_route.result["forward_inputs"]["student_control"].item()) is True
