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

from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import numpy as _numpy  # noqa: F401
except ModuleNotFoundError:
    _HAS_NUMPY = False
else:
    _HAS_NUMPY = True

pytestmark = pytest.mark.skipif(
    torch is None or not _HAS_NUMPY,
    reason="RLT synthetic refactor tests require torch and numpy.",
)

if torch is not None and _HAS_NUMPY:
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from rlinf.data.embodied_buffer_dataset import (
        ReplayBufferDataset,
        replay_buffer_collate_fn,
    )
    from rlinf.data.embodied_io_struct import (
        ChunkStepResult,
        EmbodiedRolloutResult,
        RolloutResult,
        Trajectory,
    )
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    from rlinf.envs.maniskill.rlt_intervention import (
        ManiSkillLocalCorrectionController,
    )
    from rlinf.models.embodiment.rlt_stage2.rollout import (
        COLLECTION_PHASE_ONLINE,
        COLLECTION_PHASE_WARMUP,
        RLTStage2RolloutRouteConfig,
        TransitionSource,
        route_rlt_stage2_rollout,
    )
    from rlinf.models.embodiment.rlt_stage2.rlt_stage2_policy import RLTStage2Policy
    from rlinf.models.embodiment.rlt_stage2.trajectory_adapter import (
        RLTStage2TrajectoryReplayAdapter,
    )
    from rlinf.workers.actor.fsdp_rlt_stage2_policy_worker import (
        RLTStage2FSDPPolicyWorker,
    )
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from rlinf.scheduler import Worker
    from toolkits.rlt import inspect_rlt_replay
else:
    RLTStage2Policy = object


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


ROOT = Path(__file__).resolve().parents[2]
MANISKILL_STAGE2_CONFIG = (
    ROOT / "examples/embodiment/config/rlt_stage2_maniskill_joint.yaml"
)
RLT_STAGE2_MODEL_CONFIG = (
    ROOT / "examples/embodiment/config/model/rlt_stage2_joint.yaml"
)


def _cfg(
    *,
    env_type: str = "realworld",
    task_mode: str = "critical_phase",
    intervention_enable: bool = True,
    intervention_mode: str = "human_override",
    warmup_updates: int = 5,
    replay_subsample_stride: int = 0,
) -> AttrDict:
    return AttrDict(
        algorithm=AttrDict(
            loss_type="rlt_td3",
            warmup_post_collect_updates=warmup_updates,
            intervention=AttrDict(
                enable=intervention_enable,
                mode=intervention_mode,
            ),
        ),
        actor=AttrDict(
            model=AttrDict(
                model_type="rlt_stage2",
                num_action_chunks=2,
                action_dim=2,
                rlt_stage2=AttrDict(
                    replay_subsample_stride=replay_subsample_stride,
                    replay_allow_terminal_partial=True,
                    replay_feature_batch_size=32,
                ),
            ),
        ),
        env=AttrDict(
            train=AttrDict(
                env_type=env_type,
                task_mode=task_mode,
                auto_reset=True,
            ),
            eval=AttrDict(
                env_type=env_type,
                task_mode=task_mode,
                auto_reset=True,
            ),
        ),
        rollout=AttrDict(
            expert_model=AttrDict(act_as_vla_reference=False),
        ),
    )


def test_maniskill_rlt_intervention_config_remains_user_tunable():
    cfg = OmegaConf.load(MANISKILL_STAGE2_CONFIG)
    model_cfg = OmegaConf.load(RLT_STAGE2_MODEL_CONFIG)
    cfg = OmegaConf.merge(
        OmegaConf.create({"actor": {"model": model_cfg}}),
        cfg,
    )

    intervention = cfg.algorithm.intervention
    expected_keys = {
        "enable",
        "mode",
        "deviation_patience",
        "takeover_chunks",
        "takeover_max_chunks",
        "safe_yz_margin",
        "progress_eps",
        "yz_error_eps",
        "near_hole_x_min",
        "near_hole_yz_margin",
        "exit_hole_x_min",
        "fallback_hole_radius",
    }
    assert expected_keys.issubset(set(intervention.keys()))
    assert cfg.env.train.rlt_intervention is not None
    assert OmegaConf.to_container(
        cfg.env.train.rlt_intervention,
        resolve=True,
    ) == OmegaConf.to_container(intervention, resolve=True)
    assert cfg.env.eval.rlt_intervention.enable is False

    stage2_cfg = cfg.actor.model.rlt_stage2
    assert stage2_cfg.online_gate_updates == cfg.algorithm.warmup_post_collect_updates
    assert stage2_cfg.intervention_enabled is True
    assert stage2_cfg.intervention_mode == "local_correction"

    removed_grasp_phase_keys = {
        "enable_grasp_phase",
        "grasp_near_peg_dist",
        "grasp_deviation_patience",
        "grasp_takeover_chunks",
        "grasp_takeover_max_chunks",
    }
    assert removed_grasp_phase_keys.isdisjoint(set(intervention.keys()))


def test_rlt_expert_model_override_fits_structured_stage2_schema():
    model_cfg = OmegaConf.load(RLT_STAGE2_MODEL_CONFIG)
    OmegaConf.set_struct(model_cfg, True)

    expert_overrides = OmegaConf.create(
        {
            "rlt_stage2": {
                "act_as_vla_reference": True,
                "load_feature_backbones": True,
                "load_rl_token_model": False,
            }
        }
    )

    merged = OmegaConf.merge(model_cfg, expert_overrides)

    assert merged.rlt_stage2.act_as_vla_reference is True
    assert merged.rlt_stage2.load_feature_backbones is True
    assert merged.rlt_stage2.load_rl_token_model is False


def test_maniskill_local_correction_uses_intervention_takeover_knobs():
    controller = ManiSkillLocalCorrectionController(
        cfg=OmegaConf.create(
            {
                "algorithm": {
                    "intervention": {
                        "enable": True,
                        "mode": "local_correction",
                        "deviation_patience": 1,
                        "takeover_chunks": 2,
                        "takeover_max_chunks": 3,
                        "safe_yz_margin": 2.0,
                        "progress_eps": 0.01,
                        "yz_error_eps": 0.01,
                        "near_hole_x_min": -0.03,
                        "near_hole_yz_margin": 1.5,
                        "exit_hole_x_min": -0.10,
                        "fallback_hole_radius": 0.035,
                    }
                }
            }
        ),
        batch_size=1,
        mode="train",
    )
    infos = {
        "consecutive_grasp_current": torch.tensor([True]),
        "prealigned_current": torch.tensor([True]),
        "partial_insert_current": torch.tensor([False]),
        "success_current": torch.tensor([False]),
        "peg_head_goal_yz_dist": torch.tensor([0.0]),
        "peg_body_goal_yz_dist": torch.tensor([0.0]),
        "peg_head_hole_x": torch.tensor([0.0]),
        "peg_head_hole_abs_y": torch.tensor([0.0]),
        "peg_head_hole_abs_z": torch.tensor([0.0]),
    }
    chunk_dones = torch.zeros((1, 2), dtype=torch.bool)

    first = controller.update(
        infos=infos,
        chunk_dones=chunk_dones,
        intervention_enabled=True,
    )
    second = controller.update(
        infos=infos,
        chunk_dones=chunk_dones,
        intervention_enabled=True,
    )

    assert bool(first["expert_takeover"].item()) is False
    assert bool(second["expert_takeover"].item()) is True
    assert second["intervention_phase"].item() == controller.INSERT_PHASE
    assert second["takeover_left"].item() == 2.0


def _synthetic_rollout_trajectory() -> Trajectory:
    actions = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros((3, 1, 2), dtype=torch.bool)
    dones[2, 0, 1] = True

    return Trajectory(
        max_episode_length=2,
        model_weights_id="synthetic",
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        forward_inputs={
            "x": torch.tensor(
                [
                    [[0.0, 0.1, 0.2]],
                    [[1.0, 1.1, 1.2]],
                    [[2.0, 2.1, 2.2]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.1, 0.1, 0.1, 0.1]],
                    [[0.2, 0.2, 0.2, 0.2]],
                ],
                dtype=torch.float32,
            ),
            "intervention_flags": torch.tensor(
                [
                    [[False, False]],
                    [[True, False]],
                ],
                dtype=torch.bool,
            ),
            "source_chunk": torch.tensor(
                [
                    [[int(TransitionSource.RL), int(TransitionSource.RL)]],
                    [[int(TransitionSource.HUMAN), int(TransitionSource.MIXED)]],
                ],
                dtype=torch.uint8,
            ),
            "collection_phase_id": torch.tensor(
                [
                    [[COLLECTION_PHASE_WARMUP]],
                    [[COLLECTION_PHASE_ONLINE]],
                ],
                dtype=torch.uint8,
            ),
            "record_transition": torch.ones((2, 1, 1), dtype=torch.bool),
        },
    )


def _synthetic_stride_rollout_trajectory() -> Trajectory:
    actions = torch.tensor(
        [
            [[1.0, 1.1, 2.0, 2.1]],
            [[3.0, 3.1, 4.0, 4.1]],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros((3, 1, 2), dtype=torch.bool)
    dones[2, 0, 1] = True

    return Trajectory(
        max_episode_length=2,
        model_weights_id="synthetic-stride",
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        intervene_flags=torch.tensor(
            [
                [[[False], [False]]],
                [[[False], [True]]],
            ],
            dtype=torch.bool,
        ),
        forward_inputs={
            "x": torch.tensor(
                [
                    [[0.0, 0.1, 0.2]],
                    [[2.0, 2.1, 2.2]],
                    [[4.0, 4.1, 4.2]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[10.0, 10.1, 10.2, 10.3]],
                    [[30.0, 30.1, 30.2, 30.3]],
                    [[50.0, 50.1, 50.2, 50.3]],
                ],
                dtype=torch.float32,
            ),
            "rlt_step_trace": {
                "x": torch.tensor(
                    [
                        [[[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]],
                        [[[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]],
                        [[[3.0, 3.1, 3.2], [4.0, 4.1, 4.2]]],
                    ],
                    dtype=torch.float32,
                ),
                "a_tilde": torch.tensor(
                    [
                        [[[10.0, 10.1, 10.2, 10.3], [10.0, 10.1, 10.2, 10.3]]],
                        [[[20.0, 20.1, 20.2, 20.3], [30.0, 30.1, 30.2, 30.3]]],
                        [[[40.0, 40.1, 40.2, 40.3], [50.0, 50.1, 50.2, 50.3]]],
                    ],
                    dtype=torch.float32,
                ),
            },
            "rlt_step_trace_valid": torch.tensor(
                [
                    [[False]],
                    [[True]],
                    [[True]],
                ],
                dtype=torch.bool,
            ),
            "source_chunk": torch.tensor(
                [
                    [[int(TransitionSource.RL), int(TransitionSource.RL)]],
                    [[int(TransitionSource.RL), int(TransitionSource.HUMAN)]],
                ],
                dtype=torch.uint8,
            ),
            "collection_phase_id": torch.tensor(
                [
                    [[COLLECTION_PHASE_WARMUP]],
                    [[COLLECTION_PHASE_ONLINE]],
                ],
                dtype=torch.uint8,
            ),
            "record_transition": torch.ones((2, 1, 1), dtype=torch.bool),
        },
    )


def _build_replay_trajectories() -> list[Trajectory]:
    adapter = RLTStage2TrajectoryReplayAdapter(_cfg())
    replay_trajectories, completed_episodes = adapter.build_replay_trajectories(
        _synthetic_rollout_trajectory()
    )
    assert completed_episodes == 1
    return replay_trajectories


def test_trajectory_adapter_emits_standard_replay_trajectories():
    replay_trajectories = _build_replay_trajectories()

    assert len(replay_trajectories) == 2
    first_inputs = replay_trajectories[0].forward_inputs
    second_inputs = replay_trajectories[1].forward_inputs

    required_keys = {
        "x",
        "a",
        "a_tilde",
        "action_chunk",
        "ref_chunk",
        "rewards",
        "next_x",
        "next_a_tilde",
        "next_ref_chunk",
        "dones",
        "intervention",
        "source",
        "source_chunk",
        "collection_phase_id",
        "success",
        "intervention_flag",
        "episode_id",
        "step_id",
    }
    assert required_keys.issubset(first_inputs)
    assert first_inputs["x"].shape == (1, 1, 3)
    assert first_inputs["action_chunk"].shape == (1, 1, 4)
    assert first_inputs["source_chunk"].shape == (1, 1, 2)
    assert first_inputs["dones"].item() == 0.0

    assert second_inputs["dones"].item() == 1.0
    assert bool(second_inputs["intervention_flag"].item()) is True
    assert second_inputs["collection_phase_id"].item() == COLLECTION_PHASE_ONLINE
    assert second_inputs["source"].item() == TransitionSource.MIXED
    torch.testing.assert_close(
        second_inputs["action_chunk"].reshape(-1),
        torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        second_inputs["a_tilde"].reshape(-1),
        torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32),
    )


def test_trajectory_adapter_marks_env_side_override_as_human_source():
    traj = _synthetic_rollout_trajectory()
    traj.forward_inputs["source_chunk"] = torch.tensor(
        [
            [[int(TransitionSource.RL), int(TransitionSource.RL)]],
            [[int(TransitionSource.RL), int(TransitionSource.RL)]],
        ],
        dtype=torch.uint8,
    )
    traj.forward_inputs["intervention_flags"] = torch.tensor(
        [
            [[False, False]],
            [[False, True]],
        ],
        dtype=torch.bool,
    )

    adapter = RLTStage2TrajectoryReplayAdapter(_cfg())
    replay_trajectories, _ = adapter.build_replay_trajectories(traj)
    second_inputs = replay_trajectories[1].forward_inputs

    assert second_inputs["source"].item() == TransitionSource.MIXED
    assert (
        second_inputs["source_chunk"].reshape(-1)
        == torch.tensor(
            [int(TransitionSource.RL), int(TransitionSource.HUMAN)],
            dtype=torch.uint8,
        )
    ).all()
    assert bool(second_inputs["intervention_flag"].item()) is True


def test_trajectory_adapter_builds_step_stride_windows_from_trace():
    adapter = RLTStage2TrajectoryReplayAdapter(
        _cfg(replay_subsample_stride=1),
    )

    replay_trajectories, completed_episodes = adapter.build_replay_trajectories(
        _synthetic_stride_rollout_trajectory()
    )

    assert completed_episodes == 1
    assert len(replay_trajectories) == 4
    inputs = [trajectory.forward_inputs for trajectory in replay_trajectories]

    torch.testing.assert_close(
        torch.stack([item["x"].reshape(-1) for item in inputs]),
        torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        torch.stack([item["next_x"].reshape(-1) for item in inputs]),
        torch.tensor(
            [
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
                [4.0, 4.1, 4.2],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        torch.stack([item["action_chunk"].reshape(-1) for item in inputs]),
        torch.tensor(
            [
                [1.0, 1.1, 2.0, 2.1],
                [2.0, 2.1, 3.0, 3.1],
                [3.0, 3.1, 4.0, 4.1],
                [4.0, 4.1, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        torch.stack([item["ref_chunk"].reshape(-1) for item in inputs]),
        torch.tensor(
            [
                [10.0, 10.1, 10.2, 10.3],
                [20.0, 20.1, 20.2, 20.3],
                [30.0, 30.1, 30.2, 30.3],
                [40.0, 40.1, 40.2, 40.3],
            ],
            dtype=torch.float32,
        ),
    )
    assert [item["step_id"].item() for item in inputs] == [0, 1, 2, 3]
    assert [bool(item["dones"].item()) for item in inputs] == [
        False,
        False,
        True,
        True,
    ]
    assert inputs[2]["source"].item() == TransitionSource.MIXED
    assert bool(inputs[2]["intervention_flag"].item()) is True
    assert inputs[3]["source_chunk"].reshape(-1).tolist() == [
        int(TransitionSource.HUMAN),
        int(TransitionSource.BASE),
    ]
    assert inputs[3]["source"].item() == TransitionSource.HUMAN


def test_trajectory_adapter_rejects_stride_without_step_trace():
    adapter = RLTStage2TrajectoryReplayAdapter(
        _cfg(replay_subsample_stride=1),
    )
    traj = _synthetic_stride_rollout_trajectory()
    traj.forward_inputs.pop("rlt_step_trace")

    with pytest.raises(RuntimeError, match="rlt_step_trace"):
        adapter.build_replay_trajectories(traj)


def test_rollout_pipeline_preserves_nested_rlt_step_trace_forward_inputs():
    rollout_result = RolloutResult(
        actions=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        forward_inputs={
            "x": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "rlt_step_trace": {
                "x": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
                "a_tilde": torch.arange(16, dtype=torch.float32).reshape(2, 2, 4),
            },
            "rlt_step_trace_valid": torch.ones((2, 1), dtype=torch.bool),
        },
    )

    worker = object.__new__(MultiStepRolloutWorker)
    split_results = MultiStepRolloutWorker._split_rollout_result(
        worker,
        rollout_result,
        [1, 1],
    )

    assert len(split_results) == 2
    assert split_results[0].forward_inputs["rlt_step_trace"]["x"].shape == (1, 2, 3)
    torch.testing.assert_close(
        split_results[1].forward_inputs["rlt_step_trace"]["a_tilde"].reshape(-1),
        torch.arange(8, 16, dtype=torch.float32),
    )

    builder = EmbodiedRolloutResult(max_episode_length=2)
    for split_result in split_results:
        builder.append_step_result(
            ChunkStepResult(
                actions=split_result.actions,
                forward_inputs=split_result.forward_inputs,
            )
        )

    trajectory = builder.to_trajectory()
    assert trajectory.forward_inputs["rlt_step_trace"]["x"].shape == (2, 1, 2, 3)
    torch.testing.assert_close(
        trajectory.forward_inputs["rlt_step_trace"]["x"][:, 0],
        rollout_result.forward_inputs["rlt_step_trace"]["x"],
    )


def test_rlt_replay_buffer_roundtrip_uses_rlinf_trajectory_format(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=7,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    buffer.add_trajectories(replay_trajectories)
    buffer.close(wait=True)

    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "trajectory_index.json").is_file()
    assert len(list(tmp_path.glob("trajectory_*.pt"))) == 2

    loaded = TrajectoryReplayBuffer(
        seed=7,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    loaded.load_checkpoint(str(tmp_path))
    batch = loaded.sample(2)
    loaded.close(wait=True)

    assert loaded.total_samples == 2
    assert "forward_inputs" in batch
    forward_inputs = batch["forward_inputs"]
    assert forward_inputs["source_chunk"].shape == (2, 2)
    assert forward_inputs["rewards"].shape == (2, 2)
    assert forward_inputs["x"].shape == (2, 3)
    assert set(forward_inputs).issuperset(
        {
            "x",
            "a",
            "a_tilde",
            "action_chunk",
            "ref_chunk",
            "rewards",
            "next_x",
            "next_a_tilde",
            "next_ref_chunk",
            "dones",
            "intervention",
            "source",
            "source_chunk",
            "collection_phase_id",
            "success",
            "intervention_flag",
            "episode_id",
            "step_id",
        }
    )


def test_rlt_replay_buffer_caps_active_samples_like_legacy_ring_buffer(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=8,
        enable_cache=True,
        cache_size=8,
        sample_window_size=8,
        max_num_samples=3,
        auto_save=False,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )

    buffer.add_trajectories(replay_trajectories)
    buffer.add_trajectories(replay_trajectories)
    first_active_ids = list(buffer._trajectory_id_list)
    buffer.add_trajectories(replay_trajectories)
    buffer.save_checkpoint(str(tmp_path / "checkpoint"))

    assert buffer.total_samples == 3
    assert buffer.is_ready(3)
    assert not buffer.is_ready(4)
    assert len(buffer) == 3
    assert list(buffer._trajectory_id_list) == [3, 4, 5]
    assert first_active_ids == [1, 2, 3]
    assert all(
        trajectory_id not in buffer._trajectory_index
        for trajectory_id in [0, 1, 2]
    )

    batch = buffer.sample(3)
    source_values = batch["forward_inputs"]["source"].reshape(-1).tolist()
    assert all(
        value in (int(TransitionSource.RL), int(TransitionSource.MIXED))
        for value in source_values
    )
    buffer.close(wait=True)

    loaded = TrajectoryReplayBuffer(
        seed=8,
        enable_cache=True,
        cache_size=8,
        sample_window_size=8,
        max_num_samples=3,
        auto_save=False,
        auto_save_path=str(tmp_path / "loaded"),
        trajectory_format="pt",
    )
    loaded.load_checkpoint(str(tmp_path / "checkpoint"))
    assert loaded.total_samples == 3
    assert len(loaded) == 3
    assert list(loaded._trajectory_id_list) == [3, 4, 5]
    loaded.close(wait=True)


def test_rlt_replay_buffer_autosave_checkpoint_preserves_active_cap(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=18,
        enable_cache=True,
        cache_size=2,
        sample_window_size=8,
        max_num_samples=3,
        auto_save=True,
        auto_save_path=str(tmp_path / "autosave"),
        trajectory_format="pt",
    )

    buffer.add_trajectories(replay_trajectories)
    buffer.add_trajectories(replay_trajectories)
    buffer.add_trajectories(replay_trajectories)
    buffer._save_executor.shutdown(wait=True)
    buffer.save_checkpoint(str(tmp_path / "checkpoint"))

    saved_files = sorted((tmp_path / "checkpoint").glob("trajectory_*.pt"))
    assert len(saved_files) == 3

    loaded = TrajectoryReplayBuffer(
        seed=18,
        enable_cache=True,
        cache_size=2,
        sample_window_size=8,
        max_num_samples=3,
        auto_save=False,
        auto_save_path=str(tmp_path / "loaded"),
        trajectory_format="pt",
    )
    loaded.load_checkpoint(str(tmp_path / "checkpoint"))

    assert loaded.total_samples == 3
    assert len(loaded) == 3
    assert list(loaded._trajectory_id_list) == [3, 4, 5]
    buffer.close(wait=True)
    loaded.close(wait=True)


def test_rlt_worker_replay_builder_uses_buffer_capacity_for_active_samples():
    worker = object.__new__(RLTStage2FSDPPolicyWorker)
    worker.cfg = AttrDict(
        runner=AttrDict(
            logger=AttrDict(
                log_path="/tmp/rlinf-rlt-test",
            ),
        ),
    )
    worker._rank = 0

    replay_buffer = RLTStage2FSDPPolicyWorker._build_trajectory_replay_buffer(
        worker,
        AttrDict(
            enable_cache=True,
            cache_size=2,
            sample_window_size=1,
            auto_save=False,
            trajectory_format="pt",
        ),
        seed=9,
        default_subdir="replay_buffer",
        capacity=5,
    )

    assert replay_buffer.max_num_samples == 5
    assert replay_buffer.sample_window_size == 5
    replay_buffer.close(wait=True)


def test_rlt_replay_stats_only_use_active_capped_samples(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=10,
        enable_cache=True,
        cache_size=8,
        sample_window_size=8,
        max_num_samples=1,
        auto_save=False,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    buffer.add_trajectories(replay_trajectories)

    stats = RLTStage2FSDPPolicyWorker._rlt_replay_stats(buffer)

    assert buffer.total_samples == 1
    assert stats["intervention_rate"] == 0.5
    assert stats["human_chunk_rate"] == 1.0
    buffer.close(wait=True)


def test_rlt_worker_training_batch_uses_replay_buffer_dataset():
    replay_trajectories = _build_replay_trajectories()
    replay_buffer = TrajectoryReplayBuffer(
        seed=13,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=False,
        auto_save_path="",
        trajectory_format="pt",
    )
    replay_buffer.add_trajectories(replay_trajectories)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=None,
        batch_size=2,
        min_replay_buffer_size=2,
        min_demo_buffer_size=0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        collate_fn=replay_buffer_collate_fn,
    )

    worker = object.__new__(RLTStage2FSDPPolicyWorker)
    worker.buffer_dataloader_iter = iter(dataloader)
    worker.device = torch.device("cpu")

    batch = RLTStage2FSDPPolicyWorker._next_rlt_replay_batch(worker, 2)
    replay_buffer.close(wait=True)

    assert "forward_inputs" not in batch
    assert batch["x"].shape == (2, 3)
    assert batch["a"].shape == (2, 4)
    assert batch["a_tilde"].shape == (2, 4)
    assert batch["action_chunk"].shape == (2, 4)
    assert batch["source_chunk"].shape == (2, 2)
    assert batch["rewards"].shape == (2, 2)


class _FakeRLTWorkerModel:
    def filter_rollout_state_dict(self, state_dict):
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith("actor.")
        }


def test_rlt_worker_rollout_sync_keeps_update_step_gate_version():
    worker = object.__new__(RLTStage2FSDPPolicyWorker)
    worker.model = _FakeRLTWorkerModel()
    worker.update_step = 17
    worker.version = 99
    worker._rollout_sync_key_count = 0
    worker.get_model_state_dict = lambda **kwargs: {
        "actor.weight": torch.ones(1),
        "critic.weight": torch.zeros(1),
    }

    state_dict = worker.get_rollout_state_dict()

    assert list(state_dict) == ["actor.weight"]
    assert worker.get_rollout_sync_param_names(state_dict) == ["actor.weight"]
    assert worker.get_rollout_sync_version() == 17
    assert worker._rollout_sync_key_count == 1


def test_rlt_replay_checkpoint_helpers_use_standard_rank_paths(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    component_dir = tmp_path / "components"
    worker = object.__new__(RLTStage2FSDPPolicyWorker)
    worker._rank = 3
    worker.replay_buffer = TrajectoryReplayBuffer(
        seed=21,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=False,
        auto_save_path=str(tmp_path / "unused_replay"),
        trajectory_format="pt",
    )
    worker.demo_buffer = TrajectoryReplayBuffer(
        seed=22,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=False,
        auto_save_path=str(tmp_path / "unused_demo"),
        trajectory_format="pt",
    )
    worker.replay_buffer.add_trajectories(replay_trajectories)
    worker.demo_buffer.add_trajectories(replay_trajectories[:1])

    RLTStage2FSDPPolicyWorker._save_replay_checkpoints(worker, str(component_dir))
    worker.replay_buffer.close(wait=True)
    worker.demo_buffer.close(wait=True)

    replay_path = component_dir / "replay_buffer" / "rank_3"
    demo_path = component_dir / "demo_buffer" / "rank_3"
    assert (replay_path / "metadata.json").is_file()
    assert (demo_path / "metadata.json").is_file()

    loaded = object.__new__(RLTStage2FSDPPolicyWorker)
    loaded._rank = 3
    loaded.replay_buffer = TrajectoryReplayBuffer(
        seed=21,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=False,
        auto_save_path=str(tmp_path / "loaded_replay"),
        trajectory_format="pt",
    )
    loaded.demo_buffer = TrajectoryReplayBuffer(
        seed=22,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=False,
        auto_save_path=str(tmp_path / "loaded_demo"),
        trajectory_format="pt",
    )

    RLTStage2FSDPPolicyWorker._load_replay_checkpoints(loaded, str(component_dir))

    assert loaded.replay_buffer.total_samples == 2
    assert loaded.demo_buffer.total_samples == 1
    loaded.replay_buffer.close(wait=True)
    loaded.demo_buffer.close(wait=True)


def test_inspect_rlt_replay_reads_standard_replay_directory(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=11,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    buffer.add_trajectories(replay_trajectories)
    buffer.close(wait=True)

    replay_dir = inspect_rlt_replay._resolve_replay_dir(tmp_path)
    metadata, trajectories = inspect_rlt_replay._load_replay_directory(replay_dir)
    summary = inspect_rlt_replay._inspect(metadata, trajectories)

    assert summary["num_trajectories"] == 2
    assert summary["total_samples"] == 2
    assert summary["inspected_samples"] == 2
    assert summary["source"] == {"RL": 1, "MIXED": 1}
    assert summary["source_chunk"]["histogram"] == {
        "RL": 2,
        "HUMAN": 1,
        "MIXED": 1,
    }
    assert summary["collection_phase"] == {"WARMUP": 1, "ONLINE": 1}
    assert summary["reward"]["positive_transition_rate"] == 0.5
    assert summary["intervention_flag_rate"] == 0.5


def test_hf_rollout_predict_kwargs_only_use_declared_model_context():
    class PlainKwargsModel:
        def predict_action_batch(self, env_obs, mode="train", **kwargs):
            del env_obs, mode
            return kwargs

    class ContextAwareModel:
        def predict_action_batch(
            self,
            env_obs,
            mode="train",
            env_infos=None,
            allow_expert=True,
        ):
            del env_obs, mode
            return {"env_infos": env_infos, "allow_expert": allow_expert}

    context = {
        "env_infos": {"policy_info": {"expert_takeover": torch.ones(1, 1)}},
        "allow_expert": False,
        "expert_model_getter": lambda: None,
    }
    assert (
        MultiStepRolloutWorker._filter_predict_kwargs(
            PlainKwargsModel().predict_action_batch,
            context,
        )
        == {}
    )
    filtered = MultiStepRolloutWorker._filter_predict_kwargs(
        ContextAwareModel().predict_action_batch,
        context,
    )
    assert set(filtered) == {"env_infos", "allow_expert"}
    assert filtered["allow_expert"] is False


def test_hf_rollout_predict_kwargs_can_pass_mode_without_model_type_hook():
    class ModeAwareModel:
        def predict_action_batch(self, env_obs, mode="train"):
            del env_obs
            return {"mode": mode}

    filtered = MultiStepRolloutWorker._filter_predict_kwargs(
        ModeAwareModel().predict_action_batch,
        {"mode": "eval", "env_infos": {}, "allow_expert": False},
    )
    assert filtered == {"mode": "eval"}


def test_env_worker_auto_reset_bootstrap_preserves_env_infos():
    worker = object.__new__(EnvWorker)
    worker.cfg = AttrDict(
        env=AttrDict(
            train=AttrDict(
                auto_reset=True,
            ),
        ),
        actor=AttrDict(
            model=AttrDict(
                num_action_chunks=2,
            ),
        ),
    )
    worker.train_num_envs_per_stage = 1
    worker.stage_num = 1
    worker.last_obs_list = [
        {
            "states": torch.zeros((1, 3), dtype=torch.float32),
            "main_images": None,
            "wrist_images": None,
            "extra_view_images": None,
            "task_descriptions": ["insert"],
        }
    ]
    worker.last_env_infos_list = [
        {
            "policy_info": {
                "in_critical_phase": torch.tensor([[False]]),
                "record_transition": torch.tensor([[False]]),
            }
        }
    ]
    worker.last_intervened_info_list = [(None, None)]
    worker._timer_metrics = {}
    worker._accelerator_type = Worker.accelerator_type

    env_outputs = EnvWorker.bootstrap_step(worker)

    assert len(env_outputs) == 1
    assert env_outputs[0].env_infos is not None
    policy_info = env_outputs[0].env_infos["policy_info"]
    assert policy_info["in_critical_phase"].shape == (1, 1)
    assert policy_info["record_transition"].shape == (1, 1)
    assert bool(policy_info["in_critical_phase"].item()) is False
    assert bool(policy_info["record_transition"].item()) is False


def test_env_worker_rlt_step_trace_obs_preserves_optional_none_keys():
    stacked = EnvWorker._stack_rlt_step_trace_obs(
        [
            {
                "states": torch.zeros((1, 3), dtype=torch.float32),
                "main_images": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
                "wrist_images": torch.ones((1, 4, 4, 3), dtype=torch.uint8),
                "extra_view_images": None,
                "task_descriptions": ["insert"],
            },
            {
                "states": torch.ones((1, 3), dtype=torch.float32),
                "main_images": torch.ones((1, 4, 4, 3), dtype=torch.uint8),
                "wrist_images": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
                "extra_view_images": None,
                "task_descriptions": ["insert"],
            },
        ]
    )

    assert stacked is not None
    assert stacked["states"].shape == (1, 2, 3)
    assert stacked["main_images"].shape == (1, 2, 4, 4, 3)
    assert stacked["wrist_images"].shape == (1, 2, 4, 4, 3)
    assert "extra_view_images" in stacked
    assert stacked["extra_view_images"] is None
    assert stacked["task_descriptions"] == ["insert"]


def test_env_worker_rlt_step_trace_infos_accepts_prestacked_policy_info():
    stacked = EnvWorker._stack_rlt_step_trace_infos(
        [
            {"policy_info": {"record_transition": torch.tensor([[False]])}},
            {
                "policy_info": {
                    "record_transition": torch.tensor([[False, True]]),
                    "in_critical_phase": torch.tensor([[False, True]]),
                }
            },
        ],
        expected_trace_len=2,
    )

    assert stacked is not None
    policy_info = stacked["policy_info"]
    assert policy_info["record_transition"].shape == (1, 2)
    assert policy_info["record_transition"].tolist() == [[False, True]]


def test_env_worker_rlt_step_trace_infos_stacks_per_step_policy_info():
    stacked = EnvWorker._stack_rlt_step_trace_infos(
        [
            {"policy_info": {"record_transition": torch.tensor([[False]])}},
            {"policy_info": {"record_transition": torch.tensor([[True]])}},
        ],
        expected_trace_len=2,
    )

    assert stacked is not None
    policy_info = stacked["policy_info"]
    assert policy_info["record_transition"].shape == (1, 2, 1)
    assert policy_info["record_transition"].reshape(1, 2).tolist() == [[False, True]]


class _FakeStudentModel:
    def __init__(self):
        self.base = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    def predict_action_batch(self, env_obs, **kwargs):
        del env_obs, kwargs
        actions = torch.tensor(
            [[[1.0, 1.0], [2.0, 2.0]]],
            dtype=torch.float32,
        )
        return actions, {
            "prev_logprobs": torch.zeros((1, 1), dtype=torch.float32),
            "prev_values": torch.zeros((1, 1), dtype=torch.float32),
            "forward_inputs": {
                "x": torch.ones((1, 3), dtype=torch.float32),
                "a_tilde": self.base.clone(),
            },
        }

    def rollout_state_dict(self):
        return {"student": torch.ones(1)}

    def encode_obs(self, obs):
        states = obs["states"].to(torch.float32)
        return states, torch.zeros((states.shape[0], 4), dtype=torch.float32)


class _FakeExpertModel:
    def predict_action_batch(self, env_obs, **kwargs):
        del env_obs, kwargs
        return torch.full((1, 2, 2), 9.0), {}


class _FakeRLTStage2Policy(RLTStage2Policy):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.chunk_length = 2
        self.action_dim = 2
        self.action_chunk_dim = 4
        self.global_step = 0
        self.online_gate_updates = 5
        self.intervention_enabled = True
        self.intervention_mode = "human_override"
        self.act_as_vla_reference = False
        self.replay_subsample_stride = 0

    def _prepare_features(self, env_obs):
        batch_size = int(env_obs["states"].shape[0])
        x = torch.ones((batch_size, 3), dtype=torch.float32)
        a_tilde = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float32,
        ).expand(batch_size, -1)
        processed_obs = {
            "tokenized_prompt": torch.ones((batch_size, 2), dtype=torch.int64),
            "tokenized_prompt_mask": torch.ones((batch_size, 2), dtype=torch.bool),
        }
        return x, a_tilde, processed_obs

    def actor_forward(
        self,
        x,
        a_tilde,
        *,
        deterministic=False,
        apply_ref_dropout=None,
        apply_action_noise=None,
    ):
        batch_size = int(x.shape[0])
        del x, a_tilde, deterministic, apply_ref_dropout, apply_action_noise
        return torch.tensor(
            [[1.0, 1.0, 2.0, 2.0]],
            dtype=torch.float32,
        ).expand(batch_size, -1)


def test_rlt_stage2_policy_predict_action_batch_owns_online_route():
    policy = _FakeRLTStage2Policy()
    expert = _FakeExpertModel()
    env_infos = {
        "policy_info": {
            "expert_takeover": torch.tensor([[True]]),
            "in_critical_phase": torch.tensor([[True]]),
            "record_transition": torch.tensor([[True]]),
            "intervention_phase": torch.tensor([[2.0]]),
        }
    }

    warmup_actions, warmup_result = policy.predict_action_batch(
        env_obs={"states": torch.zeros((1, 3))},
        mode="train",
        env_infos=env_infos,
        expert_model_getter=lambda: expert,
    )
    torch.testing.assert_close(
        warmup_actions,
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
    )
    assert bool(warmup_result["forward_inputs"]["ready_for_online"].item()) is False
    assert warmup_result["expert_label_flag"] is False

    policy.set_global_step(5)
    online_actions, online_result = policy.predict_action_batch(
        env_obs={"states": torch.zeros((1, 3))},
        mode="train",
        env_infos=env_infos,
        expert_model_getter=lambda: expert,
    )
    torch.testing.assert_close(online_actions, torch.full((1, 2, 2), 9.0))
    assert bool(online_result["forward_inputs"]["ready_for_online"].item()) is True
    assert online_result["expert_label_flag"] is True


def test_rlt_stage2_policy_emits_step_trace_schema_when_stride_enabled():
    policy = _FakeRLTStage2Policy()
    policy.replay_subsample_stride = 1
    env_infos = {
        "policy_info": {
            "expert_takeover": torch.tensor([[False]]),
            "in_critical_phase": torch.tensor([[True]]),
            "record_transition": torch.tensor([[True]]),
            "intervention_phase": torch.tensor([[0.0]]),
        },
        "rlt_step_trace_obs": {
            "states": torch.zeros((1, 2, 3), dtype=torch.float32),
            "task_descriptions": ["insert"],
        },
        "rlt_step_trace_infos": {
            "policy_info": {
                "record_transition": torch.tensor([[False, True]]),
            },
        },
    }

    _, result = policy.predict_action_batch(
        env_obs={"states": torch.zeros((1, 3))},
        mode="train",
        env_infos=env_infos,
    )

    forward_inputs = result["forward_inputs"]
    assert forward_inputs["rlt_step_trace"]["x"].shape == (1, 2, 3)
    assert forward_inputs["rlt_step_trace"]["a_tilde"].shape == (1, 2, 4)
    assert bool(forward_inputs["rlt_step_trace_valid"].item()) is True
    record_trace = forward_inputs["rlt_step_trace_infos"]["policy_info"][
        "record_transition"
    ]
    assert record_trace.shape == (1, 2, 1)
    assert record_trace.reshape(1, 2).tolist() == [[False, True]]


def _route(
    *,
    ready_for_online: bool,
    policy_info: dict[str, torch.Tensor],
    expert_model_getter,
):
    return route_rlt_stage2_rollout(
        env_obs={"states": torch.zeros((1, 3))},
        policy_info=policy_info,
        student_model=_FakeStudentModel(),
        expert_model_getter=expert_model_getter,
        model_kwargs={"mode": "train"},
        cfg=RLTStage2RolloutRouteConfig(
            ready_for_online=ready_for_online,
            online_gate_step=5,
            intervention_enabled=True,
            allow_expert=True,
            chunk_length=2,
            action_dim=2,
        ),
    )


def test_rlt_stage2_route_warmup_base_then_online_human_override():
    expert = _FakeExpertModel()
    policy_info = {
        "expert_takeover": torch.tensor([[True]]),
        "in_critical_phase": torch.tensor([[True]]),
        "record_transition": torch.tensor([[True]]),
        "intervention_phase": torch.tensor([[2.0]]),
    }

    warmup = _route(
        ready_for_online=False,
        policy_info=policy_info,
        expert_model_getter=lambda: expert,
    )
    torch.testing.assert_close(
        warmup.actions,
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
    )
    assert warmup.expert_label_flag is False
    assert bool(warmup.result["forward_inputs"]["ready_for_online"].item()) is False
    assert (
        warmup.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.BASE)
    ).all()

    online = _route(
        ready_for_online=True,
        policy_info=policy_info,
        expert_model_getter=lambda: expert,
    )
    torch.testing.assert_close(
        online.actions,
        torch.full((1, 2, 2), 9.0),
    )
    assert online.expert_label_flag is True
    assert bool(online.result["forward_inputs"]["ready_for_online"].item()) is True
    assert online.result["forward_inputs"]["intervention_flags"].all().item() is True
    assert (
        online.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.HUMAN)
    ).all()
    assert online.result["forward_inputs"]["student_control"].item() is True
    assert (
        online.result["forward_inputs"]["collection_phase_id"].item()
        == COLLECTION_PHASE_ONLINE
    )

    autonomous_online = _route(
        ready_for_online=True,
        policy_info={
            "expert_takeover": torch.tensor([[False]]),
            "in_critical_phase": torch.tensor([[True]]),
            "record_transition": torch.tensor([[True]]),
            "intervention_phase": torch.tensor([[0.0]]),
        },
        expert_model_getter=lambda: expert,
    )
    torch.testing.assert_close(
        autonomous_online.actions,
        torch.tensor([[[1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32),
    )
    assert autonomous_online.expert_label_flag is False
    assert autonomous_online.result["forward_inputs"]["student_control"].item() is True
    assert (
        autonomous_online.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.RL)
    ).all()

    online_before_critical_phase = _route(
        ready_for_online=True,
        policy_info={
            "expert_takeover": torch.tensor([[False]]),
            "in_critical_phase": torch.tensor([[False]]),
            "record_transition": torch.tensor([[False]]),
            "intervention_phase": torch.tensor([[0.0]]),
        },
        expert_model_getter=lambda: expert,
    )
    torch.testing.assert_close(
        online_before_critical_phase.actions,
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
    )
    assert (
        online_before_critical_phase.result["forward_inputs"][
            "student_control"
        ].item()
        is False
    )
    assert (
        online_before_critical_phase.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.BASE)
    ).all()
    assert (
        online_before_critical_phase.result["forward_inputs"][
            "record_transition"
        ].item()
        is False
    )
