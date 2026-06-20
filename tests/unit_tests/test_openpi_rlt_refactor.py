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

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlt.components import (
    DirectGaussianActor,
    TwinQCritic,
    actor_loss,
    compute_td_target,
)
from rlinf.models.embodiment.openpi_rlt.openpi_rlt_action_model import (
    build_openpi_rlt_backbone,
)
from rlinf.models.embodiment.openpi_rlt.proprio import (
    resolve_proprio_dim,
    select_proprio,
)
from rlinf.models.embodiment.openpi_rlt.rollout import (
    COLLECTION_PHASE_ONLINE,
    COLLECTION_PHASE_WARMUP,
    TransitionSource,
)
from rlinf.models.embodiment.openpi_rlt.schedule import (
    PHASE_ONLINE,
    PHASE_WARMUP,
    PHASE_WARMUP_WAIT_ONLINE,
    RLTStage2TrainingScheduler,
    SKIP_REASON_BUFFER_NOT_READY,
    SKIP_REASON_NO_PENDING_UPDATES,
    phase_id,
    resolve_rollout_phase,
    resolve_training_phase,
    write_status_json,
)
from rlinf.models.embodiment.openpi_rlt.stage1_policy import RLTStage1Policy
from rlinf.models.embodiment.openpi_rlt.stage2_policy import RLTStage2Policy
from rlinf.models.embodiment.openpi_rlt.trajectory_adapter import (
    RLTStage2TrajectoryReplayAdapter,
)
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _stage1_cfg():
    return OmegaConf.create(
        {
            "model_type": "rlt_stage1",
            "model_path": "unused",
            "precision": None,
            "load_to_device": False,
            "num_action_chunks": 10,
            "action_dim": 7,
            "is_lora": False,
            "rlt_stage1": {
                "config_name": "pi05_rlt_realworld_ee",
                "num_images_in_input": 2,
                "num_steps": 5,
                "embedding_dim": 4,
                "encoder_layers": 1,
                "encoder_heads": 2,
                "decoder_layers": 1,
                "decoder_heads": 2,
            },
        }
    )


def _stage2_cfg(*, proprio_dim: int | None = 5):
    cfg = OmegaConf.create(
        {
            "model_type": "rlt_stage2",
            "model_path": "unused",
            "precision": None,
            "load_to_device": False,
            "num_action_chunks": 2,
            "action_dim": 2,
            "num_steps": 5,
            "gamma": 0.5,
            "is_lora": False,
            "rlt_stage2": {
                "config_name": "pi05_rlt_joint",
                "num_images_in_input": 2,
                "embedding_dim": 4,
                "encoder_layers": 1,
                "encoder_heads": 2,
                "decoder_layers": 1,
                "decoder_heads": 2,
                "mlp_hidden_dim": 4,
                "mlp_num_hidden_layers": 0,
                "actor_noise_sigma": 0.0,
                "ref_action_dropout": 0.0,
                "load_feature_backbones": False,
                "load_rl_token_model": False,
                "online_gate_updates": 0,
                "intervention_enabled": False,
                "intervention_mode": "human_override",
            },
        }
    )
    if proprio_dim is not None:
        cfg.rlt_stage2.proprio_dim = proprio_dim
    return cfg


def _scheduler_cfg() -> AttrDict:
    return AttrDict(
        algorithm=AttrDict(
            warmup_post_collect_updates=2,
            replay_buffer=AttrDict(min_buffer_size=2),
            update_epoch=1,
        ),
        actor=AttrDict(model=AttrDict(rlt_stage2=AttrDict())),
    )


def _replay_cfg(*, replay_subsample_stride: int = 0) -> AttrDict:
    return AttrDict(
        actor=AttrDict(
            model=AttrDict(
                num_action_chunks=2,
                action_dim=2,
                rlt_stage2=AttrDict(
                    replay_subsample_stride=replay_subsample_stride,
                    replay_allow_terminal_partial=True,
                ),
            ),
        ),
        env=AttrDict(train=AttrDict(auto_reset=True)),
    )


def test_builtin_registry_dispatches_both_rlt_model_types_to_openpi_rlt(
    monkeypatch,
):
    import rlinf.models as model_registry
    import rlinf.models.embodiment.openpi_rlt.stage1_policy as stage1_module
    import rlinf.models.embodiment.openpi_rlt.stage2_policy as stage2_module

    class FakeStage1:
        def __init__(self, cfg, *, device="cuda"):
            self.cfg = cfg
            self.device = device

    class FakeStage2(FakeStage1):
        pass

    monkeypatch.setattr(stage1_module, "RLTStage1Policy", FakeStage1)
    monkeypatch.setattr(stage2_module, "RLTStage2Policy", FakeStage2)

    stage1 = model_registry.get_model(_stage1_cfg())
    stage2 = model_registry.get_model(_stage2_cfg())

    assert isinstance(stage1, FakeStage1)
    assert isinstance(stage2, FakeStage2)


def test_openpi_rlt_backbone_reuses_existing_openpi_loader(monkeypatch):
    import rlinf.models.embodiment.openpi_rlt.openpi_rlt_action_model as loader

    captured = {}

    class FakeOpenPI(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

    fake_model = FakeOpenPI()

    def fake_get_openpi_model(cfg, torch_dtype=None):
        captured["cfg"] = cfg
        captured["torch_dtype"] = torch_dtype
        return fake_model

    monkeypatch.setattr(loader, "get_openpi_model", fake_get_openpi_model)

    model = build_openpi_rlt_backbone(
        model_path="/tmp/openpi",
        config_name="pi05_rlt_joint",
        norm_stats_path="/tmp/norm_stats.json",
        num_images_in_input=2,
        num_action_chunks=10,
        action_dim=8,
        num_steps=5,
        device="cpu",
        freeze=True,
    )

    assert model is fake_model
    assert captured["torch_dtype"] is None
    cfg = captured["cfg"]
    assert cfg.model_type == "openpi"
    assert cfg.model_path == "/tmp/openpi"
    assert cfg.openpi.config_name == "pi05_rlt_joint"
    assert cfg.openpi.action_chunk == 10
    assert cfg.openpi.action_env_dim == 8
    assert cfg.openpi_data.norm_stats_path == "/tmp/norm_stats.json"
    assert model.training is False
    assert all(not param.requires_grad for param in model.parameters())


def test_stage1_policy_trains_only_rl_token_reconstruction(monkeypatch):
    calls = {}

    class FakeVLA:
        def extract_rlt_prefix_embeddings(self, observation, *, dtype=None):
            calls["observation"] = observation
            z = torch.arange(24, dtype=dtype or torch.float32).reshape(2, 3, 4)
            pad_mask = torch.ones((2, 3), dtype=torch.bool)
            return z, pad_mask

    def fake_build_openpi_rlt_backbone(**kwargs):
        calls["build_kwargs"] = kwargs
        return FakeVLA()

    monkeypatch.setattr(
        "rlinf.models.embodiment.openpi_rlt.stage1_policy."
        "build_openpi_rlt_backbone",
        fake_build_openpi_rlt_backbone,
    )

    policy = RLTStage1Policy(_stage1_cfg(), device="cpu")
    observation = object()
    output = policy(
        forward_type=ForwardType.SFT,
        data={
            "observation": observation,
            "actions": torch.zeros((2, 10, 7), dtype=torch.float32),
        },
    )

    assert calls["observation"] is observation
    assert calls["build_kwargs"]["freeze"] is True
    assert set(output) == {"loss", "l_ro", "z_rl", "z_hat"}
    assert "l_vla" not in output
    assert not hasattr(policy, "alpha")
    assert not hasattr(policy, "trainable_vla_parameters")
    assert output["loss"].requires_grad is True
    assert output["z_rl"].shape == (2, 4)


def test_vla_sft_worker_handles_rlt_stage1_openpi_batches():
    class FakeModel:
        def __call__(self, *, forward_type, data):
            assert forward_type == ForwardType.SFT
            assert set(data) == {"observation", "actions"}
            assert data["observation"]["state"].device.type == "cpu"
            assert data["actions"].dtype == torch.float32
            return {
                "loss": torch.tensor(3.0, requires_grad=True),
                "l_ro": torch.tensor(2.0),
            }

    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(model=SimpleNamespace(model_type="rlt_stage1"))
    )
    worker.device = torch.device("cpu")
    worker.amp_context = torch.autocast(device_type="cpu", enabled=False)
    worker.model = FakeModel()

    observation = {"state": torch.ones((2, 3), dtype=torch.float64)}
    actions = torch.ones((2, 10, 7), dtype=torch.float64)

    loss, metrics = FSDPVlaSftWorker.get_train_model_output(
        worker,
        (observation, actions),
    )

    assert loss.requires_grad is True
    assert metrics == {"loss": 3.0, "l_ro": 2.0}


def test_vla_sft_worker_exports_only_rlt_stage1_rl_token_checkpoint(
    tmp_path,
):
    class FakeStrategy:
        def get_model_state_dict(self, model, *, cpu_offload, full_state_dict):
            assert model == "wrapped-model"
            assert cpu_offload is True
            assert full_state_dict is True
            return {
                "rl_token_model.encoder.weight": torch.tensor([1.0]),
                "rl_token_model.decoder.bias": torch.tensor([2.0]),
                "vla.weight": torch.tensor([3.0]),
            }

    worker = object.__new__(FSDPVlaSftWorker)
    worker._rank = 0
    worker._strategy = FakeStrategy()
    worker.model = "wrapped-model"

    FSDPVlaSftWorker._save_rlt_stage1_rl_token_checkpoint(
        worker,
        str(tmp_path),
        step=7,
    )

    ckpt = torch.load(tmp_path / "rl_token" / "rl_token_model.pt")
    assert ckpt["step"] == 7
    assert set(ckpt["model_state_dict"]) == {
        "encoder.weight",
        "decoder.bias",
    }
    torch.testing.assert_close(
        ckpt["model_state_dict"]["encoder.weight"],
        torch.tensor([1.0]),
    )


def test_select_proprio_uses_real_state_prefix_from_model_state_tensor():
    state = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float64,
    )

    proprio = select_proprio(state, proprio_dim=2)

    assert proprio.dtype == torch.float32
    torch.testing.assert_close(proprio, state[:, :2].to(torch.float32))
    assert resolve_proprio_dim(19, unused="ignored") == 19

    with pytest.raises(ValueError, match="2D tensor"):
        select_proprio(torch.zeros((2, 3, 4)), proprio_dim=2)
    with pytest.raises(ValueError, match="enough real state dims"):
        select_proprio(torch.zeros((2, 1)), proprio_dim=2)


def test_stage2_policy_requires_explicit_full_state_proprio_dim():
    cfg = _stage2_cfg()
    policy = RLTStage2Policy(cfg, device="cpu")

    assert policy.proprio_dim == 5
    assert policy.state_dim == 9
    assert policy.vla is None
    assert policy.rl_token_model is None

    missing = _stage2_cfg(proprio_dim=None)
    with pytest.raises(ValueError, match="proprio_dim"):
        RLTStage2Policy(missing, device="cpu")


def test_stage2_feature_preparation_uses_openpi_rlt_methods_and_full_state():
    class FakeVLA:
        def __init__(self):
            self.calls = []

        def prepare_rlt_observation(self, env_obs):
            self.calls.append(("prepare", env_obs))
            observation = SimpleNamespace(state=env_obs["states"])
            processed = {
                "tokenized_prompt": env_obs["tokenized_prompt"],
                "tokenized_prompt_mask": env_obs["tokenized_prompt_mask"],
            }
            return observation, processed

        def extract_rlt_prefix_embeddings(self, observation, *, dtype=None):
            self.calls.append(("prefix", observation))
            batch_size = observation.state.shape[0]
            embeddings = torch.ones((batch_size, 3, 4), dtype=dtype)
            pad_mask = torch.ones((batch_size, 3), dtype=torch.bool)
            return embeddings, pad_mask

        def predict_rlt_reference_action(self, observation, chunk_length):
            self.calls.append(("reference", observation, chunk_length))
            batch_size = observation.state.shape[0]
            return torch.arange(
                batch_size * chunk_length * 2,
                dtype=torch.float32,
            ).reshape(batch_size, chunk_length, 2)

    class FakeRLToken:
        def encode(self, embeddings, pad_mask):
            del pad_mask
            return embeddings.mean(dim=1)

    policy = object.__new__(RLTStage2Policy)
    policy.device = torch.device("cpu")
    policy.vla = FakeVLA()
    policy.rl_token_model = FakeRLToken()
    policy.chunk_length = 2
    policy.action_dim = 2
    policy.action_chunk_dim = 4

    policy.proprio_dim = 5

    states = torch.arange(10, dtype=torch.float64).reshape(2, 5)
    env_obs = {
        "states": states,
        "tokenized_prompt": torch.ones((2, 4), dtype=torch.int64),
        "tokenized_prompt_mask": torch.ones((2, 4), dtype=torch.bool),
    }
    x, a_tilde, processed = policy._prepare_features(env_obs)

    assert [call[0] for call in policy.vla.calls] == [
        "prepare",
        "prefix",
        "reference",
    ]
    assert x.shape == (2, 9)
    torch.testing.assert_close(x[:, :4], torch.ones((2, 4)))
    torch.testing.assert_close(x[:, 4:], states.to(torch.float32))
    torch.testing.assert_close(
        a_tilde,
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )
    assert processed["tokenized_prompt"].shape == (2, 4)

    padded_states = torch.cat(
        [
            states.to(torch.float32),
            torch.full((2, 3), 99.0, dtype=torch.float32),
        ],
        dim=1,
    )
    env_obs["states"] = padded_states
    x_padded, _, _ = policy._prepare_features(env_obs)
    torch.testing.assert_close(x_padded[:, 4:], states.to(torch.float32))

    env_obs["states"] = torch.zeros((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="proprio dimension mismatch"):
        policy._prepare_features(env_obs)


def test_direct_gaussian_actor_conditions_on_reference_and_uses_fixed_noise():
    actor = DirectGaussianActor(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
        sigma=0.25,
        ref_dropout=1.0,
    )
    linear = actor.mlp.net[0]
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor([[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        )
        linear.bias.zero_()

    x = torch.tensor([[0.2]], dtype=torch.float32)
    a_tilde = torch.tensor([[0.3, 0.4]], dtype=torch.float32)

    deterministic = actor(x, a_tilde, deterministic=True)
    torch.testing.assert_close(deterministic, torch.tensor([[0.5, 0.2]]))

    dropped_ref = actor(x, a_tilde, deterministic=True, apply_ref_dropout=True)
    torch.testing.assert_close(dropped_ref, torch.tensor([[0.2, -0.2]]))

    torch.manual_seed(0)
    noisy = actor(x, a_tilde, deterministic=False)
    torch.manual_seed(0)
    expected_noise = deterministic + torch.randn_like(deterministic) * actor.sigma
    torch.testing.assert_close(noisy, expected_noise.clamp(-1.0, 1.0))


def test_compute_td_target_uses_chunk_discount_and_twin_target_minimum():
    actor = DirectGaussianActor(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
        sigma=0.0,
    )
    with torch.no_grad():
        actor.mlp.net[0].weight.copy_(
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        actor.mlp.net[0].bias.zero_()

    critic = TwinQCritic(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
    )
    with torch.no_grad():
        critic.q1_target.mlp.net[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
        critic.q1_target.mlp.net[0].bias.fill_(10.0)
        critic.q2_target.mlp.net[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
        critic.q2_target.mlp.net[0].bias.fill_(1.0)

    target = compute_td_target(
        rewards=torch.tensor([[1.0, 2.0]]),
        dones=torch.tensor([[0.0]]),
        next_x=torch.tensor([[0.5]]),
        next_a_tilde=torch.tensor([[0.25, 0.75]]),
        target_actor=actor,
        critic=critic,
        gamma=0.5,
        chunk_length=2,
    )

    next_q_min = 1.0 + 0.5 + 2.0 * 0.25 + 3.0 * 0.75
    expected = 1.0 + 0.5 * 2.0 + (0.5**2) * next_q_min
    torch.testing.assert_close(target, torch.tensor([[expected]]))


def test_actor_loss_uses_executed_human_actions_as_bc_target():
    q_value = torch.tensor([[0.0]])
    a = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    a_tilde = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    action_chunk = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    source_chunk = torch.tensor(
        [[TransitionSource.RL, TransitionSource.HUMAN]],
        dtype=torch.uint8,
    )

    total_loss, metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=1.0,
        q_weight=0.0,
        delta_weight=0.0,
    )

    torch.testing.assert_close(total_loss, torch.tensor(0.0))
    torch.testing.assert_close(metrics["bc_human_loss"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["human_mask_ratio"], torch.tensor(0.5))


def test_training_scheduler_tracks_warmup_budget_and_status_metrics(tmp_path):
    assert (
        resolve_training_phase(buffer_ready=False, ready_for_online=False)
        == PHASE_WARMUP
    )
    assert (
        resolve_training_phase(buffer_ready=True, ready_for_online=False)
        == PHASE_WARMUP_WAIT_ONLINE
    )
    assert (
        resolve_rollout_phase(ready_for_online=True, student_control_rate=0.5)
        == PHASE_ONLINE
    )

    status_path = tmp_path / "status" / "rlt_status.json"
    write_status_json(
        str(status_path),
        {"phase": PHASE_ONLINE, "phase_id": phase_id(PHASE_ONLINE)},
    )
    payload = json.loads(status_path.read_text())
    assert payload["phase_id"] == 2
    assert not status_path.with_suffix(".json.tmp").exists()

    scheduler = RLTStage2TrainingScheduler()
    cfg = _scheduler_cfg()
    not_ready = scheduler.plan(
        cfg,
        update_step=0,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 1.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 1.0,
            "total_episodes_added": 0.0,
        },
        global_min_replay_size=1,
        global_min_demo_size=0,
    )
    assert not_ready.schedule.skip_reason == SKIP_REASON_BUFFER_NOT_READY

    ready = scheduler.plan(
        cfg,
        update_step=0,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 2.0,
            "episodes_since_train": 1.0,
            "total_transitions_added": 2.0,
            "total_episodes_added": 1.0,
        },
        global_min_replay_size=2,
        global_min_demo_size=0,
    )
    assert ready.schedule.updates_to_run == 2
    scheduler.finish_updates(2)
    metrics = scheduler.metrics(
        plan=ready,
        update_step=2,
        global_counters={
            "transitions_since_train": 2.0,
            "episodes_since_train": 1.0,
            "total_transitions_added": 2.0,
            "total_episodes_added": 1.0,
        },
        global_min_replay_size=2,
        global_min_demo_size=0,
        should_train=True,
        skip_reason=0,
        critic_updates_run=2,
        actor_updates_run=1,
    )
    assert metrics["rlt_stage2/ready_for_online"] == 1.0
    assert metrics["rlt_stage2/pending_update_budget"] == 0.0

    no_pending = scheduler.plan(
        cfg,
        update_step=2,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 0.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 2.0,
            "total_episodes_added": 1.0,
        },
        global_min_replay_size=2,
        global_min_demo_size=0,
    )
    assert no_pending.schedule.skip_reason == SKIP_REASON_NO_PENDING_UPDATES


def _synthetic_rollout_trajectory() -> Trajectory:
    actions = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
        dtype=torch.float32,
    )
    rewards = torch.tensor([[[0.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
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
                [[[0.0, 0.1, 0.2]], [[1.0, 1.1, 1.2]], [[2.0, 2.1, 2.2]]],
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
                [[[False, False]], [[True, False]]],
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
                [[[COLLECTION_PHASE_WARMUP]], [[COLLECTION_PHASE_ONLINE]]],
                dtype=torch.uint8,
            ),
            "record_transition": torch.ones((2, 1, 1), dtype=torch.bool),
        },
    )


def test_trajectory_adapter_emits_standard_replay_trajectories():
    adapter = RLTStage2TrajectoryReplayAdapter(_replay_cfg())
    replay_trajectories, completed_episodes = adapter.build_replay_trajectories(
        _synthetic_rollout_trajectory()
    )

    assert completed_episodes == 1
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
        "dones",
        "source",
        "source_chunk",
        "collection_phase_id",
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
        torch.tensor([5.0, 6.0, 7.0, 8.0]),
    )


def test_trajectory_adapter_requires_step_trace_for_stride_replay():
    adapter = RLTStage2TrajectoryReplayAdapter(
        _replay_cfg(replay_subsample_stride=1),
    )
    with pytest.raises(RuntimeError, match="rlt_step_trace"):
        adapter.build_replay_trajectories(_synthetic_rollout_trajectory())
