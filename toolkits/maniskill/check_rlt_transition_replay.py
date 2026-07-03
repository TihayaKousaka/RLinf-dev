#!/usr/bin/env python3
"""Check ManiSkill RLT Stage2 transition replay and TD bootstrap semantics.

This script is intentionally read-only with respect to training code. It builds a
short local ManiSkill rollout, converts it through the current refactored
ManiSkill transition adapter, compares that conversion against the original
ablation-style indexing, and optionally probes the TD target chain with a loaded
actor checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOLKIT_DIR = Path(__file__).resolve().parent
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from check_rlt_stage2_proprio import (  # noqa: E402
    build_actor_model,
    build_env,
    build_feature_model,
    close_env,
    flatten_ref_chunk,
    last_obs_from_chunk,
    load_cfg,
    maybe_move_feature_model,
    prepare_reference_actions,
    resolve_device,
    shape_str,
    tensor_stats,
    to_float_tensor,
)
from rlinf.data.embodied_io_struct import (  # noqa: E402
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer  # noqa: E402
from rlinf.envs.action_utils import prepare_actions  # noqa: E402
from rlinf.models.embodiment.base_policy import ForwardType  # noqa: E402
from rlinf.workers.actor.rlt_sac_policy_worker import RLTSACFSDPPolicy  # noqa: E402


@dataclass
class ExpectedTransition:
    env_idx: int
    step_idx: int
    action: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    curr_obs: dict[str, torch.Tensor]
    next_obs: dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a short ManiSkill RLT rollout, compare current transition "
            "replay against ablation-style indexing, and probe TD targets."
        )
    )
    parser.add_argument(
        "--config-name",
        default="rlt_stage2_maniskill_joint",
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "examples" / "embodiment" / "config"),
        help="Hydra config directory.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=50)
    parser.add_argument(
        "--action-source",
        choices=("ref", "actor"),
        default="ref",
        help="Use OpenPI ref_chunk or the Stage2 actor to step the env.",
    )
    parser.add_argument(
        "--actor-ckpt",
        default=None,
        help=(
            "Optional actor checkpoint file or directory. Useful for checking "
            "whether q_next propagation is already healthy in a trained run."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for feature/actor models: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=512,
        help="Replay sample size for the sampled-batch TD probe.",
    )
    parser.add_argument(
        "--no-td-probe",
        action="store_true",
        help="Only check indexing/replay conversion; skip actor/Q probes.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if current replay differs from ablation-style indexing.",
    )
    return parser.parse_args()


def checkpoint_candidates(path: str) -> list[Path]:
    ckpt_path = Path(path)
    if ckpt_path.is_file():
        return [ckpt_path]
    return [
        ckpt_path / "model_state_dict" / "full_weights.pt",
        ckpt_path / "actor" / "model_state_dict" / "full_weights.pt",
    ]


def strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("_fsdp_wrapped_module.", "module.")
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        stripped[new_key] = value
    return stripped


def load_checkpoint_state_dict(ckpt: str) -> tuple[Path, dict[str, Any]]:
    for candidate in checkpoint_candidates(ckpt):
        if candidate.exists():
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported checkpoint object at {candidate}")
            return candidate, strip_state_dict_prefixes(state)
    raise FileNotFoundError(
        "Could not find actor checkpoint. Tried: "
        + ", ".join(str(path) for path in checkpoint_candidates(ckpt))
    )


def infer_rlt_head_type(state_dict: dict[str, Any]) -> str | None:
    keys = tuple(state_dict.keys())
    if any(key.startswith("direct_actor.") for key in keys):
        return "direct"
    if any(key.startswith("actor_mean.") for key in keys) or any(
        key.startswith("backbone.") for key in keys
    ):
        return "sac"
    return None


def configure_actor_from_checkpoint(cfg, actor_ckpt: str | None) -> tuple[Path | None, dict[str, Any] | None]:
    if not actor_ckpt:
        return None, None
    ckpt_path, state_dict = load_checkpoint_state_dict(actor_ckpt)
    checkpoint_head_type = infer_rlt_head_type(state_dict)
    configured_head_type = str(cfg.actor.model.get("rlt_head_type", "sac"))
    if checkpoint_head_type is not None and checkpoint_head_type != configured_head_type:
        with open_dict(cfg.actor.model):
            cfg.actor.model.rlt_head_type = checkpoint_head_type
        if "model" in cfg.rollout and "rlt_head_type" in cfg.rollout.model:
            with open_dict(cfg.rollout.model):
                cfg.rollout.model.rlt_head_type = checkpoint_head_type
    print("\n== Actor Checkpoint Inspect ==")
    print(f"path: {ckpt_path}")
    print(f"checkpoint_head_type: {checkpoint_head_type}")
    print(f"configured_head_type_before: {configured_head_type}")
    print(f"effective_actor_head_type: {cfg.actor.model.get('rlt_head_type', 'sac')}")
    return ckpt_path, state_dict


def critical_key(key: str) -> bool:
    return key.startswith(
        (
            "direct_actor.",
            "backbone.",
            "actor_mean.",
            "actor_logstd.",
            "q_head.",
        )
    )


def load_actor_checkpoint_checked(
    actor_model,
    ckpt_path: Path | None,
    state_dict: dict[str, Any] | None,
) -> None:
    if ckpt_path is None or state_dict is None:
        print("\n== Actor Checkpoint ==")
        print("No --actor-ckpt provided; checking randomly initialized actor head.")
        return

    missing, unexpected = actor_model.load_state_dict(state_dict, strict=False)
    critical_missing = [key for key in missing if critical_key(key)]
    critical_unexpected = [key for key in unexpected if critical_key(key)]
    print("\n== Actor Checkpoint Load ==")
    print(f"loaded: {ckpt_path}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    print(f"critical_missing_keys: {len(critical_missing)}")
    print(f"critical_unexpected_keys: {len(critical_unexpected)}")
    if missing:
        print(f"missing_keys_sample: {list(missing)[:8]}")
    if unexpected:
        print(f"unexpected_keys_sample: {list(unexpected)[:8]}")
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Actor checkpoint did not load key Stage2 actor/Q parameters. "
            "The TD probe would be invalid. "
            f"critical_missing_sample={critical_missing[:8]}, "
            f"critical_unexpected_sample={critical_unexpected[:8]}"
        )


def clone_rlt_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in obs.items()
        if key in {"z_rl", "proprio", "ref_chunk"}
    }


def make_zero_done(num_envs: int, chunk_len: int) -> torch.Tensor:
    return torch.zeros((num_envs, chunk_len), dtype=torch.bool)


def make_forward_inputs(
    rlt_obs: dict[str, torch.Tensor],
    action_flat: torch.Tensor,
    *,
    chunk_len: int,
    transition_obs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    batch_size = int(action_flat.shape[0])
    transition_obs = rlt_obs if transition_obs is None else transition_obs
    forward_inputs = clone_rlt_obs(rlt_obs)
    forward_inputs["action"] = action_flat.detach().cpu().contiguous()
    forward_inputs["intervention_flags"] = torch.zeros(
        (batch_size, chunk_len), dtype=torch.bool
    )
    forward_inputs["student_control"] = torch.zeros((batch_size, 1), dtype=torch.bool)
    forward_inputs["intervention_requested"] = torch.zeros(
        (batch_size, 1), dtype=torch.bool
    )
    forward_inputs["in_critical_phase"] = torch.ones((batch_size, 1), dtype=torch.bool)
    forward_inputs["record_transition"] = torch.ones((batch_size, 1), dtype=torch.bool)
    forward_inputs["ready_for_online"] = torch.ones((batch_size, 1), dtype=torch.bool)
    for key, value in transition_obs.items():
        if key in {"z_rl", "proprio", "ref_chunk"}:
            forward_inputs[f"rlt_transition_{key}"] = (
                value.detach().cpu().contiguous().clone()
            )
    return forward_inputs


def actor_raw_actions(
    cfg,
    actor_model,
    rlt_obs: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    obs_device = {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in rlt_obs.items()
    }
    with torch.no_grad():
        action_flat, _log_pi, _ = actor_model(
            forward_type=ForwardType.SAC,
            obs=obs_device,
            deterministic=True,
            action_noise_sigma=0.0,
        )
    raw_chunk = action_flat.reshape(-1, chunk_len, action_dim).detach().cpu()
    return raw_chunk, action_flat.detach().cpu()


def ref_raw_actions(cfg, rlt_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    ref_chunk = rlt_obs["ref_chunk"].detach().cpu()
    if ref_chunk.dim() == 2:
        raw_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)[
            :, :chunk_len, :action_dim
        ]
    else:
        raw_chunk = ref_chunk[:, :chunk_len, :action_dim]
    return raw_chunk.contiguous(), raw_chunk.reshape(raw_chunk.shape[0], -1).contiguous()


def prepare_raw_chunk_actions(cfg, raw_chunk: torch.Tensor) -> torch.Tensor:
    return prepare_actions(
        raw_chunk_actions=raw_chunk,
        env_type=cfg.env.train.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=int(cfg.rlt.num_action_chunks),
        action_dim=int(cfg.rlt.action_dim),
        policy=cfg.actor.model.get("policy_setup", None),
        wm_env_type=cfg.env.train.get("wm_env_type", None),
    )


def update_pending_transition(
    rollout_result: EmbodiedRolloutResult,
    pending_obs: dict[str, torch.Tensor] | None,
    forward_inputs: dict[str, torch.Tensor],
    *,
    cache_current: bool,
) -> dict[str, torch.Tensor] | None:
    if pending_obs is not None:
        next_obs = {
            "z_rl": forward_inputs["rlt_transition_z_rl"],
            "proprio": forward_inputs["rlt_transition_proprio"],
            "ref_chunk": forward_inputs["rlt_transition_ref_chunk"],
        }
        rollout_result.append_transitions(pending_obs, next_obs)
        pending_obs = None
    if cache_current:
        pending_obs = {
            "z_rl": forward_inputs["z_rl"],
            "proprio": forward_inputs["proprio"],
            "ref_chunk": forward_inputs["ref_chunk"],
        }
    return pending_obs


@torch.no_grad()
def build_local_rollout_trajectory(
    cfg,
    feature_model,
    actor_model,
    *,
    num_envs: int,
    num_chunks: int,
    action_source: str,
    device: torch.device,
) -> Trajectory:
    chunk_len = int(cfg.rlt.num_action_chunks)
    env = build_env(cfg, num_envs)
    rollout = EmbodiedRolloutResult(max_episode_length=cfg.env.train.max_episode_steps)
    pending_obs = None
    try:
        obs, _infos = env.reset()
        env_output = SimpleNamespace(
            obs=obs,
            rewards=None,
            dones=make_zero_done(num_envs, chunk_len),
            terminations=make_zero_done(num_envs, chunk_len),
            truncations=make_zero_done(num_envs, chunk_len),
        )
        for _step_idx in range(num_chunks):
            rlt_obs = feature_model.extract_rlt_stage2_obs(env_output.obs)
            if action_source == "actor":
                raw_chunk, action_flat = actor_raw_actions(
                    cfg,
                    actor_model,
                    rlt_obs,
                    device=device,
                )
            else:
                raw_chunk, action_flat = ref_raw_actions(cfg, rlt_obs)

            forward_inputs = make_forward_inputs(
                rlt_obs,
                action_flat,
                chunk_len=chunk_len,
            )
            pending_obs = update_pending_transition(
                rollout,
                pending_obs,
                forward_inputs,
                cache_current=True,
            )
            rollout.append_step_result(
                ChunkStepResult(
                    actions=forward_inputs["action"],
                    forward_inputs=forward_inputs,
                    dones=env_output.dones,
                    terminations=env_output.terminations,
                    truncations=env_output.truncations,
                    rewards=env_output.rewards,
                    versions=torch.zeros((num_envs, 1), dtype=torch.float32),
                )
            )

            prepared_actions = prepare_raw_chunk_actions(cfg, raw_chunk)
            obs_list, rewards, terminations, truncations, _infos_list = env.chunk_step(
                prepared_actions
            )
            next_obs = last_obs_from_chunk(obs_list)
            dones = torch.logical_or(
                torch.as_tensor(terminations).bool(),
                torch.as_tensor(truncations).bool(),
            )
            env_output = SimpleNamespace(
                obs=next_obs,
                rewards=torch.as_tensor(rewards).detach().cpu().contiguous(),
                dones=dones.detach().cpu().contiguous(),
                terminations=torch.as_tensor(terminations).detach().cpu().contiguous(),
                truncations=torch.as_tensor(truncations).detach().cpu().contiguous(),
            )

        final_rlt_obs = feature_model.extract_rlt_stage2_obs(env_output.obs)
        final_raw_chunk, final_action_flat = ref_raw_actions(cfg, final_rlt_obs)
        del final_raw_chunk
        final_forward_inputs = make_forward_inputs(
            final_rlt_obs,
            final_action_flat,
            chunk_len=chunk_len,
        )
        pending_obs = update_pending_transition(
            rollout,
            pending_obs,
            final_forward_inputs,
            cache_current=False,
        )
        assert pending_obs is None
        rollout.append_step_result(
            ChunkStepResult(
                dones=env_output.dones,
                terminations=env_output.terminations,
                truncations=env_output.truncations,
                rewards=env_output.rewards,
            )
        )
    finally:
        close_env(env)

    return rollout.to_trajectory()


def build_current_replay_transitions(cfg, traj: Trajectory) -> tuple[list[Trajectory], int]:
    replay_cfg = cfg.algorithm.replay_buffer
    worker = object.__new__(RLTSACFSDPPolicy)
    worker.cfg = cfg
    worker.replay_buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=int(replay_cfg.get("cache_size", 10000)),
        sample_window_size=int(replay_cfg.get("sample_window_size", 50000)),
        max_num_samples=int(replay_cfg.get("max_num_samples", 50000)),
        auto_save=False,
    )
    return RLTSACFSDPPolicy._maniskill_transition_replay_trajectories(worker, traj)


def row_tensor(tensor: torch.Tensor, step_idx: int, env_idx: int) -> torch.Tensor:
    return (
        tensor[step_idx, env_idx]
        .detach()
        .clone()
        .unsqueeze(0)
        .unsqueeze(0)
        .cpu()
        .contiguous()
    )


def row_obs(obs_dict: dict[str, torch.Tensor], step_idx: int, env_idx: int) -> dict[str, torch.Tensor]:
    return {
        key: row_tensor(value, step_idx, env_idx)
        for key, value in obs_dict.items()
        if key in {"z_rl", "proprio", "ref_chunk"}
    }


def build_ablation_style_expected(cfg, traj: Trajectory) -> tuple[list[ExpectedTransition], int]:
    del cfg
    if traj.actions is None or traj.rewards is None or not traj.forward_inputs:
        return [], 0
    traj_len = int(traj.actions.shape[0])
    bsz = int(traj.actions.shape[1])
    auto_reset = False
    expected: list[ExpectedTransition] = []
    completed = 0
    record = traj.forward_inputs.get("record_transition")
    for env_idx in range(bsz):
        for step_idx in range(traj_len):
            if isinstance(record, torch.Tensor):
                should_record = bool(
                    record[step_idx, env_idx].detach().to(torch.bool).reshape(-1).all()
                )
                if not should_record:
                    continue
            done_idx = min(step_idx + 1, int(traj.dones.shape[0]) - 1)
            done = row_tensor(traj.dones, done_idx, env_idx)
            is_done = bool(done.reshape(-1).to(torch.bool).any())
            curr_obs = row_obs(traj.forward_inputs, step_idx, env_idx)
            if is_done:
                next_obs = curr_obs
            elif step_idx + 1 < traj_len:
                next_obs = row_obs(traj.forward_inputs, step_idx + 1, env_idx)
            else:
                next_obs = row_obs(traj.next_obs, step_idx, env_idx)
            expected.append(
                ExpectedTransition(
                    env_idx=env_idx,
                    step_idx=step_idx,
                    action=row_tensor(traj.actions, step_idx, env_idx),
                    rewards=row_tensor(traj.rewards, step_idx, env_idx),
                    dones=done,
                    terminations=row_tensor(traj.terminations, done_idx, env_idx),
                    truncations=row_tensor(traj.truncations, done_idx, env_idx),
                    curr_obs=curr_obs,
                    next_obs=next_obs,
                )
            )
            if is_done:
                completed += 1
                if not auto_reset:
                    break
    return expected, completed


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_t = to_float_tensor(lhs)
    rhs_t = to_float_tensor(rhs, device=lhs_t.device)
    if lhs_t.shape != rhs_t.shape:
        return float("inf")
    return float((lhs_t - rhs_t).abs().max().item()) if lhs_t.numel() else 0.0


def compare_obs_dicts(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> float:
    worst = 0.0
    for key in ("z_rl", "proprio", "ref_chunk"):
        if key not in lhs or key not in rhs:
            return float("inf")
        worst = max(worst, max_abs(lhs[key], rhs[key]))
    return worst


def compare_replay_to_expected(
    transitions: list[Trajectory],
    expected: list[ExpectedTransition],
) -> dict[str, float]:
    metrics = {
        "count_match": float(len(transitions) == len(expected)),
        "action_max_abs": 0.0,
        "reward_max_abs": 0.0,
        "done_max_abs": 0.0,
        "termination_max_abs": 0.0,
        "truncation_max_abs": 0.0,
        "curr_obs_max_abs": 0.0,
        "next_obs_max_abs": 0.0,
    }
    for transition, exp in zip(transitions, expected):
        metrics["action_max_abs"] = max(
            metrics["action_max_abs"], max_abs(transition.actions, exp.action)
        )
        metrics["reward_max_abs"] = max(
            metrics["reward_max_abs"], max_abs(transition.rewards, exp.rewards)
        )
        metrics["done_max_abs"] = max(
            metrics["done_max_abs"], max_abs(transition.dones, exp.dones)
        )
        metrics["termination_max_abs"] = max(
            metrics["termination_max_abs"],
            max_abs(transition.terminations, exp.terminations),
        )
        metrics["truncation_max_abs"] = max(
            metrics["truncation_max_abs"],
            max_abs(transition.truncations, exp.truncations),
        )
        metrics["curr_obs_max_abs"] = max(
            metrics["curr_obs_max_abs"],
            compare_obs_dicts(transition.curr_obs, exp.curr_obs),
        )
        metrics["next_obs_max_abs"] = max(
            metrics["next_obs_max_abs"],
            compare_obs_dicts(transition.next_obs, exp.next_obs),
        )
    return metrics


def flatten_transition_batch(transitions: list[Trajectory]) -> dict[str, Any]:
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=max(len(transitions), 1),
        sample_window_size=max(len(transitions), 1),
        auto_save=False,
    )
    flats = [buffer._flatten_trajectory(transition) for transition in transitions]
    return buffer._concat_flat_trajectories(flats)


def sample_transition_batch(
    transitions: list[Trajectory],
    *,
    batch_size: int,
) -> dict[str, Any]:
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=max(len(transitions), 1),
        sample_window_size=max(len(transitions), 1),
        auto_save=False,
    )
    buffer.add_trajectories(transitions)
    return buffer.sample_chunks(min(batch_size, buffer.total_samples))


def move_obs(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in obs.items()
        if key in {"z_rl", "proprio", "ref_chunk"}
    }


def q_min(actor_model, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
    q_all = actor_model(forward_type=ForwardType.SAC_Q, obs=obs, actions=actions)
    return q_all.min(dim=-1, keepdim=True).values


def action_from_mode(
    cfg,
    actor_model,
    obs: dict[str, torch.Tensor],
    *,
    mode_key: str,
) -> torch.Tensor:
    mode = str(cfg.algorithm.get("rlt_action_sampling", {}).get(mode_key, "deterministic"))
    kwargs: dict[str, Any]
    if mode == "td3_action_noise":
        sampling_cfg = cfg.algorithm.get("rlt_action_sampling", {})
        kwargs = {
            "deterministic": True,
            "action_noise_sigma": float(
                sampling_cfg.get(
                    "target_noise_sigma"
                    if mode_key == "target_action_mode"
                    else "action_noise_sigma",
                    sampling_cfg.get("action_noise_sigma", 0.0),
                )
            ),
            "action_noise_clip": float(
                sampling_cfg.get(
                    "target_noise_clip"
                    if mode_key == "target_action_mode"
                    else "action_noise_clip",
                    sampling_cfg.get("action_noise_clip", 0.0),
                )
            ),
        }
    elif mode == "sac_sample":
        kwargs = {"deterministic": False}
    elif mode == "deterministic":
        kwargs = {"deterministic": True}
    else:
        raise ValueError(f"Unsupported action mode: {mode}")
    actions, _log_pi, _ = actor_model(forward_type=ForwardType.SAC, obs=obs, **kwargs)
    return actions


@torch.no_grad()
def td_probe(
    cfg,
    actor_model,
    batch: dict[str, Any],
    expected: list[ExpectedTransition] | None,
    *,
    label: str,
    device: torch.device,
) -> None:
    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    gamma = float(cfg.algorithm.gamma)
    curr_obs = move_obs(batch["curr_obs"], device)
    next_obs = move_obs(batch["next_obs"], device)
    actions = batch["actions"].detach().to(device=device, dtype=torch.float32)
    rewards = batch["rewards"].detach().to(device=device, dtype=torch.float32)
    dones = batch["dones"].detach().to(device=device)
    rewards_flat = rewards.reshape(rewards.shape[0], -1)
    dones_flat = dones.reshape(dones.shape[0], -1)
    discounts = torch.pow(
        torch.as_tensor(gamma, device=device, dtype=rewards_flat.dtype),
        torch.arange(rewards_flat.shape[-1], device=device, dtype=rewards_flat.dtype),
    )
    reward_target = torch.sum(rewards_flat * discounts, dim=-1, keepdim=True)
    not_done = ~dones_flat.bool().any(dim=-1, keepdim=True)
    next_actor_action = action_from_mode(
        cfg,
        actor_model,
        next_obs,
        mode_key="target_action_mode",
    )
    q_next_actor = q_min(actor_model, next_obs, next_actor_action)
    next_ref = flatten_ref_chunk(
        next_obs["ref_chunk"],
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    q_next_ref = q_min(actor_model, next_obs, next_ref)
    q_data = q_min(actor_model, curr_obs, actions)
    q_target = reward_target + not_done.to(reward_target.dtype) * (
        gamma**rewards_flat.shape[-1]
    ) * q_next_actor

    print(f"\n== TD Probe: {label} ==")
    print(f"batch_size={actions.shape[0]}")
    print(f"actions: {shape_str(actions)}")
    print(f"rewards: {shape_str(rewards)}")
    print(f"dones: {shape_str(dones)}")
    tensor_stats("reward_target", reward_target)
    tensor_stats("q_data", q_data)
    tensor_stats("q_next_actor", q_next_actor)
    tensor_stats("q_next_ref", q_next_ref)
    tensor_stats("q_target", q_target)
    print(f"reward_target_nonzero_rate={reward_target.ne(0).float().mean().item():.8g}")
    print(f"done_rate={dones_flat.bool().any(dim=-1).float().mean().item():.8g}")
    print(f"not_done_mean={not_done.float().mean().item():.8g}")
    print(f"bootstrap_discount={gamma**rewards_flat.shape[-1]:.8g}")

    positive_mask = reward_target.reshape(-1).ne(0)
    if positive_mask.any():
        print("\n-- positive reward chunks --")
        tensor_stats("positive/q_data", q_data[positive_mask])
        tensor_stats("positive/q_next_actor", q_next_actor[positive_mask])
        tensor_stats("positive/q_next_ref", q_next_ref[positive_mask])
        tensor_stats("positive/q_target", q_target[positive_mask])

    if expected is None or len(expected) != actions.shape[0]:
        return
    predecessor_indices = []
    positive_by_env_step = {
        (row.env_idx, row.step_idx)
        for row, is_pos in zip(expected, positive_mask.detach().cpu().tolist())
        if is_pos
    }
    for idx, row in enumerate(expected):
        if (row.env_idx, row.step_idx + 1) in positive_by_env_step:
            predecessor_indices.append(idx)
    if predecessor_indices:
        pred_idx = torch.as_tensor(predecessor_indices, device=device, dtype=torch.long)
        print("\n-- predecessor of positive reward chunks --")
        tensor_stats("pre_positive/q_data", q_data.index_select(0, pred_idx))
        tensor_stats(
            "pre_positive/q_next_actor", q_next_actor.index_select(0, pred_idx)
        )
        tensor_stats("pre_positive/q_next_ref", q_next_ref.index_select(0, pred_idx))
        tensor_stats("pre_positive/q_target", q_target.index_select(0, pred_idx))
        print(f"pre_positive_count={len(predecessor_indices)}")


def print_rollout_summary(traj: Trajectory, expected: list[ExpectedTransition]) -> None:
    print("\n== Rollout Tensor Summary ==")
    for name in (
        "actions",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "forward_inputs",
        "curr_obs",
        "next_obs",
    ):
        value = getattr(traj, name)
        if isinstance(value, dict):
            shapes = {key: tuple(val.shape) for key, val in value.items()}
            print(f"{name}: {shapes}")
        else:
            print(f"{name}: {shape_str(value)}")
    positive = [
        row for row in expected if row.rewards.reshape(-1).float().sum().item() != 0.0
    ]
    done_rows = [row for row in expected if row.dones.reshape(-1).bool().any()]
    print("\n== Expected Transition Summary ==")
    print(f"expected_count={len(expected)}")
    print(f"positive_reward_count={len(positive)}")
    print(f"done_count={len(done_rows)}")
    for row in positive[:8]:
        reward_sum = row.rewards.reshape(-1).float().sum().item()
        done_any = bool(row.dones.reshape(-1).bool().any())
        done_substep = (
            int(row.dones.reshape(-1).to(torch.int64).argmax().item())
            if done_any
            else -1
        )
        print(
            "positive_row "
            f"env={row.env_idx} step={row.step_idx} "
            f"reward_sum={reward_sum:.6g} done={done_any} done_substep={done_substep}"
        )


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)
    cfg = load_cfg(args.config_name, args.config_dir)
    device = resolve_device(args.device)
    ckpt_path, actor_state_dict = configure_actor_from_checkpoint(
        cfg,
        args.actor_ckpt,
    )

    feature_model = build_feature_model(cfg)
    feature_model = maybe_move_feature_model(feature_model, device)

    actor_model = None
    if args.action_source == "actor" or not args.no_td_probe:
        actor_model = build_actor_model(cfg, device)
        load_actor_checkpoint_checked(actor_model, ckpt_path, actor_state_dict)
        actor_model.eval()

    print("== Config Summary ==")
    print(f"config_name={args.config_name}")
    print(f"num_envs={args.num_envs}")
    print(f"num_chunks={args.num_chunks}")
    print(f"action_source={args.action_source}")
    print(f"actor_head_type={cfg.actor.model.get('rlt_head_type', 'sac')}")
    print(f"actor_update_mode={cfg.algorithm.rlt_action_sampling.get('actor_update_mode')}")
    print(f"target_action_mode={cfg.algorithm.rlt_action_sampling.get('target_action_mode')}")
    print(f"gamma={cfg.algorithm.gamma}")
    print(f"num_action_chunks={cfg.rlt.num_action_chunks}")

    traj = build_local_rollout_trajectory(
        cfg,
        feature_model,
        actor_model,
        num_envs=args.num_envs,
        num_chunks=args.num_chunks,
        action_source=args.action_source,
        device=device,
    )
    current_transitions, current_completed = build_current_replay_transitions(cfg, traj)
    expected, expected_completed = build_ablation_style_expected(cfg, traj)

    print_rollout_summary(traj, expected)

    print("\n== Current Adapter vs Ablation-Style Indexing ==")
    print(f"current_transition_count={len(current_transitions)}")
    print(f"current_completed_episodes={current_completed}")
    print(f"expected_transition_count={len(expected)}")
    print(f"expected_completed_episodes={expected_completed}")
    compare = compare_replay_to_expected(current_transitions, expected)
    for key, value in compare.items():
        print(f"{key}={value:.8g}")
    mismatch = (
        len(current_transitions) != len(expected)
        or any(
            value != 0.0
            for key, value in compare.items()
            if key != "count_match"
        )
    )
    if mismatch:
        print("VERDICT: FAIL, current replay differs from ablation-style indexing.")
    else:
        print(
            "VERDICT: PASS, current replay matches ablation-style chunk-transition "
            "indexing on this rollout."
        )

    if args.no_td_probe:
        if mismatch and args.fail_on_mismatch:
            raise SystemExit(1)
        return

    assert actor_model is not None
    all_batch = flatten_transition_batch(current_transitions)
    td_probe(
        cfg,
        actor_model,
        all_batch,
        expected,
        label="all converted transitions",
        device=device,
    )
    sample_batch = sample_transition_batch(
        current_transitions,
        batch_size=args.sample_batch_size,
    )
    td_probe(
        cfg,
        actor_model,
        sample_batch,
        None,
        label=f"replay sample batch size {args.sample_batch_size}",
        device=device,
    )

    if mismatch and args.fail_on_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
