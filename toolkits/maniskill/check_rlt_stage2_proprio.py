#!/usr/bin/env python3
"""Check ManiSkill RLT Stage2 proprio against OpenPI's processed state.

This is a cheap data-flow check for the current Stage2 path. It verifies whether
the `proprio` stored by `extract_rlt_stage2_obs()` is numerically equivalent to
the state that OpenPI actually consumes after its input transforms.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from openpi.models import model as openpi_model


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ManiSkill raw/configured states with OpenPI's transformed "
            "observation.state for RLT Stage2."
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
    parser.add_argument(
        "--obs-path",
        default=None,
        help=(
            "Optional torch-saved env obs dict. If omitted, the script creates a "
            "small ManiSkill env and calls reset()."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="Number of ManiSkill envs to create when --obs-path is omitted.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Max absolute difference threshold for reporting PASS/FAIL.",
    )
    parser.add_argument(
        "--run-extract",
        action="store_true",
        help=(
            "Also run model.extract_rlt_stage2_obs(). This is slower because it "
            "runs the RLT prefix path and reference action sampling."
        ),
    )
    parser.add_argument(
        "--check-next",
        action="store_true",
        help=(
            "Create a ManiSkill env, step once with the current reference chunk, "
            "and compare current vs next extracted RLT features."
        ),
    )
    parser.add_argument(
        "--check-direct-head",
        action="store_true",
        help=(
            "Run a cheap RLT MLP actor/critic integration check on extracted "
            "Stage2 features. This reports direct-head action scale, clamp "
            "saturation, BC gradient flow, Q outputs, and one-step TD target "
            "when an env can be stepped."
        ),
    )
    parser.add_argument(
        "--check-ref-rollout",
        action="store_true",
        help=(
            "Run full episodes using only the Stage2 OpenPI ref_chunk path. "
            "This checks whether extract_rlt_stage2_obs -> ref_chunk -> "
            "prepare_actions can still solve ManiSkill before any RL update."
        ),
    )
    parser.add_argument(
        "--max-rollout-chunks",
        type=int,
        default=50,
        help=(
            "Maximum chunks for --check-ref-rollout. With num_action_chunks=10 "
            "and max_episode_steps=500, 50 chunks covers a full episode."
        ),
    )
    parser.add_argument(
        "--actor-ckpt",
        default=None,
        help=(
            "Optional actor checkpoint file or directory to load before "
            "--check-direct-head. Directories may contain model_state_dict/"
            "full_weights.pt or actor/model_state_dict/full_weights.pt."
        ),
    )
    parser.add_argument(
        "--skip-step",
        action="store_true",
        help=(
            "With --check-direct-head, skip env.chunk_step and only check the "
            "current-observation actor/critic path. Useful with --obs-path."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for the RLT MLP check: auto, cpu, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


def load_cfg(config_name: str, config_dir: str) -> DictConfig:
    os.environ.setdefault("EMBODIED_PATH", str(REPO_ROOT / "examples" / "embodiment"))
    with initialize_config_dir(version_base="1.1", config_dir=str(Path(config_dir))):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def build_feature_model(cfg: DictConfig):
    from rlinf.models import get_model

    model_cfg = cfg.rollout.rlt_feature_model
    with open_dict(model_cfg):
        model_cfg.load_to_device = False
    model = get_model(model_cfg)
    if model is None:
        raise RuntimeError(f"Failed to build model_type={model_cfg.model_type!r}.")
    model.eval()
    return model


def maybe_move_feature_model(model, device: torch.device):
    if device.type == "cuda":
        model.to(device)
    return model


def build_actor_model(cfg: DictConfig, device: torch.device):
    from rlinf.models import get_model

    actor_model = get_model(cfg.actor.model)
    if actor_model is None:
        raise RuntimeError(
            f"Failed to build actor model_type={cfg.actor.model.model_type!r}."
        )
    actor_model.to(device)
    actor_model.eval()
    return actor_model


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _checkpoint_candidates(path: str) -> list[Path]:
    ckpt_path = Path(path)
    if ckpt_path.is_file():
        return [ckpt_path]
    return [
        ckpt_path / "model_state_dict" / "full_weights.pt",
        ckpt_path / "actor" / "model_state_dict" / "full_weights.pt",
    ]


def _strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
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


def load_actor_checkpoint(actor_model, ckpt: str | None) -> None:
    if not ckpt:
        print("\n== Actor Checkpoint ==")
        print("No --actor-ckpt provided; checking randomly initialized actor head.")
        return

    for candidate in _checkpoint_candidates(ckpt):
        if candidate.exists():
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported actor checkpoint object: {candidate}")
            state = _strip_state_dict_prefixes(state)
            missing, unexpected = actor_model.load_state_dict(state, strict=False)
            print("\n== Actor Checkpoint ==")
            print(f"loaded: {candidate}")
            print(f"missing_keys: {len(missing)}")
            print(f"unexpected_keys: {len(unexpected)}")
            if missing:
                print(f"missing_keys_sample: {list(missing)[:8]}")
            if unexpected:
                print(f"unexpected_keys_sample: {list(unexpected)[:8]}")
            return
    raise FileNotFoundError(
        "Could not find actor checkpoint. Tried: "
        + ", ".join(str(path) for path in _checkpoint_candidates(ckpt))
    )


def build_env(cfg: DictConfig, num_envs: int):
    from rlinf.envs import get_env_cls

    env_cfg = cfg.env.train
    with open_dict(env_cfg):
        env_cfg.total_num_envs = num_envs
        env_cfg.group_size = 1
        env_cfg.use_fixed_reset_state_ids = False
        env_cfg.init_params.num_envs = num_envs
        if "video_cfg" in env_cfg:
            env_cfg.video_cfg.save_video = False

    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    return env


def close_env(env) -> None:
    inner_env = getattr(env, "env", None)
    if hasattr(inner_env, "close"):
        inner_env.close()
    elif hasattr(env, "close"):
        env.close()


def build_reset_obs(cfg: DictConfig, num_envs: int) -> dict[str, Any]:
    env = build_env(cfg, num_envs)
    try:
        obs, _infos = env.reset()
        return obs
    finally:
        close_env(env)


def load_obs(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, tuple):
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain an obs dict or a tuple starting with it.")

    for key in ("obs", "env_obs", "extracted_obs"):
        value = obj.get(key)
        if isinstance(value, dict) and "states" in value:
            return value
    if "states" in obj:
        return obj
    raise ValueError(
        f"{path} does not look like a ManiSkill obs dict. Available keys: {list(obj)}"
    )


def shape_str(value: Any) -> str:
    if torch.is_tensor(value):
        return f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, np.ndarray):
        return f"shape={value.shape} dtype={value.dtype}"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if value is None:
        return "None"
    return type(value).__name__


def to_float_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.detach()
    if device is not None:
        tensor = tensor.to(device)
    return tensor.float()


def compare_tensors(name: str, lhs: Any, rhs: Any, *, tol: float) -> bool:
    lhs_t = to_float_tensor(lhs)
    rhs_t = to_float_tensor(rhs, device=lhs_t.device)
    if lhs_t.shape != rhs_t.shape:
        print(f"[{name}] SHAPE_MISMATCH lhs={tuple(lhs_t.shape)} rhs={tuple(rhs_t.shape)}")
        common_dim = min(lhs_t.shape[-1], rhs_t.shape[-1])
        lhs_t = lhs_t[..., :common_dim]
        rhs_t = rhs_t[..., :common_dim]
        print(f"[{name}] comparing common trailing dim={common_dim}")

    diff = (lhs_t - rhs_t).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    status = "PASS" if max_abs <= tol else "FAIL"
    print(f"[{name}] {status} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
    return max_abs <= tol


def sample_rows(value: Any, *, rows: int = 2, cols: int = 12) -> torch.Tensor:
    tensor = to_float_tensor(value).cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor[:rows, : min(cols, tensor.shape[-1])]


def diff_tensors(name: str, lhs: Any, rhs: Any) -> tuple[float, float]:
    lhs_t = to_float_tensor(lhs)
    rhs_t = to_float_tensor(rhs, device=lhs_t.device)
    if lhs_t.shape != rhs_t.shape:
        common_dim = min(lhs_t.shape[-1], rhs_t.shape[-1])
        lhs_t = lhs_t[..., :common_dim]
        rhs_t = rhs_t[..., :common_dim]
        print(
            f"[{name}] SHAPE_MISMATCH comparing common trailing dim={common_dim} "
            f"lhs={tuple(lhs_t.shape)} rhs={tuple(rhs_t.shape)}"
        )
    diff = (lhs_t - rhs_t).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    print(f"[{name}] max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
    return max_abs, mean_abs


def flatten_ref_chunk(
    ref_chunk: torch.Tensor,
    *,
    chunk_len: int,
    action_dim: int,
) -> torch.Tensor:
    if ref_chunk.dim() == 3:
        ref = ref_chunk[:, :chunk_len, :action_dim]
    else:
        ref = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)[
            :, :chunk_len, :action_dim
        ]
    return ref.reshape(ref.shape[0], -1)


def tensor_stats(name: str, value: Any) -> dict[str, float]:
    tensor = to_float_tensor(value).detach()
    if tensor.numel() == 0:
        stats = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
            "near_bound_rate": 0.0,
            "over_bound_rate": 0.0,
        }
    else:
        stats = {
            "mean": tensor.mean().item(),
            "std": tensor.std(unbiased=False).item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "abs_mean": tensor.abs().mean().item(),
            "abs_max": tensor.abs().max().item(),
            "near_bound_rate": tensor.abs().ge(0.99).float().mean().item(),
            "over_bound_rate": tensor.abs().gt(1.0).float().mean().item(),
        }
    print(
        f"{name}: mean={stats['mean']:.6g} std={stats['std']:.6g} "
        f"min={stats['min']:.6g} max={stats['max']:.6g} "
        f"abs_mean={stats['abs_mean']:.6g} abs_max={stats['abs_max']:.6g} "
        f"near|x|>=0.99={stats['near_bound_rate']:.6g} "
        f"over|x|>1={stats['over_bound_rate']:.6g}"
    )
    return stats


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().square().sum().item()
    return float(total**0.5)


@torch.no_grad()
def check_proprio(model, obs: dict[str, Any], *, tol: float, run_extract: bool) -> None:
    print("== Env Obs ==")
    for key in sorted(obs):
        print(f"{key}: {shape_str(obs[key])}")

    raw_configured_state = model._select_configured_state(obs["states"])
    to_process_obs = model.obs_processor(obs)
    obs_processor_state = to_process_obs["observation/state"]
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = openpi_model.Observation.from_dict(processed_obs)
    openpi_state = observation.state

    state_dim = to_float_tensor(raw_configured_state).shape[-1]
    openpi_state_prefix = openpi_state[..., :state_dim]
    proposed_stage2_proprio = openpi_state_prefix

    print("\n== State Shapes ==")
    print(f"raw_configured_state: {shape_str(raw_configured_state)}")
    print(f"obs_processor observation/state: {shape_str(obs_processor_state)}")
    print(f"OpenPI observation.state: {shape_str(openpi_state)}")

    print("\n== Numerical Checks ==")
    obs_ok = compare_tensors(
        "obs_processor_state_vs_raw",
        obs_processor_state,
        raw_configured_state,
        tol=tol,
    )
    rlt_ok = compare_tensors(
        "current_stage2_proprio_vs_openpi_state",
        raw_configured_state,
        openpi_state_prefix,
        tol=tol,
    )
    proposed_ok = compare_tensors(
        "proposed_stage2_proprio_vs_openpi_state",
        proposed_stage2_proprio,
        openpi_state_prefix,
        tol=tol,
    )

    print("\n== Samples ==")
    print("raw_configured_state[:2]:")
    print(sample_rows(raw_configured_state))
    print("OpenPI observation.state[:2, :state_dim]:")
    print(sample_rows(openpi_state_prefix))
    print("proposed_stage2_proprio[:2]:")
    print(sample_rows(proposed_stage2_proprio))

    if run_extract:
        stage2_obs = model.extract_rlt_stage2_obs(obs)
        print("\n== extract_rlt_stage2_obs ==")
        for key, value in stage2_obs.items():
            print(f"{key}: {shape_str(value)}")
        compare_tensors(
            "extract_proprio_vs_openpi_state",
            stage2_obs["proprio"],
            openpi_state_prefix,
            tol=tol,
        )

    print("\n== Verdict ==")
    if not obs_ok:
        print("obs_processor already changes state before OpenPI transforms. Check state_indices.")
    elif not rlt_ok and proposed_ok:
        print(
            "MISMATCH: current Stage2 stores raw/configured proprio, but OpenPI "
            "consumes a transformed observation.state. This is a real data-flow "
            "semantic mismatch versus an ablation path that used OpenPI-processed state. "
            "The proposed fix aligns on this batch."
        )
    elif not rlt_ok:
        print(
            "MISMATCH: current Stage2 stores raw/configured proprio, but the "
            "proposed transformed-state fix did not align either. Inspect state "
            "dimension/order before changing training code."
        )
    else:
        print(
            "PASS: current Stage2 proprio is numerically equivalent to OpenPI "
            "observation.state for this batch. This suspected mismatch is not the cause."
        )


def prepare_reference_actions(cfg: DictConfig, rlt_obs: dict[str, torch.Tensor]):
    from rlinf.envs.action_utils import prepare_actions

    action_dim = int(cfg.rlt.action_dim)
    chunk_len = int(cfg.rlt.num_action_chunks)
    ref_chunk = rlt_obs["ref_chunk"]
    if ref_chunk.dim() == 2:
        ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
    ref_actions = ref_chunk[:, :chunk_len, :action_dim].contiguous()
    return prepare_actions(
        raw_chunk_actions=ref_actions,
        env_type=cfg.env.train.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=chunk_len,
        action_dim=action_dim,
        policy=cfg.actor.model.get("policy_setup", None),
        wm_env_type=cfg.env.train.get("wm_env_type", None),
    )


def last_obs_from_chunk(obs_list):
    if isinstance(obs_list, (list, tuple)):
        if not obs_list:
            raise RuntimeError("env.chunk_step returned an empty obs_list.")
        return obs_list[-1]
    return obs_list


def run_actor_current_obs_checks(
    cfg: DictConfig,
    actor_model,
    rlt_obs: dict[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    device = next(actor_model.parameters()).device
    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    obs = {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in rlt_obs.items()
    }
    actor_state = actor_model._actor_state(obs)
    critic_state = actor_model._critic_state(obs)
    ref_flat = flatten_ref_chunk(
        obs["ref_chunk"],
        chunk_len=chunk_len,
        action_dim=action_dim,
    )

    print(f"\n== Direct Head Current Obs Check: {label} ==")
    print(f"rlt_head_type: {getattr(actor_model, 'rlt_head_type', None)!r}")
    print(f"actor_state: {shape_str(actor_state)}")
    print(f"critic_state: {shape_str(critic_state)}")
    print(f"ref_flat: {shape_str(ref_flat)}")
    tensor_stats("z_rl", obs["z_rl"])
    tensor_stats("proprio", obs["proprio"])
    tensor_stats("ref_flat", ref_flat)

    if getattr(actor_model, "rlt_head_type", None) == "direct":
        raw_action = actor_model.direct_actor(actor_state)
        clamped_action = raw_action.clamp(-1.0, 1.0)
        action_for_q = clamped_action
        print("\n== Direct Actor Output ==")
        tensor_stats("raw_direct_action", raw_action)
        tensor_stats("clamped_direct_action", clamped_action)
        tensor_stats("direct_action_minus_ref", clamped_action - ref_flat)
        saturated = raw_action.abs().ge(1.0).float().mean().item()
        near_saturated = clamped_action.abs().ge(0.99).float().mean().item()
        print(
            "direct_clamp_summary: "
            f"raw_outside_rate={saturated:.6g} "
            f"clamped_near_bound_rate={near_saturated:.6g}"
        )
    else:
        action_for_q, log_pi, _ = actor_model.sac_forward(
            obs,
            deterministic=True,
            action_noise_sigma=0.0,
        )
        raw_action = action_for_q
        clamped_action = action_for_q
        print("\n== SAC Actor Output ==")
        tensor_stats("deterministic_sac_action", action_for_q)
        tensor_stats("sac_action_minus_ref", action_for_q - ref_flat)
        print(f"log_pi: {shape_str(log_pi)}")

    actor_model.zero_grad(set_to_none=True)
    pred_action, _log_pi, _ = actor_model.sac_forward(
        obs,
        deterministic=True,
        action_noise_sigma=0.0,
    )
    bc_loss = torch.nn.functional.mse_loss(pred_action, ref_flat)
    bc_loss.backward()
    actor_params = (
        actor_model.direct_actor.parameters()
        if hasattr(actor_model, "direct_actor")
        else list(actor_model.backbone.parameters()) + list(actor_model.actor_mean.parameters())
    )
    bc_actor_grad_norm = grad_norm(actor_params)
    print("\n== BC Gradient Check ==")
    print(f"bc_loss_to_ref={bc_loss.detach().item():.8g}")
    print(f"actor_grad_norm_from_bc={bc_actor_grad_norm:.8g}")
    print(
        "bc_grad_verdict="
        + (
            "PASS"
            if np.isfinite(bc_actor_grad_norm) and bc_actor_grad_norm > 0.0
            else "FAIL"
        )
    )
    actor_model.zero_grad(set_to_none=True)

    with torch.no_grad():
        data_q = actor_model.sac_q_forward(obs, ref_flat)
        pi_q = actor_model.sac_q_forward(obs, action_for_q)
    print("\n== Q Head Current Obs Check ==")
    print(f"data_q(ref): {shape_str(data_q)}")
    print(f"pi_q(actor): {shape_str(pi_q)}")
    tensor_stats("q_ref", data_q)
    tensor_stats("q_pi", pi_q)
    tensor_stats("q_pi_minus_q_ref", pi_q - data_q)

    return {
        "obs": obs,
        "ref_flat": ref_flat.detach(),
        "raw_action": raw_action.detach(),
        "action": clamped_action.detach(),
        "data_q": data_q.detach(),
        "pi_q": pi_q.detach(),
    }


@torch.no_grad()
def run_td_target_probe(
    cfg: DictConfig,
    actor_model,
    curr_rlt: dict[str, torch.Tensor],
    next_rlt: dict[str, torch.Tensor],
    rewards: Any,
    dones: Any,
) -> None:
    device = next(actor_model.parameters()).device
    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    gamma = float(cfg.algorithm.gamma)
    curr_obs = {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in curr_rlt.items()
    }
    next_obs = {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in next_rlt.items()
    }
    rewards_t = to_float_tensor(rewards, device=device).reshape(-1, chunk_len)
    dones_t = to_float_tensor(dones, device=device).reshape(-1, chunk_len)
    not_done = ~dones_t.bool().any(dim=-1, keepdim=True)
    discounts = torch.pow(
        torch.as_tensor(gamma, device=device, dtype=rewards_t.dtype),
        torch.arange(chunk_len, device=device, dtype=rewards_t.dtype),
    )
    reward_target = torch.sum(rewards_t * discounts, dim=-1, keepdim=True)
    next_actions, _log_pi, _ = actor_model.sac_forward(
        next_obs,
        deterministic=True,
        action_noise_sigma=0.0,
    )
    q_next_all = actor_model.sac_q_forward(next_obs, next_actions)
    q_next = q_next_all.min(dim=-1, keepdim=True).values
    q_target = reward_target + not_done.to(reward_target.dtype) * (
        gamma**chunk_len
    ) * q_next
    curr_ref = flatten_ref_chunk(
        curr_obs["ref_chunk"],
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    q_data = actor_model.sac_q_forward(curr_obs, curr_ref)

    print("\n== One-Step TD Target Probe ==")
    print(f"rewards: {shape_str(rewards_t)}")
    print(f"dones: {shape_str(dones_t)}")
    tensor_stats("reward_target", reward_target)
    tensor_stats("q_next", q_next)
    tensor_stats("q_target", q_target)
    tensor_stats("q_data_curr_ref", q_data)
    print(f"not_done_mean={not_done.float().mean().item():.6g}")
    print(f"bootstrap_discount={gamma**chunk_len:.8g}")


def check_direct_head_integration(
    cfg: DictConfig,
    feature_model,
    *,
    num_envs: int,
    actor_ckpt: str | None,
    device: torch.device,
    skip_step: bool,
    obs_path: str | None,
) -> None:
    actor_model = build_actor_model(cfg, device)
    load_actor_checkpoint(actor_model, actor_ckpt)

    if obs_path:
        curr_obs = load_obs(obs_path)
        with torch.no_grad():
            curr_rlt = feature_model.extract_rlt_stage2_obs(curr_obs)
        run_actor_current_obs_checks(cfg, actor_model, curr_rlt, label="obs-path")
        if not skip_step:
            print("\n--skip-step is required when --obs-path is used.")
        return

    env = build_env(cfg, num_envs)
    try:
        curr_obs, _infos = env.reset()
        with torch.no_grad():
            curr_rlt = feature_model.extract_rlt_stage2_obs(curr_obs)
        current = run_actor_current_obs_checks(
            cfg,
            actor_model,
            curr_rlt,
            label="reset",
        )
        if skip_step:
            return

        chunk_actions = prepare_reference_actions(cfg, curr_rlt)
        obs_list, rewards, terminations, truncations, _infos_list = env.chunk_step(
            chunk_actions
        )
        next_obs = last_obs_from_chunk(obs_list)
        with torch.no_grad():
            next_rlt = feature_model.extract_rlt_stage2_obs(next_obs)
    finally:
        close_env(env)

    print("\n== Direct Head Env Step Probe ==")
    print("step action source: ref_chunk, not direct actor")
    print(f"prepared_ref_actions: {shape_str(chunk_actions)}")
    print(f"rewards: {shape_str(rewards)}")
    print(f"terminations: {shape_str(terminations)}")
    print(f"truncations: {shape_str(truncations)}")
    for key in ("proprio", "ref_chunk", "z_rl"):
        diff_tensors(f"curr_vs_next_{key}", curr_rlt[key], next_rlt[key])

    run_actor_current_obs_checks(cfg, actor_model, next_rlt, label="next")
    dones = torch.logical_or(
        torch.as_tensor(terminations).bool(),
        torch.as_tensor(truncations).bool(),
    )
    run_td_target_probe(
        cfg,
        actor_model,
        curr_rlt,
        next_rlt,
        rewards,
        dones,
    )

    print("\n== Direct Head Verdict Hints ==")
    raw_outside_rate = to_float_tensor(current["raw_action"]).abs().gt(1.0).float().mean()
    action_ref_abs = (current["action"] - current["ref_flat"]).abs().mean()
    if raw_outside_rate.item() > 0.05:
        print(
            "WARN: direct actor raw output is often outside [-1, 1]. Because "
            "the current implementation clamps before BC/Q loss, saturated "
            "dimensions get zero clamp gradient."
        )
    if action_ref_abs.item() > 0.5:
        print(
            "WARN: direct actor starts far from ref_chunk. With large BC weight "
            "this can dominate early actor updates and make Q improvement slow."
        )
    if raw_outside_rate.item() <= 0.05 and action_ref_abs.item() <= 0.5:
        print("PASS: no obvious direct-head scale/clamp issue on this batch.")


def _info_tensor_rate(infos: Any, key: str) -> float | None:
    if not isinstance(infos, dict):
        return None
    value = infos.get(key)
    if value is None and isinstance(infos.get("episode"), dict):
        value = infos["episode"].get(key)
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return None
    return float(tensor.detach().float().mean().item())


@torch.no_grad()
def check_ref_rollout(
    cfg: DictConfig,
    feature_model,
    *,
    num_envs: int,
    max_rollout_chunks: int,
) -> None:
    env = build_env(cfg, num_envs)
    chunk_len = int(cfg.rlt.num_action_chunks)
    total_reward = None
    done_once = None
    success_once = torch.zeros(num_envs, dtype=torch.bool)
    chunk_rows = []
    try:
        obs, infos = env.reset()
        for chunk_idx in range(max_rollout_chunks):
            rlt_obs = feature_model.extract_rlt_stage2_obs(obs)
            chunk_actions = prepare_reference_actions(cfg, rlt_obs)
            obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
                chunk_actions
            )
            obs = last_obs_from_chunk(obs_list)
            last_infos = infos_list[-1] if infos_list else {}
            rewards_t = torch.as_tensor(rewards).detach().float().cpu()
            dones_t = torch.logical_or(
                torch.as_tensor(terminations).detach().bool().cpu(),
                torch.as_tensor(truncations).detach().bool().cpu(),
            )
            if total_reward is None:
                total_reward = torch.zeros(rewards_t.shape[0], dtype=torch.float32)
                done_once = torch.zeros(rewards_t.shape[0], dtype=torch.bool)
            total_reward += rewards_t.sum(dim=1)
            done_once |= dones_t.any(dim=1)

            success_rate = _info_tensor_rate(last_infos, "success")
            success_once_rate = _info_tensor_rate(last_infos, "success_once")
            if success_rate is not None:
                success_tensor = torch.as_tensor(last_infos["success"]).detach().bool().cpu()
                success_once |= success_tensor.reshape(-1)
            if success_once_rate is not None and isinstance(last_infos, dict):
                value = last_infos.get("success_once")
                if value is None and isinstance(last_infos.get("episode"), dict):
                    value = last_infos["episode"].get("success_once")
                if value is not None:
                    success_once |= torch.as_tensor(value).detach().bool().cpu().reshape(-1)

            ref_flat = flatten_ref_chunk(
                rlt_obs["ref_chunk"].detach().cpu(),
                chunk_len=chunk_len,
                action_dim=int(cfg.rlt.action_dim),
            )
            row = {
                "chunk": chunk_idx,
                "reward_mean": rewards_t.sum(dim=1).mean().item(),
                "reward_nonzero_rate": rewards_t.ne(0).float().mean().item(),
                "done_rate": dones_t.any(dim=1).float().mean().item(),
                "success_rate": -1.0 if success_rate is None else success_rate,
                "success_once_rate": success_once.float().mean().item()
                if success_once_rate is None
                else max(success_once.float().mean().item(), success_once_rate),
                "ref_abs_mean": ref_flat.abs().mean().item(),
                "ref_abs_max": ref_flat.abs().max().item(),
            }
            chunk_rows.append(row)
            print(
                "ref_rollout_chunk "
                f"{chunk_idx:03d}: reward_mean={row['reward_mean']:.6g} "
                f"reward_nonzero_rate={row['reward_nonzero_rate']:.6g} "
                f"done_rate={row['done_rate']:.6g} "
                f"success_rate={row['success_rate']:.6g} "
                f"success_once_rate={row['success_once_rate']:.6g} "
                f"ref_abs_mean={row['ref_abs_mean']:.6g} "
                f"ref_abs_max={row['ref_abs_max']:.6g}"
            )
            if done_once is not None and done_once.all():
                break
    finally:
        close_env(env)

    total_reward = (
        torch.zeros(num_envs, dtype=torch.float32)
        if total_reward is None
        else total_reward
    )
    done_once = (
        torch.zeros(num_envs, dtype=torch.bool) if done_once is None else done_once
    )
    print("\n== Ref Rollout Summary ==")
    print(f"num_envs={num_envs}")
    print(f"chunks_run={len(chunk_rows)}")
    print(f"control_steps_run={len(chunk_rows) * chunk_len}")
    print(f"total_reward_mean={total_reward.mean().item():.6g}")
    print(f"total_reward_max={total_reward.max().item():.6g}")
    print(f"done_once_rate={done_once.float().mean().item():.6g}")
    print(f"success_once_rate={success_once.float().mean().item():.6g}")
    if success_once.float().mean().item() <= 0.0:
        print(
            "VERDICT: ref_chunk rollout got zero successes in this probe. "
            "If SFT eval succeeds under the same checkpoint/config, the mismatch is "
            "before RL training: Stage2 feature/ref action extraction or action "
            "formatting differs from the SFT eval path."
        )
    else:
        print(
            "VERDICT: ref_chunk rollout can reach success. Then the remaining "
            "problem is more likely in actor update/replay/Q learning, not the "
            "OpenPI ref action path."
        )


@torch.no_grad()
def check_next_obs_alignment(cfg: DictConfig, model, *, num_envs: int) -> None:
    env = build_env(cfg, num_envs)
    try:
        curr_obs, _infos = env.reset()
        curr_rlt = model.extract_rlt_stage2_obs(curr_obs)
        chunk_actions = prepare_reference_actions(cfg, curr_rlt)
        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
            chunk_actions
        )
        next_obs = last_obs_from_chunk(obs_list)
        next_rlt = model.extract_rlt_stage2_obs(next_obs)
    finally:
        close_env(env)

    print("\n== Next Obs Alignment ==")
    print(f"chunk_actions: {shape_str(chunk_actions)}")
    print(f"rewards: {shape_str(rewards)}")
    print(f"terminations: {shape_str(terminations)}")
    print(f"truncations: {shape_str(truncations)}")
    print(f"infos_list: {shape_str(infos_list)}")

    for key in ("proprio", "ref_chunk", "z_rl"):
        print(f"curr_{key}: {shape_str(curr_rlt[key])}")
        print(f"next_{key}: {shape_str(next_rlt[key])}")
        diff_tensors(f"curr_vs_next_{key}", curr_rlt[key], next_rlt[key])

    print("\n== Next Ref Samples ==")
    print("curr_ref_chunk[:2]:")
    print(sample_rows(curr_rlt["ref_chunk"]))
    print("next_ref_chunk[:2]:")
    print(sample_rows(next_rlt["ref_chunk"]))


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)

    cfg = load_cfg(args.config_name, args.config_dir)
    model = build_feature_model(cfg)
    if args.check_ref_rollout:
        model = maybe_move_feature_model(model, resolve_device(args.device))
        check_ref_rollout(
            cfg,
            model,
            num_envs=args.num_envs,
            max_rollout_chunks=args.max_rollout_chunks,
        )
        return
    if args.check_direct_head:
        device = resolve_device(args.device)
        model = maybe_move_feature_model(model, device)
        if args.obs_path and not args.skip_step:
            print("NOTE: --obs-path has no env handle; enabling --skip-step behavior.")
        check_direct_head_integration(
            cfg,
            model,
            num_envs=args.num_envs,
            actor_ckpt=args.actor_ckpt,
            device=device,
            skip_step=bool(args.skip_step or args.obs_path),
            obs_path=args.obs_path,
        )
        return
    if args.check_next:
        if args.obs_path:
            raise ValueError("--check-next creates an env and cannot be used with --obs-path.")
        check_next_obs_alignment(cfg, model, num_envs=args.num_envs)
        return
    if args.obs_path:
        obs = load_obs(args.obs_path)
    else:
        obs = build_reset_obs(cfg, args.num_envs)
    check_proprio(model, obs, tol=args.tol, run_extract=args.run_extract)


if __name__ == "__main__":
    main()
