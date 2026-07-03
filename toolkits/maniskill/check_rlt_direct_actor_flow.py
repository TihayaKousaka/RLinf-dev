#!/usr/bin/env python3
"""Check RLT Stage2 direct-actor update/load/eval flow.

This script is intentionally read-only with respect to training code.  It is a
fast diagnostic for the current refactored ManiSkill Stage2 path:

1. Inspect whether an actor checkpoint contains the direct MLP actor.
2. Optionally compare it against a base checkpoint to verify the actor changed.
3. Load the same checkpoint into actor and rollout-shaped MLP models and verify
   direct_actor weights/actions match.
4. Optionally run short ManiSkill rollouts under ref/eval/train/td3 action modes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether current RLT Stage2 direct actor is updated, "
            "load-compatible with rollout, and action-equivalent across modes."
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
        "--actor-ckpt",
        default=None,
        help=(
            "Current Stage2 actor checkpoint file or directory. Directories may "
            "contain model_state_dict/full_weights.pt or actor/model_state_dict/full_weights.pt."
        ),
    )
    parser.add_argument(
        "--base-actor-ckpt",
        default=None,
        help=(
            "Optional base/earlier actor checkpoint for direct_actor delta. "
            "Use this to compare global_step_50 vs global_step_0/20, etc."
        ),
    )
    parser.add_argument(
        "--obs-path",
        default=None,
        help=(
            "Optional torch-saved obs dict or extracted RLT obs dict. If omitted "
            "and --skip-obs is not set, the script creates a small env reset."
        ),
    )
    parser.add_argument(
        "--with-obs",
        action="store_true",
        help=(
            "Also build/load one observation and compare action modes. By default "
            "the script avoids ManiSkill env reset, because CUDA sim failures can "
            "mask the checkpoint/load checks we need first."
        ),
    )
    parser.add_argument(
        "--skip-obs",
        action="store_true",
        help="Deprecated alias kept for compatibility; it forces obs/action checks off.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for feature/action checks: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--run-rollout-modes",
        default="",
        help=(
            "Optional comma-separated modes to rollout: ref,eval,train_default,"
            "train_deterministic,td3. Empty means no env rollout."
        ),
    )
    parser.add_argument(
        "--rollout-env",
        choices=("train", "eval"),
        default="eval",
        help="Env split for --run-rollout-modes.",
    )
    parser.add_argument("--rollout-chunks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fail-on-suspect",
        action="store_true",
        help="Exit non-zero if a hard check fails.",
    )
    parser.add_argument(
        "--update-step",
        type=int,
        default=None,
        help=(
            "Optional learner update_step. If provided, print the same RLT "
            "actor-loss weights/ready gate used during training."
        ),
    )
    return parser.parse_args()


def load_cfg(config_name: str, config_dir: str) -> DictConfig:
    os.environ.setdefault("EMBODIED_PATH", str(REPO_ROOT / "examples" / "embodiment"))
    with initialize_config_dir(version_base="1.1", config_dir=str(Path(config_dir))):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_candidates(path: str | None) -> list[Path]:
    if not path:
        return []
    ckpt_path = Path(path)
    if ckpt_path.is_file():
        return [ckpt_path]
    return [
        ckpt_path / "model_state_dict" / "full_weights.pt",
        ckpt_path / "actor" / "model_state_dict" / "full_weights.pt",
        ckpt_path / "full_weights.pt",
    ]


def strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("_fsdp_wrapped_module.", "module.", "model.")
    stripped: dict[str, Any] = {}
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


def load_checkpoint_state_dict(path: str) -> tuple[Path, dict[str, torch.Tensor]]:
    for candidate in checkpoint_candidates(path):
        if not candidate.exists():
            continue
        state = torch.load(candidate, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                nested = state.get(key)
                if isinstance(nested, dict):
                    state = nested
                    break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint object at {candidate}: {type(state)}")
        state = strip_state_dict_prefixes(state)
        tensor_state = {k: v for k, v in state.items() if torch.is_tensor(v)}
        return candidate, tensor_state
    raise FileNotFoundError(
        "Could not find actor checkpoint. Tried: "
        + ", ".join(str(p) for p in checkpoint_candidates(path))
    )


def is_ablation_stage2_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("actor.mlp.net.") for key in state_dict) or any(
        key.startswith("critic.q1.mlp.net.") for key in state_dict
    )


def map_ablation_stage2_to_rlt_mlp(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ablation RLTStage2Policy keys to current RLTMLPPolicy direct keys."""
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("actor.mlp.net."):
            mapped[f"direct_actor.{key[len('actor.mlp.net.'):]}"] = value
        elif key.startswith("critic.q1.mlp.net."):
            mapped[f"q_head.qs.0.{key[len('critic.q1.mlp.net.'):]}"] = value
        elif key.startswith("critic.q2.mlp.net."):
            mapped[f"q_head.qs.1.{key[len('critic.q2.mlp.net.'):]}"] = value

    for key, value in state_dict.items():
        if key.startswith("target_actor.mlp.net."):
            mapped.setdefault(
                f"direct_actor.{key[len('target_actor.mlp.net.'):]}",
                value,
            )
        elif key.startswith("critic.q1_target.mlp.net."):
            mapped.setdefault(
                f"q_head.qs.0.{key[len('critic.q1_target.mlp.net.'):]}",
                value,
            )
        elif key.startswith("critic.q2_target.mlp.net."):
            mapped.setdefault(
                f"q_head.qs.1.{key[len('critic.q2_target.mlp.net.'):]}",
                value,
            )
    return mapped


def normalize_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[str, dict[str, torch.Tensor]]:
    if is_ablation_stage2_state_dict(state_dict):
        return "ablation_stage2_mapped", map_ablation_stage2_to_rlt_mlp(state_dict)
    return "current", state_dict


def infer_head_type(state_dict: dict[str, torch.Tensor]) -> str | None:
    if any(k.startswith("direct_actor.") for k in state_dict):
        return "direct"
    if any(k.startswith("actor.mlp.net.") for k in state_dict):
        return "direct"
    if any(k.startswith("actor_mean.") for k in state_dict) or any(
        k.startswith("backbone.") for k in state_dict
    ):
        return "sac"
    return None


def configure_head_type(cfg: DictConfig, state_dict: dict[str, torch.Tensor]) -> None:
    head_type = infer_head_type(state_dict)
    if head_type is None:
        return
    with open_dict(cfg.actor.model):
        cfg.actor.model.rlt_head_type = head_type
    if "model" in cfg.rollout:
        with open_dict(cfg.rollout.model):
            cfg.rollout.model.rlt_head_type = head_type


def build_model(model_cfg: DictConfig, device: torch.device):
    from rlinf.models import get_model

    model = get_model(model_cfg)
    if model is None:
        raise RuntimeError(f"Failed to build model_type={model_cfg.model_type!r}")
    model.to(device)
    model.eval()
    return model


def build_feature_model(cfg: DictConfig, device: torch.device):
    from rlinf.models import get_model

    model_cfg = copy.deepcopy(cfg.rollout.rlt_feature_model)
    with open_dict(model_cfg):
        model_cfg.load_to_device = False
    model = get_model(model_cfg)
    if model is None:
        raise RuntimeError(f"Failed to build feature model: {model_cfg.model_type!r}")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_model_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    label: str,
) -> tuple[list[str], list[str]]:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"{label}.missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print(f"{label}.missing_sample={list(missing)[:12]}")
    if unexpected:
        print(f"{label}.unexpected_sample={list(unexpected)[:12]}")
    return list(missing), list(unexpected)


def select_keys(
    state_dict: dict[str, torch.Tensor],
    prefixes: str | tuple[str, ...],
) -> list[str]:
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return sorted(k for k in state_dict if k.startswith(prefixes))


def tensor_hash(tensors: Iterable[torch.Tensor]) -> str:
    sha = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        sha.update(str(arr.shape).encode("utf-8"))
        sha.update(str(arr.dtype).encode("utf-8"))
        sha.update(arr.tobytes())
    return sha.hexdigest()[:16]


def group_stats(
    state_dict: dict[str, torch.Tensor],
    keys: list[str],
    *,
    label: str,
) -> dict[str, float | int | str]:
    tensors = [state_dict[k].detach().float().cpu().reshape(-1) for k in keys]
    if tensors:
        flat = torch.cat(tensors)
        stats: dict[str, float | int | str] = {
            "count": len(keys),
            "numel": int(flat.numel()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "abs_mean": float(flat.abs().mean().item()),
            "abs_max": float(flat.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(flat).item()),
            "sha16": tensor_hash(state_dict[k] for k in keys),
        }
    else:
        stats = {
            "count": 0,
            "numel": 0,
            "mean": 0.0,
            "std": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
            "l2": 0.0,
            "sha16": "none",
        }
    print(
        f"{label}: count={stats['count']} numel={stats['numel']} "
        f"mean={stats['mean']:.6g} std={stats['std']:.6g} "
        f"abs_mean={stats['abs_mean']:.6g} abs_max={stats['abs_max']:.6g} "
        f"l2={stats['l2']:.6g} sha16={stats['sha16']}"
    )
    return stats


def compare_state_groups(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    keys: list[str],
    *,
    label: str,
) -> dict[str, float | int]:
    common = [k for k in keys if k in lhs and k in rhs and lhs[k].shape == rhs[k].shape]
    missing_lhs = [k for k in keys if k not in lhs]
    missing_rhs = [k for k in keys if k not in rhs]
    shape_mismatch = [
        k for k in keys if k in lhs and k in rhs and lhs[k].shape != rhs[k].shape
    ]
    if not common:
        print(
            f"{label}: common=0 missing_lhs={len(missing_lhs)} "
            f"missing_rhs={len(missing_rhs)} shape_mismatch={len(shape_mismatch)}"
        )
        return {"common": 0, "mean_abs": 0.0, "max_abs": 0.0, "l2": 0.0}

    diffs = [(lhs[k].detach().float().cpu() - rhs[k].detach().float().cpu()).reshape(-1) for k in common]
    flat = torch.cat(diffs)
    stats = {
        "common": len(common),
        "mean_abs": float(flat.abs().mean().item()),
        "max_abs": float(flat.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "changed_rate": float(flat.abs().gt(1e-8).float().mean().item()),
    }
    print(
        f"{label}: common={stats['common']} missing_lhs={len(missing_lhs)} "
        f"missing_rhs={len(missing_rhs)} shape_mismatch={len(shape_mismatch)} "
        f"mean_abs={stats['mean_abs']:.8g} max_abs={stats['max_abs']:.8g} "
        f"l2={stats['l2']:.8g} changed_rate={stats['changed_rate']:.6g}"
    )
    return stats


def tensor_stats(name: str, value: Any) -> dict[str, float]:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.detach().float()
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


def compare_tensors(name: str, lhs: Any, rhs: Any) -> dict[str, float]:
    lhs_t = lhs if torch.is_tensor(lhs) else torch.as_tensor(lhs)
    rhs_t = rhs if torch.is_tensor(rhs) else torch.as_tensor(rhs)
    lhs_t = lhs_t.detach().float().cpu()
    rhs_t = rhs_t.detach().float().cpu()
    if lhs_t.shape != rhs_t.shape:
        print(f"{name}: SHAPE_MISMATCH lhs={tuple(lhs_t.shape)} rhs={tuple(rhs_t.shape)}")
        return {"mean_abs": float("inf"), "max_abs": float("inf")}
    diff = lhs_t - rhs_t
    stats = {
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(diff.reshape(-1)).item())
        if diff.numel()
        else 0.0,
    }
    print(
        f"{name}: mean_abs={stats['mean_abs']:.8g} "
        f"max_abs={stats['max_abs']:.8g} l2={stats['l2']:.8g}"
    )
    return stats


def flatten_ref_chunk(ref_chunk: torch.Tensor, *, chunk_len: int, action_dim: int) -> torch.Tensor:
    if ref_chunk.dim() == 3:
        ref = ref_chunk[:, :chunk_len, :action_dim]
    else:
        ref = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)[
            :, :chunk_len, :action_dim
        ]
    return ref.reshape(ref.shape[0], -1)


def normalize_obs_tensor_dict(
    obs: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach().to(device=device, dtype=torch.float32) if torch.is_tensor(value) else value)
        for key, value in obs.items()
    }


def load_obs(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, tuple):
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain an obs dict.")
    for key in ("rlt_obs", "extracted_obs"):
        if isinstance(obj.get(key), dict) and "z_rl" in obj[key]:
            return obj[key]
    for key in ("obs", "env_obs"):
        if isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def build_env(cfg: DictConfig, split: str, num_envs: int, seed: int | None):
    from rlinf.envs import get_env_cls

    env_cfg = copy.deepcopy(cfg.env[split])
    with open_dict(env_cfg):
        env_cfg.total_num_envs = int(num_envs)
        env_cfg.group_size = 1
        env_cfg.init_params.num_envs = int(num_envs)
        if seed is not None:
            env_cfg.seed = int(seed)
        if "video_cfg" in env_cfg:
            env_cfg.video_cfg.save_video = False
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    return env, env_cfg


def close_env(env) -> None:
    inner_env = getattr(env, "env", None)
    if hasattr(inner_env, "close"):
        inner_env.close()
    elif hasattr(env, "close"):
        env.close()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def extract_rlt_obs(
    cfg: DictConfig,
    feature_model,
    *,
    device: torch.device,
    obs_path: str | None,
    num_envs: int,
) -> tuple[dict[str, torch.Tensor], Any | None]:
    obs = load_obs(obs_path) if obs_path else None
    env = None
    if obs is None:
        env, _env_cfg = build_env(cfg, "train", num_envs, seed=int(cfg.env.train.seed))
        obs, _infos = env.reset()

    if "z_rl" in obs and "proprio" in obs and "ref_chunk" in obs:
        rlt_obs = obs
    else:
        rlt_obs = feature_model.extract_rlt_stage2_obs(obs)
    rlt_obs = normalize_obs_tensor_dict(rlt_obs, device=device)
    return rlt_obs, env


@torch.no_grad()
def predict_actions(
    model,
    rlt_obs: dict[str, torch.Tensor],
    *,
    mode_name: str,
    cfg: DictConfig,
) -> torch.Tensor:
    sampling_cfg = cfg.algorithm.get("rlt_action_sampling", {})
    kwargs: dict[str, Any] = {}
    mode = "eval"
    if mode_name == "eval":
        mode = "eval"
    elif mode_name == "train_default":
        mode = "train"
    elif mode_name == "train_deterministic":
        mode = "train"
        kwargs["sampling_mode"] = "deterministic"
    elif mode_name == "td3":
        mode = "train"
        kwargs.update(
            {
                "sampling_mode": "td3_action_noise",
                "action_noise_sigma": float(sampling_cfg.get("action_noise_sigma", 0.0)),
                "action_noise_clip": sampling_cfg.get("action_noise_clip", None),
            }
        )
    else:
        raise ValueError(f"Unsupported action mode: {mode_name!r}")
    actions, _ = model.predict_action_batch(
        env_obs=rlt_obs,
        mode=mode,
        return_obs=True,
        **kwargs,
    )
    return actions.detach()


def prepare_env_actions(
    cfg: DictConfig,
    env_cfg: DictConfig,
    raw_chunk_actions: torch.Tensor,
) -> torch.Tensor | np.ndarray:
    from rlinf.envs.action_utils import prepare_actions

    return prepare_actions(
        raw_chunk_actions=raw_chunk_actions,
        env_type=env_cfg.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=int(cfg.rlt.num_action_chunks),
        action_dim=int(cfg.rlt.action_dim),
        policy=cfg.actor.model.get("policy_setup", None),
        wm_env_type=env_cfg.get("wm_env_type", None),
    )


def last_obs_from_chunk(obs_list):
    if isinstance(obs_list, (list, tuple)):
        if not obs_list:
            raise RuntimeError("env.chunk_step returned empty obs_list.")
        return obs_list[-1]
    return obs_list


def info_rate(infos: Any, key: str) -> float | None:
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
def run_rollout_mode(
    cfg: DictConfig,
    feature_model,
    actor_model,
    *,
    mode_name: str,
    split: str,
    num_envs: int,
    chunks: int,
    seed: int,
    device: torch.device,
) -> None:
    env, env_cfg = build_env(cfg, split, num_envs, seed)
    chunk_len = int(cfg.rlt.num_action_chunks)
    total_reward: torch.Tensor | None = None
    done_once: torch.Tensor | None = None
    success_once = torch.zeros(num_envs, dtype=torch.bool)
    try:
        obs, _infos = env.reset()
        for chunk_idx in range(chunks):
            rlt_obs = feature_model.extract_rlt_stage2_obs(obs)
            rlt_obs = normalize_obs_tensor_dict(rlt_obs, device=device)
            if mode_name == "ref":
                raw_actions = rlt_obs["ref_chunk"]
                if raw_actions.dim() == 2:
                    raw_actions = raw_actions.reshape(
                        raw_actions.shape[0], -1, int(cfg.rlt.action_dim)
                    )
                raw_actions = raw_actions[:, :chunk_len, : int(cfg.rlt.action_dim)]
            else:
                raw_actions = predict_actions(
                    actor_model,
                    rlt_obs,
                    mode_name=mode_name,
                    cfg=cfg,
                )
            env_actions = prepare_env_actions(cfg, env_cfg, raw_actions)
            obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
                env_actions
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

            success_rate = info_rate(last_infos, "success")
            success_once_rate = info_rate(last_infos, "success_once")
            if success_rate is not None and "success" in last_infos:
                success_once |= torch.as_tensor(last_infos["success"]).detach().bool().cpu().reshape(-1)
            if success_once_rate is not None:
                value = last_infos.get("success_once")
                if value is None and isinstance(last_infos.get("episode"), dict):
                    value = last_infos["episode"].get("success_once")
                if value is not None:
                    success_once |= torch.as_tensor(value).detach().bool().cpu().reshape(-1)

            if chunk_idx == 0 or chunk_idx == chunks - 1 or (chunk_idx + 1) % 5 == 0:
                print(
                    f"{mode_name}/{split} chunk {chunk_idx:03d}: "
                    f"reward_mean={rewards_t.sum(dim=1).mean().item():.6g} "
                    f"reward_nonzero_rate={rewards_t.ne(0).float().mean().item():.6g} "
                    f"done_once_rate={done_once.float().mean().item():.6g} "
                    f"success_once_rate={success_once.float().mean().item():.6g}"
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
    print(
        f"ROLLOUT_SUMMARY mode={mode_name} split={split} chunks_run={chunk_idx + 1} "
        f"total_reward_mean={total_reward.mean().item():.6g} "
        f"total_reward_max={total_reward.max().item():.6g} "
        f"done_once_rate={done_once.float().mean().item():.6g} "
        f"success_once_rate={success_once.float().mean().item():.6g}"
    )


def inspect_optimizer_and_sync_names(model: torch.nn.Module) -> dict[str, int]:
    from rlinf.utils.utils import collect_param_names_need_sync

    trainable_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    sync_names = collect_param_names_need_sync(model)
    critic_filters = ("encoders", "encoder", "q_head", "state_proj")
    critic_names = [
        name for name in trainable_names if any(f in name for f in critic_filters)
    ]
    actor_names = [name for name in trainable_names if name not in set(critic_names)]
    direct_actor_names = [name for name in trainable_names if name.startswith("direct_actor.")]
    q_head_names = [name for name in trainable_names if name.startswith("q_head.")]
    print("\n== Optimizer/Sync Name Check ==")
    print(f"trainable_params={len(trainable_names)}")
    print(f"main_actor_optimizer_params={len(actor_names)}")
    print(f"critic_optimizer_params={len(critic_names)}")
    print(f"direct_actor_trainable_params={len(direct_actor_names)}")
    print(f"q_head_trainable_params={len(q_head_names)}")
    print(f"sync_names={len(sync_names)}")
    print(
        "sync_contains_direct_actor="
        f"{any(name.startswith('direct_actor.') for name in sync_names)}"
    )
    print(f"sync_contains_q_head={any(name.startswith('q_head.') for name in sync_names)}")
    print(f"direct_actor_name_sample={direct_actor_names[:8]}")
    print(f"q_head_name_sample={q_head_names[:8]}")
    return {
        "direct_actor_trainable": len(direct_actor_names),
        "q_head_trainable": len(q_head_names),
        "direct_actor_sync": int(any(name.startswith("direct_actor.") for name in sync_names)),
        "q_head_sync": int(any(name.startswith("q_head.") for name in sync_names)),
    }


def inspect_actions(
    cfg: DictConfig,
    actor_model,
    rollout_model,
    rlt_obs: dict[str, torch.Tensor],
) -> dict[str, float]:
    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    ref_flat = flatten_ref_chunk(
        rlt_obs["ref_chunk"],
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    print("\n== Same-Obs Action Path Check ==")
    print(f"rlt_head_type={getattr(actor_model, 'rlt_head_type', None)!r}")
    print(f"z_rl_shape={tuple(rlt_obs['z_rl'].shape)}")
    print(f"proprio_shape={tuple(rlt_obs['proprio'].shape)}")
    print(f"ref_chunk_shape={tuple(rlt_obs['ref_chunk'].shape)}")
    tensor_stats("ref_flat", ref_flat)

    modes = ("eval", "train_default", "train_deterministic", "td3")
    actions: dict[str, torch.Tensor] = {}
    for mode_name in modes:
        actions[mode_name] = predict_actions(
            actor_model,
            rlt_obs,
            mode_name=mode_name,
            cfg=cfg,
        ).reshape(ref_flat.shape[0], -1)
        tensor_stats(f"actor_action_{mode_name}", actions[mode_name])
        compare_tensors(
            f"actor_action_{mode_name}_minus_ref",
            actions[mode_name],
            ref_flat,
        )

    rollout_eval = predict_actions(
        rollout_model,
        rlt_obs,
        mode_name="eval",
        cfg=cfg,
    ).reshape(ref_flat.shape[0], -1)
    tensor_stats("rollout_loaded_action_eval", rollout_eval)

    metrics = {}
    metrics["actor_eval_vs_train_det_max"] = compare_tensors(
        "actor_eval_vs_train_deterministic",
        actions["eval"],
        actions["train_deterministic"],
    )["max_abs"]
    metrics["actor_eval_vs_train_default_max"] = compare_tensors(
        "actor_eval_vs_train_default",
        actions["eval"],
        actions["train_default"],
    )["max_abs"]
    metrics["actor_eval_vs_td3_max"] = compare_tensors(
        "actor_eval_vs_td3",
        actions["eval"],
        actions["td3"],
    )["max_abs"]
    metrics["actor_eval_vs_rollout_eval_max"] = compare_tensors(
        "actor_eval_vs_rollout_loaded_eval",
        actions["eval"],
        rollout_eval,
    )["max_abs"]
    return metrics


def _sample_mode_kwargs(cfg: DictConfig, key: str, default: str) -> dict[str, Any]:
    sampling_cfg = cfg.algorithm.get("rlt_action_sampling", {})
    mode = str(sampling_cfg.get(key, default))
    if mode == "sac_sample":
        return {"deterministic": False}
    if mode == "deterministic":
        return {"deterministic": True}
    if mode == "td3_action_noise":
        prefix = "target_" if key.startswith("target_") else ""
        return {
            "deterministic": True,
            "action_noise_sigma": float(
                sampling_cfg.get(
                    f"{prefix}noise_sigma",
                    sampling_cfg.get("action_noise_sigma", 0.0),
                )
            ),
            "action_noise_clip": sampling_cfg.get(
                f"{prefix}noise_clip",
                sampling_cfg.get("action_noise_clip", None),
            ),
        }
    raise ValueError(f"Unsupported RLT action mode {mode!r} for {key}.")


def _resolve_loss_weights(
    cfg: DictConfig,
    update_step: int | None,
) -> tuple[float, float, float, float]:
    step = 0 if update_step is None else int(update_step)
    rlt_loss_cfg = cfg.algorithm.get("rlt_actor_loss", {})
    warmup_updates = int(
        rlt_loss_cfg.get(
            "actor_loss_warmup_updates",
            cfg.algorithm.get("actor_loss_warmup_updates", 0),
        )
    )
    ramp_updates = int(
        rlt_loss_cfg.get(
            "actor_loss_ramp_updates",
            cfg.algorithm.get("actor_loss_ramp_updates", 0),
        )
    )
    warmup_bc = float(
        rlt_loss_cfg.get(
            "warmup_bc_weight",
            cfg.algorithm.get("warmup_bc_weight", cfg.algorithm.get("bc_weight", 1.0)),
        )
    )
    warmup_q = float(
        rlt_loss_cfg.get(
            "warmup_q_weight",
            cfg.algorithm.get("warmup_q_weight", cfg.algorithm.get("q_weight", 1.0)),
        )
    )
    online_bc = float(
        rlt_loss_cfg.get(
            "online_bc_weight",
            cfg.algorithm.get("online_bc_weight", cfg.algorithm.get("bc_weight", 1.0)),
        )
    )
    online_q = float(
        rlt_loss_cfg.get(
            "online_q_weight",
            cfg.algorithm.get("online_q_weight", cfg.algorithm.get("q_weight", 1.0)),
        )
    )
    if step < warmup_updates:
        bc_weight = warmup_bc
        q_weight = warmup_q
        ramp = 0.0
    elif ramp_updates > 0:
        ramp = min(
            1.0,
            max(0.0, float(step - warmup_updates + 1) / float(ramp_updates)),
        )
        bc_weight = warmup_bc + ramp * (online_bc - warmup_bc)
        q_weight = warmup_q + ramp * (online_q - warmup_q)
    else:
        bc_weight = online_bc
        q_weight = online_q
        ramp = 1.0
    delta_weight = float(
        rlt_loss_cfg.get("delta_weight", cfg.algorithm.get("delta_weight", 0.0))
    )
    return bc_weight, q_weight, delta_weight, ramp


def inspect_actor_loss_consistency(
    cfg: DictConfig,
    actor_model,
    rlt_obs: dict[str, torch.Tensor],
    *,
    update_step: int | None,
) -> None:
    from rlinf.models.embodiment.base_policy import ForwardType

    chunk_len = int(cfg.rlt.num_action_chunks)
    action_dim = int(cfg.rlt.action_dim)
    ref_flat = flatten_ref_chunk(
        rlt_obs["ref_chunk"],
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    bc_weight, q_weight, delta_weight, ramp = _resolve_loss_weights(
        cfg,
        update_step,
    )
    warmup_required = int(
        cfg.algorithm.get("rlt_schedule", {}).get(
            "warmup_post_collect_updates",
            cfg.algorithm.get("warmup_post_collect_updates", 0),
        )
    )
    ready_for_online = None if update_step is None else int(update_step) >= warmup_required

    print("\n== Actor Loss Consistency Check ==")
    print(f"update_step={update_step}")
    print(f"warmup_post_collect_updates={warmup_required}")
    print(f"ready_for_online={ready_for_online}")
    print(
        f"loss_weights: bc={bc_weight:.6g} q={q_weight:.6g} "
        f"delta={delta_weight:.6g} ramp={ramp:.6g}"
    )

    with torch.no_grad():
        eval_pi, _, _ = actor_model(
            forward_type=ForwardType.SAC,
            obs=rlt_obs,
            deterministic=True,
            action_noise_sigma=0.0,
        )
        train_pi, _, _ = actor_model(
            forward_type=ForwardType.SAC,
            obs=rlt_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=float(
                cfg.algorithm.get("reference_dropout_prob", 0.0)
            ),
            **_sample_mode_kwargs(cfg, "actor_update_mode", "sac_sample"),
        )

    eval_flat = eval_pi.reshape(ref_flat.shape[0], -1)
    train_flat = train_pi.reshape(ref_flat.shape[0], -1)
    eval_bc = torch.mean(torch.square(eval_flat - ref_flat)).item()
    train_bc = torch.mean(torch.square(train_flat - ref_flat)).item()
    print(
        f"eval_bc_loss_to_ref={eval_bc:.8g} "
        f"weighted={bc_weight * eval_bc:.8g}"
    )
    print(
        f"train_actor_mode_bc_loss_to_ref={train_bc:.8g} "
        f"weighted={bc_weight * train_bc:.8g}"
    )
    compare_tensors("eval_action_minus_ref", eval_flat, ref_flat)
    compare_tensors("train_actor_mode_action_minus_ref", train_flat, ref_flat)


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)
    set_all_seeds(args.seed)

    cfg = load_cfg(args.config_name, args.config_dir)
    device = resolve_device(args.device)

    raw_state: dict[str, torch.Tensor] | None = None
    state_format = "none"
    ckpt_path: Path | None = None
    if args.actor_ckpt:
        ckpt_path, raw_state = load_checkpoint_state_dict(args.actor_ckpt)
        configure_head_type(cfg, raw_state)
        state_format, raw_state = normalize_checkpoint_state_dict(raw_state)

    actor_model = build_model(cfg.actor.model, device)
    rollout_model = build_model(cfg.rollout.model, device)

    print("== Config/Checkpoint ==")
    print(f"config_name={args.config_name}")
    print(f"device={device}")
    print(f"actor_model_type={cfg.actor.model.model_type}")
    print(f"rollout_model_type={cfg.rollout.model.model_type}")
    print(f"configured_rlt_head_type={cfg.actor.model.get('rlt_head_type', None)}")
    print(f"actor_ckpt={ckpt_path}")
    print(f"actor_ckpt_format={state_format}")

    suspected = False
    if raw_state is not None:
        direct_keys = select_keys(raw_state, "direct_actor.")
        q_keys = select_keys(raw_state, "q_head.")
        other_keys = [k for k in raw_state if k not in set(direct_keys) | set(q_keys)]
        print("\n== Checkpoint Key Summary ==")
        print(f"total_tensor_keys={len(raw_state)}")
        print(f"direct_actor_keys={len(direct_keys)}")
        print(f"q_head_keys={len(q_keys)}")
        print(f"other_tensor_keys={len(other_keys)}")
        print(f"direct_actor_key_sample={direct_keys[:8]}")
        print(f"q_head_key_sample={q_keys[:8]}")
        group_stats(raw_state, direct_keys, label="ckpt.direct_actor")
        group_stats(raw_state, q_keys, label="ckpt.q_head")
        if not direct_keys:
            suspected = True
            print("VERDICT_DIRECT_KEYS=FAIL no direct_actor keys found in checkpoint.")
        else:
            print("VERDICT_DIRECT_KEYS=PASS")

        print("\n== Load Into Actor/Rollout MLP ==")
        actor_missing, actor_unexpected = load_model_state(
            actor_model,
            raw_state,
            label="actor_load",
        )
        rollout_missing, rollout_unexpected = load_model_state(
            rollout_model,
            raw_state,
            label="rollout_load",
        )
        actor_sd = actor_model.state_dict()
        rollout_sd = rollout_model.state_dict()
        compare_state_groups(
            actor_sd,
            rollout_sd,
            select_keys(actor_sd, ("direct_actor.", "q_head.")),
            label="actor_loaded_vs_rollout_loaded",
        )
        if actor_missing or rollout_missing:
            hard_missing = [
                k
                for k in actor_missing + rollout_missing
                if k.startswith(("direct_actor.", "q_head."))
            ]
            if hard_missing:
                suspected = True
                print(f"VERDICT_LOAD=FAIL missing critical keys: {hard_missing[:12]}")
            else:
                print("VERDICT_LOAD=PASS critical keys loaded")
        else:
            print("VERDICT_LOAD=PASS")

        if args.base_actor_ckpt:
            base_path, base_raw = load_checkpoint_state_dict(args.base_actor_ckpt)
            base_format, base_state = normalize_checkpoint_state_dict(base_raw)
            print("\n== Base Checkpoint Delta ==")
            print(f"base_actor_ckpt={base_path}")
            print(f"base_actor_ckpt_format={base_format}")
            base_direct_keys = select_keys(base_state, "direct_actor.")
            base_q_keys = select_keys(base_state, "q_head.")
            group_stats(base_state, base_direct_keys, label="base.direct_actor")
            group_stats(base_state, base_q_keys, label="base.q_head")
            direct_delta = compare_state_groups(
                raw_state,
                base_state,
                direct_keys,
                label="current_minus_base.direct_actor",
            )
            q_delta = compare_state_groups(
                raw_state,
                base_state,
                q_keys,
                label="current_minus_base.q_head",
            )
            if direct_delta["common"] == 0 or direct_delta["max_abs"] <= 1e-8:
                suspected = True
                print("VERDICT_DIRECT_ACTOR_DELTA=FAIL direct_actor did not change vs base.")
            else:
                print("VERDICT_DIRECT_ACTOR_DELTA=PASS direct_actor changed vs base.")
            if q_delta["common"] == 0 or q_delta["max_abs"] <= 1e-8:
                print("VERDICT_Q_HEAD_DELTA=WARN q_head did not change vs base.")
            else:
                print("VERDICT_Q_HEAD_DELTA=PASS q_head changed vs base.")

    name_metrics = inspect_optimizer_and_sync_names(actor_model)
    if not name_metrics["direct_actor_trainable"] or not name_metrics["direct_actor_sync"]:
        suspected = True
        print("VERDICT_OPT_SYNC=FAIL direct_actor not trainable or not in sync names.")
    elif not name_metrics["q_head_trainable"] or not name_metrics["q_head_sync"]:
        suspected = True
        print("VERDICT_OPT_SYNC=FAIL q_head not trainable or not in sync names.")
    else:
        print("VERDICT_OPT_SYNC=PASS")

    should_do_obs = (args.with_obs or bool(args.obs_path) or bool(args.run_rollout_modes)) and not args.skip_obs
    if should_do_obs:
        try:
            feature_model = build_feature_model(cfg, device)
            rlt_obs, env = extract_rlt_obs(
                cfg,
                feature_model,
                device=device,
                obs_path=args.obs_path,
                num_envs=args.num_envs,
            )
            if env is not None:
                close_env(env)
            action_metrics = inspect_actions(cfg, actor_model, rollout_model, rlt_obs)
            if action_metrics["actor_eval_vs_rollout_eval_max"] > 1e-6:
                suspected = True
                print(
                    "VERDICT_ROLLOUT_LOAD_ACTION=FAIL actor and rollout loaded actions differ."
                )
            else:
                print("VERDICT_ROLLOUT_LOAD_ACTION=PASS")
            if action_metrics["actor_eval_vs_train_det_max"] > 1e-6:
                suspected = True
                print("VERDICT_EVAL_TRAIN_DET=FAIL eval and train_deterministic differ.")
            else:
                print("VERDICT_EVAL_TRAIN_DET=PASS")
            inspect_actor_loss_consistency(
                cfg,
                actor_model,
                rlt_obs,
                update_step=args.update_step,
            )

            rollout_modes = [
                m.strip() for m in args.run_rollout_modes.split(",") if m.strip()
            ]
            for mode_name in rollout_modes:
                print(f"\n== Rollout Mode: {mode_name} ==")
                run_rollout_mode(
                    cfg,
                    feature_model,
                    actor_model,
                    mode_name=mode_name,
                    split=args.rollout_env,
                    num_envs=args.num_envs,
                    chunks=args.rollout_chunks,
                    seed=args.seed,
                    device=device,
                )
        except RuntimeError as exc:
            if "CUDA error" not in str(exc):
                raise
            suspected = True
            print("\nVERDICT_CUDA_RUNTIME=FAIL")
            print(str(exc))
            print(
                "This happened inside the optional obs/rollout path. Re-run the "
                "checkpoint-only path with `--skip-obs --device cpu`, and if you "
                "need the obs path use a fresh process with `CUDA_LAUNCH_BLOCKING=1` "
                "and a small `--num-envs 1`."
            )

    print("\n== Final Verdict ==")
    if suspected:
        print(
            "SUSPECT: at least one direct actor update/load/sync-path check failed. "
            "Use the FAIL line above as the next target."
        )
        if args.fail_on_suspect:
            raise SystemExit(1)
    else:
        print(
            "PASS: no obvious offline direct-actor update/load/action-path issue. "
            "If eval still stays flat, the next suspect is runtime weight_syncer "
            "application inside the distributed training loop."
        )


if __name__ == "__main__":
    main()
