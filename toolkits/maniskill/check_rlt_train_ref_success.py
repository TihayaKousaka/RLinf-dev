#!/usr/bin/env python3
"""Probe ManiSkill train success for Stage2 reference-action paths.

This script intentionally checks the training rollout metric semantics
(`env/success_once`), not evaluation.  It runs the train environment with
reference actions only and compares several OpenPI construction paths that
should be equivalent if the refactor preserved ablation behavior.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
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
            "Run ManiSkill train env with ref-only actions and report the exact "
            "env/success_once-style metric for current vs ablation-like OpenPI "
            "reference paths."
        )
    )
    parser.add_argument(
        "--config-name",
        default="rlt_stage2_maniskill_joint",
        help="Hydra config under examples/embodiment/config.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "examples" / "embodiment" / "config"),
        help="Hydra config directory.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of train envs to run. Use 64 to match env/train by default.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        help="Number of action chunks. 50 chunks x 10 actions = 500 env steps.",
    )
    parser.add_argument(
        "--sources",
        default="current_feature,plain_same_data,plain_no_data",
        help=(
            "Comma-separated sources: current_feature, plain_same_data, "
            "plain_no_data. current_feature uses rollout.rlt_feature_model; "
            "plain_same_data disables use_rlt/rlt_module but keeps openpi_data; "
            "plain_no_data disables use_rlt/rlt_module and removes openpi_data."
        ),
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=0,
        help=(
            "If >=0, reset torch/numpy/random before each chunk as "
            "action_seed + chunk_idx so diffusion sampling is comparable."
        ),
    )
    parser.add_argument(
        "--use-fixed-reset-state-ids",
        action="store_true",
        help="Force train env use_fixed_reset_state_ids=True for controlled reset ids.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.env.train.seed.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print one chunk row every N chunks, plus first and last chunk.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for OpenPI model: auto, cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--compare-ckpt-tensors",
        action="store_true",
        help=(
            "Also load the checkpoint state dict and compare a few critical "
            "tensors against the constructed model. This is stronger, but can "
            "use a lot of CPU memory for .pt checkpoints."
        ),
    )
    parser.add_argument(
        "--inspect-data-config-only",
        action="store_true",
        help=(
            "Only print OpenPI DataConfig/norm_stats selection for --sources and "
            "exit. This does not build ManiSkill envs, run chunks, or load model "
            "weights."
        ),
    )
    parser.add_argument(
        "--compare-first-actions-only",
        action="store_true",
        help=(
            "Build one train env reset and compare the first ref action chunk "
            "from --sources on the same obs. This does not step the env."
        ),
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_cfg(config_name: str, config_dir: str) -> DictConfig:
    os.environ.setdefault("EMBODIED_PATH", str(REPO_ROOT / "examples" / "embodiment"))
    with initialize_config_dir(version_base="1.1", config_dir=str(Path(config_dir))):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def close_env(env) -> None:
    inner_env = getattr(env, "env", None)
    if hasattr(inner_env, "close"):
        inner_env.close()
    elif hasattr(env, "close"):
        env.close()


def build_train_env(
    cfg: DictConfig,
    *,
    num_envs: int,
    seed: int | None,
    use_fixed_reset_state_ids: bool,
):
    from rlinf.envs import get_env_cls

    env_cfg = copy.deepcopy(cfg.env.train)
    with open_dict(env_cfg):
        env_cfg.total_num_envs = num_envs
        env_cfg.group_size = 1
        env_cfg.init_params.num_envs = num_envs
        if seed is not None:
            env_cfg.seed = int(seed)
        if use_fixed_reset_state_ids:
            env_cfg.use_fixed_reset_state_ids = True
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


def build_current_feature_model(cfg: DictConfig, device: torch.device):
    from rlinf.models import get_model

    model_cfg = copy.deepcopy(cfg.rollout.rlt_feature_model)
    with open_dict(model_cfg):
        model_cfg.load_to_device = False
    model = get_model(model_cfg)
    model.to(device)
    model.eval()
    return model, model_cfg


def checkpoint_weight_files(path: str | None) -> list[Path]:
    if not path:
        return []
    ckpt_path = Path(path)
    if ckpt_path.is_file():
        return [ckpt_path]
    candidates = [
        ckpt_path / "model_state_dict" / "full_weights.pt",
        ckpt_path / "actor" / "model_state_dict" / "full_weights.pt",
        ckpt_path / "model.safetensors",
    ]
    if ckpt_path.is_dir():
        candidates.extend(sorted(ckpt_path.glob("*.safetensors")))
    seen = set()
    existing = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            existing.append(candidate)
    return existing


def print_path_summary(label: str, path: str | None) -> None:
    print(f"{label}={path}")
    if not path:
        return
    p = Path(path)
    print(f"{label}.exists={p.exists()} is_dir={p.is_dir()} is_file={p.is_file()}")
    files = checkpoint_weight_files(path)
    if not files:
        print(f"{label}.resolved_weight_files=[]")
        return
    print(f"{label}.resolved_weight_files:")
    for file_path in files:
        stat = file_path.stat()
        print(f"  - {file_path} size={stat.st_size / (1024**3):.3f}GiB")


def to_plain_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().reshape(-1)
    if value.numel() == 0:
        return "empty"
    sample = value[: min(4096, value.numel())].cpu()
    first = sample[: min(5, sample.numel())].tolist()
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"sample_mean={sample.mean().item():.8g} "
        f"sample_std={sample.std().item() if sample.numel() > 1 else 0.0:.8g} "
        f"sample_sum={sample.sum().item():.8g} "
        f"sample_absmax={sample.abs().max().item():.8g} "
        f"first={first}"
    )


def _summarize_stats_array(value: Any) -> str:
    if value is None:
        return "None"
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return "empty"
    flat = array.reshape(-1)
    first = flat[: min(5, flat.size)].tolist()
    return (
        f"shape={array.shape} mean={float(flat.mean()):.8g} "
        f"std={float(flat.std()):.8g} min={float(flat.min()):.8g} "
        f"max={float(flat.max()):.8g} first={first}"
    )


def _stat_container_keys(value: Any) -> list[str]:
    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return sorted(str(key) for key in value.keys())
    if is_dataclass(value):
        return sorted(field.name for field in fields(value))
    return sorted(
        key
        for key in ("mean", "std", "q01", "q99")
        if hasattr(value, key)
    )


def _stat_container_get(value: Any, key: str) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_stats_file_candidates(checkpoint_dir: str | Path, asset_id: str) -> list[Path]:
    root = Path(checkpoint_dir)
    asset_path = Path(*str(asset_id).split("/"))
    return [
        root / asset_path / "norm_stats.json",
        root / "norm_stats" / asset_path / "norm_stats.json",
        root / "stats" / asset_path / "norm_stats.json",
        root / "assets" / asset_path / "norm_stats.json",
        root / "assets" / "norm_stats" / asset_path / "norm_stats.json",
        root / "norm_stats.json",
    ]


def _print_norm_stats_file_candidates(
    checkpoint_dir: str | Path,
    asset_id: str,
) -> None:
    print("norm_stats_file_candidates:")
    seen = set()
    for path in _norm_stats_file_candidates(checkpoint_dir, asset_id):
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        if path.exists():
            stat = path.stat()
            print(
                f"  EXISTS {path} size={stat.st_size} "
                f"sha256={_sha256_file(path)}"
            )
        else:
            print(f"  MISSING {path}")


def _norm_stats_arrays(norm_stats: Any) -> dict[tuple[str, str], np.ndarray]:
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for top_key in _stat_container_keys(norm_stats):
        stats = _stat_container_get(norm_stats, top_key)
        if stats is None:
            continue
        for stat_key in ("mean", "std", "q01", "q99"):
            stat_value = _stat_container_get(stats, stat_key)
            if stat_value is None:
                continue
            arrays[(top_key, stat_key)] = np.asarray(stat_value, dtype=np.float64)
    return arrays


def _print_loaded_norm_stats_diff(
    source: str,
    base_source: str,
    current_stats: Any,
    base_stats: Any,
) -> None:
    current_arrays = _norm_stats_arrays(current_stats)
    base_arrays = _norm_stats_arrays(base_stats)
    common_keys = sorted(set(current_arrays) & set(base_arrays))
    if not common_keys:
        print(f"loaded_norm_stats_diff({source}-{base_source}): no_common_keys")
        return

    print(f"\n== Loaded NormStats Diff: {source} - {base_source} ==")
    for key in common_keys:
        current = current_arrays[key]
        base = base_arrays[key]
        label = f"{key[0]}.{key[1]}"
        if current.shape != base.shape:
            print(
                f"{label}: shape_mismatch current={current.shape} base={base.shape}"
            )
            continue
        diff = np.abs(current - base)
        print(
            f"{label}: max_abs={float(diff.max()):.8g} "
            f"mean_abs={float(diff.mean()):.8g}"
        )


def print_openpi_data_and_norm_stats(model_cfg: DictConfig) -> Any | None:
    print("\n== OpenPI DataConfig / NormStats Selection ==")
    try:
        import openpi.shared.download as download
        from openpi.training import checkpoints as _checkpoints

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    except Exception as exc:
        print(f"SKIP: failed to import OpenPI config helpers: {exc}")
        return

    config_name = getattr(model_cfg.openpi, "config_name", None)
    data_kwargs = getattr(model_cfg, "openpi_data", None)
    data_kwargs_plain = to_plain_container(data_kwargs) if data_kwargs is not None else None
    print(f"requested_config_name={config_name}")
    print(f"requested_data_kwargs={data_kwargs_plain}")
    try:
        actor_train_config = get_openpi_config(
            config_name,
            model_path=model_cfg.model_path,
            data_kwargs=data_kwargs,
        )
        actor_model_config = actor_train_config.model
        override_model_config_kwargs = model_cfg.openpi
        if override_model_config_kwargs is not None:
            actor_model_config = copy.deepcopy(actor_model_config)
            for key, val in override_model_config_kwargs.items():
                actor_model_config.__dict__[key] = val
        checkpoint_dir = download.maybe_download(str(model_cfg.model_path))
        data_config = actor_train_config.data.create(
            actor_train_config.assets_dirs,
            actor_model_config,
        )
        print(f"resolved_checkpoint_dir={checkpoint_dir}")
        print(f"train_config.name={actor_train_config.name}")
        print(f"train_config.assets_dirs={actor_train_config.assets_dirs}")
        print(f"data_factory_type={type(actor_train_config.data).__name__}")
        print(f"data_factory.repo_id={getattr(actor_train_config.data, 'repo_id', None)}")
        print(f"data_factory.assets={getattr(actor_train_config.data, 'assets', None)}")
        print(f"data_config.repo_id={getattr(data_config, 'repo_id', None)}")
        print(f"data_config.asset_id={getattr(data_config, 'asset_id', None)}")
        print(f"data_config.assets_dirs={getattr(data_config, 'assets_dirs', None)}")
        print(f"data_config.use_quantile_norm={getattr(data_config, 'use_quantile_norm', None)}")
        print(f"data_config.prompt_from_task={getattr(data_config, 'prompt_from_task', None)}")

        asset_id = getattr(data_config, "asset_id", None)
        if asset_id is None:
            print("norm_stats: SKIP asset_id=None")
            return None
        _print_norm_stats_file_candidates(checkpoint_dir, asset_id)
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, asset_id)
        print(f"norm_stats_asset_id={asset_id}")
        print(f"norm_stats_type={type(norm_stats).__name__}")
        print(f"norm_stats_keys={_stat_container_keys(norm_stats)}")
        for top_key in ("state", "actions", "action"):
            stats = _stat_container_get(norm_stats, top_key)
            if stats is None:
                continue
            print(f"norm_stats[{top_key}] type={type(stats).__name__}")
            print(f"norm_stats[{top_key}] keys={_stat_container_keys(stats)}")
            for stat_key in ("mean", "std", "q01", "q99"):
                stat_value = _stat_container_get(stats, stat_key)
                if stat_value is not None:
                    print(
                        f"norm_stats[{top_key}][{stat_key}]: "
                        f"{_summarize_stats_array(stat_value)}"
                    )
        return norm_stats
    except Exception as exc:
        print(f"ERROR while resolving OpenPI DataConfig/norm_stats: {type(exc).__name__}: {exc}")
        return None


def print_model_weight_fingerprints(model) -> None:
    state = model.state_dict()
    print("\n== Loaded Model Weight Fingerprints ==")
    print(f"state_dict_num_tensors={len(state)}")
    total_elems = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    print(f"state_dict_num_elements={total_elems}")
    exact_keys = [
        "action_in_proj.weight",
        "action_out_proj.weight",
        "state_proj.weight",
    ]
    printed = set()
    for key in exact_keys:
        value = state.get(key)
        if torch.is_tensor(value):
            print(f"{key}: {tensor_fingerprint(value)}")
            printed.add(key)
    for needle in (
        "paligemma_with_expert.paligemma",
        "paligemma_with_expert.gemma_expert",
        "rlt_module",
    ):
        for key, value in state.items():
            if key in printed:
                continue
            if needle in key and torch.is_tensor(value):
                print(f"{key}: {tensor_fingerprint(value)}")
                printed.add(key)
                break
    if not printed:
        print("WARN: no selected model tensors found for fingerprinting.")


def _unwrap_checkpoint_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                return nested
        return obj
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


def _candidate_ckpt_keys(model_key: str) -> list[str]:
    return [
        model_key,
        f"module.{model_key}",
        f"_fsdp_wrapped_module.{model_key}",
        f"model.{model_key}",
    ]


def compare_loaded_tensors_with_checkpoint(model_cfg: DictConfig, model) -> None:
    files = checkpoint_weight_files(str(model_cfg.get("model_path", "")))
    if not files:
        print("\n== Checkpoint Tensor Compare ==")
        print("SKIP: no checkpoint weight files resolved.")
        return

    weight_file = files[0]
    print("\n== Checkpoint Tensor Compare ==")
    print(f"checkpoint_file={weight_file}")
    model_state = model.state_dict()
    keys_to_check = [
        key
        for key in (
            "action_in_proj.weight",
            "action_out_proj.weight",
            "state_proj.weight",
        )
        if key in model_state
    ]
    if not keys_to_check:
        print("SKIP: selected model keys not found.")
        return

    if weight_file.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            ckpt_keys = set(handle.keys())
            for model_key in keys_to_check:
                ckpt_key = next(
                    (key for key in _candidate_ckpt_keys(model_key) if key in ckpt_keys),
                    None,
                )
                if ckpt_key is None:
                    print(f"{model_key}: MISSING_IN_CKPT")
                    continue
                ckpt_tensor = handle.get_tensor(ckpt_key)
                compare_one_loaded_tensor(model_key, ckpt_key, model_state[model_key], ckpt_tensor)
        return

    print("loading .pt checkpoint for tensor compare; this may use significant CPU RAM")
    checkpoint = torch.load(weight_file, map_location="cpu", weights_only=False)
    checkpoint_state = _unwrap_checkpoint_state_dict(checkpoint)
    ckpt_keys = set(checkpoint_state)
    for model_key in keys_to_check:
        ckpt_key = next(
            (key for key in _candidate_ckpt_keys(model_key) if key in ckpt_keys),
            None,
        )
        if ckpt_key is None:
            print(f"{model_key}: MISSING_IN_CKPT")
            continue
        compare_one_loaded_tensor(
            model_key,
            ckpt_key,
            model_state[model_key],
            checkpoint_state[ckpt_key],
        )
    del checkpoint
    del checkpoint_state
    gc.collect()


def compare_one_loaded_tensor(
    model_key: str,
    ckpt_key: str,
    model_tensor: torch.Tensor,
    ckpt_tensor: torch.Tensor,
) -> None:
    model_value = model_tensor.detach().float().cpu()
    ckpt_value = ckpt_tensor.detach().float().cpu()
    if model_value.shape != ckpt_value.shape:
        print(
            f"{model_key}: SHAPE_MISMATCH model={tuple(model_value.shape)} "
            f"ckpt[{ckpt_key}]={tuple(ckpt_value.shape)}"
        )
        return
    diff = (model_value - ckpt_value).abs()
    print(
        f"{model_key} vs ckpt[{ckpt_key}]: "
        f"max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g}"
    )


def _delete_key(cfg: DictConfig, key: str) -> None:
    if key in cfg:
        with open_dict(cfg):
            del cfg[key]


def build_current_feature_model_cfg(cfg: DictConfig) -> DictConfig:
    model_cfg = copy.deepcopy(cfg.rollout.rlt_feature_model)
    with open_dict(model_cfg):
        model_cfg.load_to_device = False
    return model_cfg


def build_plain_vla_model_cfg(
    cfg: DictConfig,
    *,
    keep_openpi_data: bool,
) -> DictConfig:
    model_cfg = copy.deepcopy(cfg.rollout.rlt_feature_model)
    with open_dict(model_cfg):
        model_cfg.load_to_device = False
        if "rlt_module_path" in model_cfg:
            del model_cfg.rlt_module_path
        if "rlt_token_path" in model_cfg:
            del model_cfg.rlt_token_path
        model_cfg.openpi.use_rlt = False
        model_cfg.openpi.add_value_head = False
        model_cfg.openpi.train_expert_only = False
        if not keep_openpi_data and "openpi_data" in model_cfg:
            del model_cfg.openpi_data
    return model_cfg


def build_source_model_cfg(cfg: DictConfig, source: str) -> DictConfig:
    if source == "current_feature":
        return build_current_feature_model_cfg(cfg)
    if source == "plain_same_data":
        return build_plain_vla_model_cfg(cfg, keep_openpi_data=True)
    if source == "plain_no_data":
        return build_plain_vla_model_cfg(cfg, keep_openpi_data=False)
    raise ValueError(f"Unknown source: {source}")


def build_plain_vla_model(
    cfg: DictConfig,
    *,
    device: torch.device,
    keep_openpi_data: bool,
):
    from rlinf.models import get_model

    src_cfg = build_plain_vla_model_cfg(cfg, keep_openpi_data=keep_openpi_data)
    model = get_model(src_cfg)
    model.to(device)
    model.eval()
    return model, src_cfg


def model_summary(
    name: str,
    model_cfg: DictConfig,
    model,
    *,
    compare_ckpt_tensors: bool,
) -> None:
    print(f"\n== Source Config: {name} ==")
    print_path_summary("model_path", model_cfg.get("model_path", None))
    print_path_summary("rlt_module_path", model_cfg.get("rlt_module_path", None))
    print(f"openpi_data={to_plain_container(model_cfg.get('openpi_data', {}))}")
    openpi_cfg = model_cfg.get("openpi", {})
    for key in (
        "config_name",
        "use_rlt",
        "rlt_image_only",
        "rlt_use_mask",
        "action_chunk",
        "action_horizon",
        "action_env_dim",
        "num_steps",
        "num_images_in_input",
    ):
        print(f"openpi.{key}={openpi_cfg.get(key, None)}")
    if hasattr(model, "config"):
        for key in (
            "config_name",
            "use_rlt",
            "rlt_image_only",
            "rlt_use_mask",
            "action_chunk",
            "action_horizon",
            "action_env_dim",
            "num_steps",
            "num_images_in_input",
        ):
            print(f"model.config.{key}={getattr(model.config, key, None)}")
    print_openpi_data_and_norm_stats(model_cfg)
    print_model_weight_fingerprints(model)
    if compare_ckpt_tensors:
        compare_loaded_tensors_with_checkpoint(model_cfg, model)


def as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach()
    return torch.as_tensor(value)


def to_float_cpu(value: Any) -> torch.Tensor:
    return as_tensor(value).detach().float().cpu()


def bool_rate(value: Any) -> float:
    tensor = as_tensor(value).detach().bool().float()
    return float(tensor.mean().item()) if tensor.numel() else 0.0


def action_stats(name: str, value: Any) -> None:
    tensor = to_float_cpu(value)
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"mean={tensor.mean().item():.6g} std={tensor.std().item():.6g} "
        f"min={tensor.min().item():.6g} max={tensor.max().item():.6g} "
        f"abs_mean={tensor.abs().mean().item():.6g} "
        f"abs_max={tensor.abs().max().item():.6g}"
    )


def get_episode_ids(env) -> torch.Tensor | None:
    base_env = getattr(env, "env", env)
    unwrapped = getattr(base_env, "unwrapped", base_env)
    value = getattr(unwrapped, "episode_id", None)
    if value is None:
        value = getattr(env, "reset_state_ids", None)
    if value is None:
        return None
    return as_tensor(value).detach().cpu().reshape(-1)


def print_reset_summary(env, obs: dict[str, Any]) -> None:
    print("\n== Train Reset Summary ==")
    print(f"reset_state_ids[:16]={getattr(env, 'reset_state_ids', None)[:16].detach().cpu().tolist() if hasattr(env, 'reset_state_ids') else None}")
    episode_ids = get_episode_ids(env)
    if episode_ids is not None:
        print(f"episode_ids[:16]={episode_ids[:16].tolist()}")
        print(
            f"episode_ids_unique={int(torch.unique(episode_ids).numel())} "
            f"episode_ids_min={int(episode_ids.min().item())} "
            f"episode_ids_max={int(episode_ids.max().item())}"
        )
    for key in ("states", "main_images", "wrist_images", "extra_view_images"):
        if key not in obs:
            continue
        value = obs[key]
        if torch.is_tensor(value):
            print(f"obs.{key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
        elif value is None:
            print(f"obs.{key}: None")
        else:
            print(f"obs.{key}: {type(value).__name__}")
    if "task_descriptions" in obs:
        print(f"task_descriptions[:4]={obs['task_descriptions'][:4]}")


@torch.no_grad()
def ref_actions_from_current_feature(
    cfg: DictConfig,
    model,
    obs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rlt_obs = model.extract_rlt_stage2_obs(obs)
    ref_chunk = rlt_obs["ref_chunk"]
    action_dim = int(cfg.rlt.action_dim)
    chunk_len = int(cfg.rlt.num_action_chunks)
    if ref_chunk.dim() == 2:
        ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
    actions = ref_chunk[:, :chunk_len, :action_dim].contiguous()
    return actions, rlt_obs


@torch.no_grad()
def ref_actions_from_plain_vla(
    cfg: DictConfig,
    model,
    obs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from openpi.models import model as openpi_model

    to_process_obs = model.obs_processor(obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = openpi_model.Observation.from_dict(processed_obs)
    outputs = model.sample_actions(
        observation,
        mode="eval",
        compute_values=False,
    )
    actions = model.output_transform(
        {"actions": outputs["actions"], "state": observation.state}
    )["actions"]
    action_dim = int(cfg.rlt.action_dim)
    chunk_len = int(cfg.rlt.num_action_chunks)
    if actions.dim() == 2:
        actions = actions.reshape(actions.shape[0], -1, action_dim)
    actions = actions[:, :chunk_len, :action_dim].contiguous()
    return actions.to(dtype=torch.float32), {
        "ref_chunk": actions.to(dtype=torch.float32),
        "openpi_state": observation.state.detach().to(dtype=torch.float32),
    }


def prepare_for_env(cfg: DictConfig, raw_ref_actions: torch.Tensor):
    from rlinf.envs.action_utils import prepare_actions

    return prepare_actions(
        raw_chunk_actions=raw_ref_actions,
        env_type=cfg.env.train.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=int(cfg.rlt.num_action_chunks),
        action_dim=int(cfg.rlt.action_dim),
        policy=cfg.actor.model.get("policy_setup", None),
        wm_env_type=cfg.env.train.get("wm_env_type", None),
    )


def info_value(infos: dict[str, Any], key: str) -> Any | None:
    if not isinstance(infos, dict):
        return None
    if key in infos:
        return infos[key]
    episode = infos.get("episode", None)
    if isinstance(episode, dict) and key in episode:
        return episode[key]
    return None


def collect_chunk_success(infos_list: list[dict[str, Any]]) -> torch.Tensor | None:
    values = []
    for infos in infos_list:
        value = info_value(infos, "success")
        if value is not None:
            values.append(as_tensor(value).detach().bool().cpu())
    if not values:
        return None
    return torch.stack(values, dim=1)


def run_source(
    cfg: DictConfig,
    *,
    source: str,
    num_envs: int,
    max_chunks: int,
    action_seed: int,
    seed: int | None,
    use_fixed_reset_state_ids: bool,
    print_every: int,
    device: torch.device,
    compare_ckpt_tensors: bool,
) -> dict[str, Any]:
    if source == "current_feature":
        model, model_cfg = build_current_feature_model(cfg, device)
        get_actions = ref_actions_from_current_feature
    elif source == "plain_same_data":
        model, model_cfg = build_plain_vla_model(
            cfg,
            device=device,
            keep_openpi_data=True,
        )
        get_actions = ref_actions_from_plain_vla
    elif source == "plain_no_data":
        model, model_cfg = build_plain_vla_model(
            cfg,
            device=device,
            keep_openpi_data=False,
        )
        get_actions = ref_actions_from_plain_vla
    else:
        raise ValueError(f"Unknown source: {source}")

    model_summary(
        source,
        model_cfg,
        model,
        compare_ckpt_tensors=compare_ckpt_tensors,
    )
    env, env_cfg = build_train_env(
        cfg,
        num_envs=num_envs,
        seed=seed,
        use_fixed_reset_state_ids=use_fixed_reset_state_ids,
    )
    print("\n== Train Env Config ==")
    print(f"seed={env_cfg.seed}")
    print(f"use_fixed_reset_state_ids={env_cfg.use_fixed_reset_state_ids}")
    print(f"auto_reset={env_cfg.auto_reset}")
    print(f"ignore_terminations={env_cfg.ignore_terminations}")
    print(f"task={env_cfg.init_params.id}")
    print(f"control_mode={env_cfg.init_params.control_mode}")

    success_once = torch.zeros(num_envs, dtype=torch.bool)
    done_once = torch.zeros(num_envs, dtype=torch.bool)
    terminal_success = torch.zeros(num_envs, dtype=torch.bool)
    total_reward = torch.zeros(num_envs, dtype=torch.float32)
    last_success_once_from_info = torch.zeros(num_envs, dtype=torch.bool)
    last_return = torch.zeros(num_envs, dtype=torch.float32)
    chunks_run = 0
    first_raw_actions_cpu = None
    first_env_actions_cpu = None
    reset_episode_ids_cpu = None

    try:
        obs, _infos = env.reset()
        print_reset_summary(env, obs)
        reset_episode_ids_cpu = get_episode_ids(env)

        for chunk_idx in range(max_chunks):
            if action_seed >= 0:
                set_all_seeds(action_seed + chunk_idx)

            raw_actions, extra = get_actions(cfg, model, obs)
            if chunk_idx == 0:
                first_raw_actions_cpu = to_float_cpu(raw_actions)
                action_stats("raw_ref_actions_first_chunk", raw_actions)
                if "openpi_state" in extra:
                    action_stats("openpi_state_first_batch", extra["openpi_state"])
                if "proprio" in extra:
                    action_stats("rlt_proprio_first_batch", extra["proprio"])
                if "z_rl" in extra:
                    action_stats("z_rl_first_batch", extra["z_rl"])

            env_actions = prepare_for_env(cfg, raw_actions)
            if chunk_idx == 0:
                first_env_actions_cpu = to_float_cpu(env_actions)
                action_stats("env_prepared_actions_first_chunk", env_actions)
                prepared_t = to_float_cpu(env_actions)
                raw_t = to_float_cpu(raw_actions)
                if prepared_t.shape == raw_t.shape:
                    diff = (prepared_t - raw_t).abs()
                    print(
                        "prepared_minus_raw: "
                        f"max_abs={diff.max().item():.8g} "
                        f"mean_abs={diff.mean().item():.8g}"
                    )
                else:
                    print(
                        "prepared_minus_raw: shape_mismatch "
                        f"prepared={tuple(prepared_t.shape)} raw={tuple(raw_t.shape)}"
                    )

            obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
                env_actions
            )
            obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
            infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) else {}

            rewards_t = to_float_cpu(rewards)
            term_t = as_tensor(terminations).detach().bool().cpu()
            trunc_t = as_tensor(truncations).detach().bool().cpu()
            dones_t = term_t | trunc_t
            chunk_success_t = collect_chunk_success(infos_list)
            if chunk_success_t is not None:
                success_once |= chunk_success_t.any(dim=1)
            done_once |= dones_t.any(dim=1)
            terminal_success |= term_t.any(dim=1)
            total_reward += rewards_t.sum(dim=1)

            info_success_once = info_value(infos, "success_once")
            if info_success_once is not None:
                last_success_once_from_info = (
                    as_tensor(info_success_once).detach().bool().cpu().reshape(-1)
                )
                success_once |= last_success_once_from_info
            info_return = info_value(infos, "return")
            if info_return is not None:
                last_return = to_float_cpu(info_return).reshape(-1)

            chunks_run = chunk_idx + 1
            should_print = (
                chunk_idx == 0
                or chunk_idx == max_chunks - 1
                or (print_every > 0 and chunk_idx % print_every == 0)
                or done_once.all()
            )
            if should_print:
                print(
                    "chunk "
                    f"{chunk_idx:03d}: "
                    f"reward_sum_mean={rewards_t.sum(dim=1).mean().item():.6g} "
                    f"reward_nonzero_rate={rewards_t.ne(0).float().mean().item():.6g} "
                    f"raw_success_any_rate={success_once.float().mean().item():.6g} "
                    f"info_success_once_rate={last_success_once_from_info.float().mean().item():.6g} "
                    f"done_once_rate={done_once.float().mean().item():.6g} "
                    f"terminal_success_rate={terminal_success.float().mean().item():.6g} "
                    f"return_mean={last_return.mean().item():.6g} "
                    f"return_max={last_return.max().item():.6g}"
                )
            if done_once.all():
                break
    finally:
        close_env(env)
        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "success_once_raw": float(success_once.float().mean().item()),
        "success_once_info": float(last_success_once_from_info.float().mean().item()),
        "done_once": float(done_once.float().mean().item()),
        "terminal_success": float(terminal_success.float().mean().item()),
        "total_reward_mean": float(total_reward.mean().item()),
        "total_reward_max": float(total_reward.max().item()),
        "last_return_mean": float(last_return.mean().item()),
        "chunks_run": float(chunks_run),
        "_first_raw_actions": first_raw_actions_cpu,
        "_first_env_actions": first_env_actions_cpu,
        "_reset_episode_ids": reset_episode_ids_cpu,
    }
    print(f"\n== Source Summary: {source} ==")
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        print(f"{key}={value:.8g}")
    return summary


def inspect_data_configs_only(cfg: DictConfig, sources: list[str]) -> None:
    print("\n== DataConfig / NormStats Only ==")
    loaded_stats: dict[str, Any] = {}
    for source in sources:
        print(f"\n######## Source: {source} ########")
        model_cfg = build_source_model_cfg(cfg, source)
        print_path_summary("model_path", model_cfg.get("model_path", None))
        print_path_summary("rlt_module_path", model_cfg.get("rlt_module_path", None))
        print(f"openpi_data={to_plain_container(model_cfg.get('openpi_data', {}))}")
        openpi_cfg = model_cfg.get("openpi", {})
        for key in (
            "config_name",
            "use_rlt",
            "rlt_image_only",
            "rlt_use_mask",
            "action_chunk",
            "action_horizon",
            "action_env_dim",
            "num_steps",
            "num_images_in_input",
        ):
            print(f"openpi.{key}={openpi_cfg.get(key, None)}")
        norm_stats = print_openpi_data_and_norm_stats(model_cfg)
        if norm_stats is not None:
            loaded_stats[source] = norm_stats

    if len(loaded_stats) >= 2:
        base_source = next(source for source in sources if source in loaded_stats)
        for source in sources:
            if source == base_source or source not in loaded_stats:
                continue
            _print_loaded_norm_stats_diff(
                source,
                base_source,
                loaded_stats[source],
                loaded_stats[base_source],
            )


def build_source_model_and_action_getter(
    cfg: DictConfig,
    source: str,
    device: torch.device,
):
    if source == "current_feature":
        model, model_cfg = build_current_feature_model(cfg, device)
        return model, model_cfg, ref_actions_from_current_feature
    if source == "plain_same_data":
        model, model_cfg = build_plain_vla_model(
            cfg,
            device=device,
            keep_openpi_data=True,
        )
        return model, model_cfg, ref_actions_from_plain_vla
    if source == "plain_no_data":
        model, model_cfg = build_plain_vla_model(
            cfg,
            device=device,
            keep_openpi_data=False,
        )
        return model, model_cfg, ref_actions_from_plain_vla
    raise ValueError(f"Unknown source: {source}")


def _print_tensor_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if lhs.shape != rhs.shape:
        print(f"{name}: shape_mismatch lhs={tuple(lhs.shape)} rhs={tuple(rhs.shape)}")
        return
    diff = (lhs - rhs).abs()
    print(
        f"{name}: max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g} "
        f"p95_abs={torch.quantile(diff.reshape(-1), 0.95).item():.8g}"
    )


def compare_first_actions_only(
    cfg: DictConfig,
    *,
    sources: list[str],
    num_envs: int,
    action_seed: int,
    seed: int | None,
    use_fixed_reset_state_ids: bool,
    device: torch.device,
) -> None:
    print("\n== First Ref Action Chunk Only ==")
    env, env_cfg = build_train_env(
        cfg,
        num_envs=num_envs,
        seed=seed,
        use_fixed_reset_state_ids=use_fixed_reset_state_ids,
    )
    results: dict[str, dict[str, torch.Tensor | None]] = {}
    try:
        obs, _infos = env.reset()
        print("\n== Train Env Config ==")
        print(f"seed={env_cfg.seed}")
        print(f"use_fixed_reset_state_ids={env_cfg.use_fixed_reset_state_ids}")
        print(f"auto_reset={env_cfg.auto_reset}")
        print(f"ignore_terminations={env_cfg.ignore_terminations}")
        print(f"task={env_cfg.init_params.id}")
        print(f"control_mode={env_cfg.init_params.control_mode}")
        print_reset_summary(env, obs)

        for source in sources:
            print(f"\n######## Source First Actions: {source} ########")
            model = None
            try:
                model, model_cfg, get_actions = build_source_model_and_action_getter(
                    cfg,
                    source,
                    device,
                )
                print(f"openpi_data={to_plain_container(model_cfg.get('openpi_data', {}))}")
                if action_seed >= 0:
                    set_all_seeds(action_seed)
                raw_actions, extra = get_actions(cfg, model, obs)
                env_actions = prepare_for_env(cfg, raw_actions)
                action_stats("raw_ref_actions_first_chunk", raw_actions)
                action_stats("env_prepared_actions_first_chunk", env_actions)
                if "openpi_state" in extra:
                    action_stats("openpi_state_first_batch", extra["openpi_state"])
                if "proprio" in extra:
                    action_stats("rlt_proprio_first_batch", extra["proprio"])
                if "z_rl" in extra:
                    action_stats("z_rl_first_batch", extra["z_rl"])
                prepared_t = to_float_cpu(env_actions)
                raw_t = to_float_cpu(raw_actions)
                if prepared_t.shape == raw_t.shape:
                    _print_tensor_diff("prepared_minus_raw", prepared_t, raw_t)
                results[source] = {
                    "raw": raw_t,
                    "env": prepared_t,
                    "openpi_state": (
                        to_float_cpu(extra["openpi_state"])
                        if "openpi_state" in extra
                        else None
                    ),
                    "proprio": (
                        to_float_cpu(extra["proprio"]) if "proprio" in extra else None
                    ),
                }
            finally:
                if model is not None:
                    model.to("cpu")
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        close_env(env)

    if len(results) < 2:
        return

    print("\n== First Action Cross-Source Diff ==")
    base_name = sources[0]
    base = results[base_name]
    for source in sources[1:]:
        current = results[source]
        _print_tensor_diff(
            f"raw_ref_action_diff({source}-{base_name})",
            current["raw"],
            base["raw"],
        )
        _print_tensor_diff(
            f"env_action_diff({source}-{base_name})",
            current["env"],
            base["env"],
        )
        if current.get("openpi_state") is not None and base.get("openpi_state") is not None:
            _print_tensor_diff(
                f"openpi_state_diff({source}-{base_name})",
                current["openpi_state"],
                base["openpi_state"],
            )
        if current.get("proprio") is not None and base.get("proprio") is not None:
            _print_tensor_diff(
                f"proprio_diff({source}-{base_name})",
                current["proprio"],
                base["proprio"],
            )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_name, args.config_dir)
    device = resolve_device(args.device)
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]

    print("== Probe Setup ==")
    print(f"config_name={args.config_name}")
    print(f"sources={sources}")
    print(f"num_envs={args.num_envs}")
    print(f"max_chunks={args.max_chunks}")
    print(f"action_seed={args.action_seed}")
    print(f"device={device}")
    print("metric_scope=train env/success_once, not eval/success_once")

    if args.inspect_data_config_only:
        inspect_data_configs_only(cfg, sources)
        return

    if args.compare_first_actions_only:
        compare_first_actions_only(
            cfg,
            sources=sources,
            num_envs=args.num_envs,
            action_seed=args.action_seed,
            seed=args.seed,
            use_fixed_reset_state_ids=args.use_fixed_reset_state_ids,
            device=device,
        )
        return

    summaries = {}
    for source in sources:
        summaries[source] = run_source(
            cfg,
            source=source,
            num_envs=args.num_envs,
            max_chunks=args.max_chunks,
            action_seed=args.action_seed,
            seed=args.seed,
            use_fixed_reset_state_ids=args.use_fixed_reset_state_ids,
            print_every=args.print_every,
            device=device,
            compare_ckpt_tensors=args.compare_ckpt_tensors,
        )

    print("\n== Cross-Source Comparison ==")
    for source, summary in summaries.items():
        print(
            f"{source}: "
            f"success_once_info={summary['success_once_info']:.8g} "
            f"success_once_raw={summary['success_once_raw']:.8g} "
            f"terminal_success={summary['terminal_success']:.8g} "
            f"last_return_mean={summary['last_return_mean']:.8g}"
        )

    if len(summaries) >= 2:
        base_name = sources[0]
        base = summaries[base_name]["success_once_info"]
        for source in sources[1:]:
            diff = summaries[source]["success_once_info"] - base
            print(f"delta_success_once_info({source}-{base_name})={diff:.8g}")

        base_episode_ids = summaries[base_name].get("_reset_episode_ids")
        base_raw_actions = summaries[base_name].get("_first_raw_actions")
        base_env_actions = summaries[base_name].get("_first_env_actions")
        for source in sources[1:]:
            episode_ids = summaries[source].get("_reset_episode_ids")
            raw_actions = summaries[source].get("_first_raw_actions")
            env_actions = summaries[source].get("_first_env_actions")
            if base_episode_ids is not None and episode_ids is not None:
                same_ids = torch.equal(base_episode_ids, episode_ids)
                diff_count = int((base_episode_ids != episode_ids).sum().item())
                print(
                    f"reset_episode_ids_equal({source},{base_name})="
                    f"{same_ids} diff_count={diff_count}"
                )
            if base_raw_actions is not None and raw_actions is not None:
                if base_raw_actions.shape == raw_actions.shape:
                    diff_t = (raw_actions - base_raw_actions).abs()
                    print(
                        f"first_raw_action_diff({source}-{base_name}): "
                        f"max_abs={diff_t.max().item():.8g} "
                        f"mean_abs={diff_t.mean().item():.8g}"
                    )
                else:
                    print(
                        f"first_raw_action_diff({source}-{base_name}): "
                        f"shape_mismatch {tuple(raw_actions.shape)} vs "
                        f"{tuple(base_raw_actions.shape)}"
                    )
            if base_env_actions is not None and env_actions is not None:
                if base_env_actions.shape == env_actions.shape:
                    diff_t = (env_actions - base_env_actions).abs()
                    print(
                        f"first_env_action_diff({source}-{base_name}): "
                        f"max_abs={diff_t.max().item():.8g} "
                        f"mean_abs={diff_t.mean().item():.8g}"
                    )
                else:
                    print(
                        f"first_env_action_diff({source}-{base_name}): "
                        f"shape_mismatch {tuple(env_actions.shape)} vs "
                        f"{tuple(base_env_actions.shape)}"
                    )


if __name__ == "__main__":
    main()
