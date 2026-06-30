"""Evaluate unified OpenPI-RLT Stage1 checkpoints on ManiSkill data.

The checkpoint is the new unified actor directory saved by FSDP, for example:

    logs/.../checkpoints/global_step_1000/actor

The script reports both SFT losses and RL-token prefix-reconstruction metrics.
It does not require the old split ``actor/vla`` and ``actor/rl_token`` layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.openpi import get_model as get_openpi_model  # noqa: E402
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config  # noqa: E402
from rlinf.utils.pytree import register_pytree_dataclasses  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Unified Stage1 actor checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="LeRobot repo id or local ManiSkill dataset path.",
    )
    parser.add_argument("--config-name", default="pi05_rlt_maniskill_joint")
    parser.add_argument("--default-prompt", default="insert the peg in the hole")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-action-chunks", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--rlt-alpha", type=float, default=1.0)
    parser.add_argument("--rlt-input-dim", type=int, default=2048)
    parser.add_argument("--rlt-embed-dim", type=int, default=2048)
    parser.add_argument("--rlt-num-rl-tokens", type=int, default=1)
    parser.add_argument("--rlt-prefix-seq-len", type=int, default=1024)
    parser.add_argument("--rlt-num-layers", type=int, default=2)
    parser.add_argument("--rlt-num-heads", type=int, default=8)
    parser.add_argument("--rlt-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--rlt-image-only", action="store_true", default=False)
    parser.add_argument("--no-rlt-mask", action="store_true")
    parser.add_argument(
        "--use-action-chunk-loss",
        action="store_true",
        help="Match action loss to action_chunk/action_env_dim before averaging.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser.parse_args()


def _build_model_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "model_path": str(args.checkpoint),
            "precision": None,
            "model_type": "openpi",
            "num_action_chunks": int(args.num_action_chunks),
            "action_dim": int(args.action_dim),
            "add_value_head": False,
            "num_steps": int(args.num_steps),
            "openpi": {
                "config_name": str(args.config_name),
                "num_images_in_input": 2,
                "train_expert_only": False,
                "detach_critic_input": True,
                "action_horizon": int(args.num_action_chunks),
                "action_chunk": int(args.num_action_chunks),
                "action_env_dim": int(args.action_dim),
                "num_steps": int(args.num_steps),
                "use_rlt": True,
                "rlt_alpha": float(args.rlt_alpha),
                "rlt_input_dim": int(args.rlt_input_dim),
                "rlt_embed_dim": int(args.rlt_embed_dim),
                "rlt_num_rl_tokens": int(args.rlt_num_rl_tokens),
                "rlt_prefix_seq_len": int(args.rlt_prefix_seq_len),
                "rlt_num_layers": int(args.rlt_num_layers),
                "rlt_num_heads": int(args.rlt_num_heads),
                "rlt_mlp_ratio": float(args.rlt_mlp_ratio),
                "rlt_image_only": bool(args.rlt_image_only),
                "rlt_use_mask": not bool(args.no_rlt_mask),
            },
            "openpi_data": {
                "default_prompt": str(args.default_prompt),
            },
        }
    )


def _build_dataloader(args: argparse.Namespace):
    import openpi.training.data_loader as openpi_data_loader

    config = get_openpi_config(
        args.config_name,
        model_path=args.checkpoint,
        batch_size=args.batch_size,
        repo_id=args.dataset_path,
        data_kwargs={"default_prompt": args.default_prompt},
    )
    return openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
    )


def _to_device(batch: Any, device: torch.device) -> dict[str, Any]:
    observation, actions = batch
    register_pytree_dataclasses(observation)
    observation = tree_map(
        lambda x: (
            torch.as_tensor(x, device=device).contiguous().clone()
            if x is not None
            else x
        ),
        observation,
    )
    actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
    return {"observation": observation, "actions": actions}


def _masked_mse(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    sq_error = torch.square(reconstructed.to(torch.float32) - target.to(torch.float32))
    if mask is None:
        return sq_error.mean()

    mask_expanded = mask.to(device=sq_error.device, dtype=sq_error.dtype)[..., None]
    denom = torch.clamp(mask_expanded.sum() * target.shape[-1], min=1.0)
    return (sq_error * mask_expanded).sum() / denom


def _reconstruction_metrics(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    target = target.detach().to(torch.float32)
    reconstructed = reconstructed.to(torch.float32)
    mse = _masked_mse(reconstructed, target, mask)
    rmse = torch.sqrt(mse)

    if mask is None:
        target_rms = torch.sqrt(torch.mean(torch.square(target))).clamp(min=1e-12)
        cosine = F.cosine_similarity(
            reconstructed.reshape(-1, reconstructed.shape[-1]),
            target.reshape(-1, target.shape[-1]),
            dim=-1,
        ).mean()
    else:
        valid = mask.to(device=target.device, dtype=torch.bool)
        target_rms = torch.sqrt(_masked_mse(torch.zeros_like(target), target, valid))
        target_rms = target_rms.clamp(min=1e-12)
        cosine = F.cosine_similarity(reconstructed[valid], target[valid], dim=-1).mean()

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": rmse / target_rms,
        "cosine": cosine,
    }


@torch.no_grad()
def _evaluate_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace,
):
    loss, prefix_output, prefix_mask = model._sft_forward_with_rlt_prefix(
        batch["observation"],
        batch["actions"],
    )
    if args.use_action_chunk_loss:
        loss = loss[:, : model.config.action_chunk, : model.config.action_env_dim]
    vla_loss = loss.mean()

    rlt_param = next(model.rlt_module.parameters())
    prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
    rlt_mask = prefix_mask if model.config.rlt_use_mask else None
    reconstructed, rl_tokens = model.rlt_module.reconstruct(prefix_output, rlt_mask)
    rlt_loss = _masked_mse(reconstructed, prefix_output.detach(), rlt_mask)
    total_loss = rlt_loss + model.config.rlt_alpha * vla_loss
    metrics = _reconstruction_metrics(reconstructed, prefix_output.detach(), rlt_mask)

    return {
        "loss": total_loss,
        "vla_loss": vla_loss,
        "rlt_loss": rlt_loss,
        "z_rl_l2": torch.linalg.vector_norm(
            rl_tokens.reshape(rl_tokens.shape[0], -1),
            dim=-1,
        ).mean(),
        **metrics,
    }


def _mean_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: sum(values) / max(1, len(values))
        for key, values in sorted(metrics.items())
    }


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    model = get_openpi_model(_build_model_cfg(args))
    model.to(device)
    model.eval()
    if not getattr(model.config, "use_rlt", False) or not hasattr(model, "rlt_module"):
        raise ValueError("Loaded model is not configured with openpi.use_rlt=True.")

    data_loader = _build_dataloader(args)
    iterator = iter(data_loader)
    metrics: dict[str, list[float]] = {}

    for _ in range(args.num_batches):
        batch = _to_device(next(iterator), device)
        output = _evaluate_batch(model, batch, args)
        for key, value in output.items():
            metrics.setdefault(key, []).append(float(value.detach().cpu()))

    summary = _mean_metrics(metrics)
    summary["batch_size"] = int(args.batch_size)
    summary["num_batches"] = int(args.num_batches)
    summary["checkpoint"] = str(checkpoint)
    summary["dataset_path"] = str(args.dataset_path)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
