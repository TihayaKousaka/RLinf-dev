"""Evaluate RLT Stage 1 RL-token reconstruction on a LeRobot dataset.

This script is offline-only: it loads the frozen SFT VLA, a Stage 1
``rl_token_model.pt`` checkpoint, and reports how well the RL token reconstructs
the VLA prefix embeddings on sampled dataset batches.

Example:

    python toolkits/realworld_rlt/evaluate_stage1_reconstruction.py \
        --dataset-path /path/to/realworld_ee_lerobot \
        --vla-checkpoint /path/to/rlt_realworld_ee_pi05_sft/checkpoints/global_step_5000/actor \
        --rl-token-checkpoint /path/to/rlt_stage1_realworld_ee/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt \
        --norm-stats-path /path/to/realworld_ee_lerobot/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi_rlt.stage1_policy import RLTStage1Policy
from rlinf.utils.pytree import register_pytree_dataclasses


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--vla-checkpoint", required=True)
    parser.add_argument("--rl-token-checkpoint", required=True)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--config-name", default="pi05_rlt_realworld_ee")
    parser.add_argument("--default-prompt", default="insert the peg in the hole")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--num-action-chunks", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=5)
    return parser.parse_args()


def _build_stage1_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "model_type": "rlt_stage1",
            "model_path": str(args.vla_checkpoint),
            "precision": None,
            "num_action_chunks": int(args.num_action_chunks),
            "action_dim": int(args.action_dim),
            "rlt_stage1": {
                "config_name": str(args.config_name),
                "norm_stats_path": args.norm_stats_path,
                "num_images_in_input": 2,
                "num_steps": int(args.num_steps),
                "embedding_dim": int(args.embedding_dim),
                "encoder_layers": int(args.encoder_layers),
                "encoder_heads": int(args.encoder_heads),
                "decoder_layers": int(args.decoder_layers),
                "decoder_heads": int(args.decoder_heads),
            },
        }
    )


def _load_rl_token(policy: RLTStage1Policy, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = policy.rl_token_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[WARN] Loaded RL token with non-strict key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _build_dataloader(args: argparse.Namespace):
    import openpi.training.data_loader as openpi_data_loader

    data_kwargs: dict[str, Any] = {"default_prompt": args.default_prompt}
    if args.norm_stats_path is not None:
        data_kwargs["norm_stats_path"] = args.norm_stats_path
    config = get_openpi_config(
        args.config_name,
        model_path=args.vla_checkpoint,
        batch_size=args.batch_size,
        repo_id=args.dataset_path,
        data_kwargs=data_kwargs,
    )
    data_loader = openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
    )
    return data_loader


def _to_device(batch, device: torch.device):
    observation, actions = batch
    register_pytree_dataclasses(observation)
    observation = tree_map(
        lambda x: torch.as_tensor(x, device=device).contiguous().clone()
        if x is not None
        else x,
        observation,
    )
    actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
    return {"observation": observation, "actions": actions}


def _update_metric(metrics: dict[str, list[float]], output: dict[str, torch.Tensor]) -> None:
    z = output["_z"].detach().to(torch.float32)
    z_hat = output["z_hat"].detach().to(torch.float32)
    pad_mask = output["_pad_mask"].detach().to(torch.bool)
    valid = pad_mask.unsqueeze(-1)
    diff = (z_hat - z)[valid.expand_as(z)]
    target = z[valid.expand_as(z)]

    mse = diff.square().mean()
    rmse = mse.sqrt()
    target_rms = target.square().mean().sqrt().clamp(min=1e-12)
    rel_rmse = rmse / target_rms
    cosine = torch.nn.functional.cosine_similarity(
        z_hat[pad_mask],
        z[pad_mask],
        dim=-1,
    ).mean()

    metrics["l_ro"].append(float(output["l_ro"].detach().cpu()))
    metrics["mse"].append(float(mse.cpu()))
    metrics["rmse"].append(float(rmse.cpu()))
    metrics["relative_rmse"].append(float(rel_rmse.cpu()))
    metrics["cosine"].append(float(cosine.cpu()))


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    for path_arg in ("dataset_path", "vla_checkpoint", "rl_token_checkpoint"):
        path = Path(getattr(args, path_arg)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path_arg} does not exist: {path}")
    if args.norm_stats_path is not None and not Path(args.norm_stats_path).exists():
        raise FileNotFoundError(f"norm_stats_path does not exist: {args.norm_stats_path}")

    policy = RLTStage1Policy(_build_stage1_cfg(args), device=device)
    _load_rl_token(policy, args.rl_token_checkpoint)
    policy.eval()

    data_loader = _build_dataloader(args)
    metrics: dict[str, list[float]] = {
        "l_ro": [],
        "mse": [],
        "rmse": [],
        "relative_rmse": [],
        "cosine": [],
    }

    iterator = iter(data_loader)
    with torch.no_grad():
        for _ in tqdm.trange(args.num_batches, desc="Evaluating Stage1 reconstruction"):
            batch = _to_device(next(iterator), device)
            z, pad_mask = policy.vla.extract_rlt_prefix_embeddings(
                batch["observation"],
                dtype=torch.float32,
            )
            l_ro, z_rl, z_hat = policy.rl_token_model(
                z.to(device),
                pad_mask.to(device),
            )
            output = {
                "l_ro": l_ro,
                "z_rl": z_rl,
                "z_hat": z_hat,
                "_z": z,
                "_pad_mask": pad_mask,
            }
            _update_metric(metrics, output)

    summary = {
        key: sum(values) / max(1, len(values))
        for key, values in metrics.items()
    }
    summary["num_batches"] = len(metrics["l_ro"])
    summary["batch_size"] = int(args.batch_size)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
