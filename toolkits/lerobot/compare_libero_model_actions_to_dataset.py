#!/usr/bin/env python3
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

"""Compare offline OpenPI LIBERO action predictions against dataset labels.

This script is offline-only. It loads one LeRobot LIBERO dataset episode,
runs an OpenPI checkpoint on dataset observations, and compares the model's
first predicted action step against the dataset's 7D action label.

Example:

    python toolkits/lerobot/compare_libero_model_actions_to_dataset.py \
      --dataset-path /mnt/public2/xiekaizhi/rlt-openpi-sim/data/libero_10_task6 \
      --model-path /mnt/public2/xiekaizhi/rlt-openpi-sim/pi05_base \
      --config-name pi05_libero \
      --default-prompt "put the white mug on the plate and put the chocolate pudding to the right of the plate" \
      --norm-stats-path /mnt/public2/xiekaizhi/rlt-openpi-sim/data/libero_10_task6/norm_stats.json \
      --max-steps 200 \
      --output-dir /tmp/libero_action_compare
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from rlinf.data.datasets.recap.utils import (
    decode_image_struct_batch,
    load_task_descriptions,
)
from rlinf.models.embodiment.openpi import build_openpi_rlt_backbone

ACTION_DIM_NAMES = [
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--default-prompt", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--num-action-chunks", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="/tmp/libero_action_compare")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def _output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"LeRobot dataset root not found: {root}")
    return root


def _resolve_action_key(meta: LeRobotDatasetMetadata) -> str:
    if "actions" in meta.features:
        return "actions"
    if "action" in meta.features:
        return "action"
    raise KeyError("Dataset must contain either 'actions' or 'action'.")


def _build_dataset(root: Path, num_action_chunks: int) -> tuple[LeRobotDataset, str]:
    meta = LeRobotDatasetMetadata(root.name, root=root)
    action_key = _resolve_action_key(meta)

    required = ["state", "image", "wrist_image"]
    missing = [key for key in required if key not in meta.features]
    if missing:
        raise KeyError(f"Dataset is missing required keys: {missing}")

    delta_timestamps = {
        action_key: [t / meta.fps for t in range(num_action_chunks)],
    }
    dataset = LeRobotDataset(
        root.name,
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    dataset.hf_dataset.set_transform(decode_image_struct_batch)
    return dataset, action_key


def _episode_frame_indices(dataset: LeRobotDataset, episode_index: int) -> list[int]:
    idx = dataset.episode_data_index
    if episode_index < 0 or episode_index >= len(idx["from"]):
        raise IndexError(
            f"episode_index={episode_index} is out of range; "
            f"dataset has {len(idx['from'])} episodes."
        )
    start = int(idx["from"][episode_index].item())
    end = int(idx["to"][episode_index].item())
    return list(range(start, end))


def _as_numpy_image(value: Any) -> np.ndarray:
    if hasattr(value, "convert"):
        value = value.convert("RGB")
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _sample_prompt(
    sample: dict[str, Any],
    task_map: dict[int, str],
    default_prompt: str,
) -> str:
    prompt = sample.get("prompt")
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    task_value = sample.get("task")
    if isinstance(task_value, bytes):
        task_value = task_value.decode("utf-8")
    if isinstance(task_value, str) and task_value.strip():
        return task_value

    task_index = sample.get("task_index")
    if task_index is not None:
        try:
            mapped = task_map.get(int(task_index))
        except Exception:
            mapped = None
        if mapped:
            return mapped

    return default_prompt


def _sample_to_env_obs(
    sample: dict[str, Any],
    *,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    state = np.asarray(sample["state"], dtype=np.float32)
    if state.ndim != 1:
        state = state.reshape(-1)

    main_image = _as_numpy_image(sample["image"])
    wrist_image = _as_numpy_image(sample["wrist_image"])

    return {
        "main_images": torch.from_numpy(main_image[None, ...]).to(device),
        "wrist_images": torch.from_numpy(wrist_image[None, ...]).to(device),
        "extra_view_images": None,
        "states": torch.from_numpy(state[None, ...]).to(device),
        "task_descriptions": [prompt],
    }


def _build_model(args: argparse.Namespace, device: torch.device):
    return build_openpi_rlt_backbone(
        model_path=args.model_path,
        config_name=args.config_name,
        num_images_in_input=args.num_images_in_input,
        num_action_chunks=args.num_action_chunks,
        action_dim=args.action_dim,
        num_steps=args.num_steps,
        device=device,
        freeze=True,
    )


def _predict_first_action(
    model: torch.nn.Module,
    env_obs: dict[str, Any],
) -> np.ndarray:
    with torch.no_grad():
        actions, _ = model.predict_action_batch(
            env_obs,
            mode="eval",
            compute_values=False,
        )
    first = actions[0, 0].detach().to(torch.float32).cpu().numpy()
    return np.asarray(first, dtype=np.float32)


def _gt_first_action(sample: dict[str, Any], action_key: str) -> np.ndarray:
    action = np.asarray(sample[action_key], dtype=np.float32)
    if action.ndim == 1:
        return action
    if action.ndim == 2:
        return action[0]
    raise ValueError(f"Unexpected action shape: {action.shape}")


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "frame_index",
        "episode_index",
        "dataset_index",
        "prompt",
        "gt_action",
        "pred_action",
        "diff_action",
        "l2",
        "cosine",
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ("gt_action", "pred_action", "diff_action"):
                csv_row[key] = json.dumps(csv_row[key])
            writer.writerow(csv_row)


def _plot_action_overlay(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    path: Path,
) -> None:
    num_steps = gt_actions.shape[0]
    fig, axes = plt.subplots(len(ACTION_DIM_NAMES), 1, figsize=(12, 14), sharex=True)
    x = np.arange(num_steps)
    for dim, ax in enumerate(axes):
        ax.plot(x, gt_actions[:, dim], label="gt", linewidth=1.5)
        ax.plot(x, pred_actions[:, dim], label="pred", linewidth=1.2)
        ax.set_ylabel(ACTION_DIM_NAMES[dim])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("sample step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error_hist(diff_actions: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(len(ACTION_DIM_NAMES), 1, figsize=(10, 14), sharex=False)
    for dim, ax in enumerate(axes):
        ax.hist(diff_actions[:, dim], bins=40, alpha=0.85)
        ax.set_ylabel(ACTION_DIM_NAMES[dim])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("pred - gt")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summarize(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    l2_values: np.ndarray,
    cosine_values: np.ndarray,
) -> dict[str, Any]:
    diff = pred_actions - gt_actions
    per_dim_mae = np.mean(np.abs(diff), axis=0)
    per_dim_rmse = np.sqrt(np.mean(np.square(diff), axis=0))
    return {
        "num_samples": int(gt_actions.shape[0]),
        "mean_l2": float(np.mean(l2_values)),
        "median_l2": float(np.median(l2_values)),
        "mean_cosine": float(np.mean(cosine_values)),
        "gt_mean": gt_actions.mean(axis=0).tolist(),
        "pred_mean": pred_actions.mean(axis=0).tolist(),
        "diff_mean": diff.mean(axis=0).tolist(),
        "diff_abs_mean": np.abs(diff).mean(axis=0).tolist(),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_rmse": per_dim_rmse.tolist(),
        "action_dim_names": ACTION_DIM_NAMES,
    }


def main() -> None:
    args = _parse_args()
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")

    root = _dataset_root(args.dataset_path)
    if args.norm_stats_path is not None and not Path(args.norm_stats_path).expanduser().exists():
        raise FileNotFoundError(f"norm_stats_path does not exist: {args.norm_stats_path}")

    device = _device(args.device)
    output_dir = _output_dir(args.output_dir)

    dataset, action_key = _build_dataset(root, args.num_action_chunks)
    task_map = load_task_descriptions(root)
    model = _build_model(args, device)

    frame_indices = _episode_frame_indices(dataset, args.episode_index)
    if args.max_steps is not None and args.max_steps > 0:
        frame_indices = frame_indices[: args.max_steps]
    frame_indices = frame_indices[:: args.stride]
    if not frame_indices:
        raise ValueError("No frames selected after applying episode/max_steps/stride.")

    rows: list[dict[str, Any]] = []
    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    l2_values: list[float] = []
    cosine_values: list[float] = []

    for dataset_index in frame_indices:
        sample = dataset[dataset_index]
        prompt = _sample_prompt(sample, task_map, args.default_prompt)
        env_obs = _sample_to_env_obs(sample, prompt=prompt, device=device)
        pred = _predict_first_action(model, env_obs)
        gt = _gt_first_action(sample, action_key)
        diff = pred - gt

        gt_norm = np.linalg.norm(gt)
        pred_norm = np.linalg.norm(pred)
        cosine = float(np.dot(gt, pred) / max(gt_norm * pred_norm, 1e-12))
        l2 = float(np.linalg.norm(diff))

        gt_actions.append(gt)
        pred_actions.append(pred)
        l2_values.append(l2)
        cosine_values.append(cosine)
        rows.append(
            {
                "frame_index": int(sample.get("frame_index", -1)),
                "episode_index": int(sample.get("episode_index", args.episode_index)),
                "dataset_index": int(dataset_index),
                "prompt": prompt,
                "gt_action": gt.astype(float).tolist(),
                "pred_action": pred.astype(float).tolist(),
                "diff_action": diff.astype(float).tolist(),
                "l2": l2,
                "cosine": cosine,
            }
        )

    gt_arr = np.stack(gt_actions, axis=0)
    pred_arr = np.stack(pred_actions, axis=0)
    diff_arr = pred_arr - gt_arr
    summary = _summarize(
        gt_arr,
        pred_arr,
        np.asarray(l2_values, dtype=np.float64),
        np.asarray(cosine_values, dtype=np.float64),
    )
    summary.update(
        {
            "dataset_path": str(root),
            "model_path": str(Path(args.model_path).expanduser()),
            "config_name": args.config_name,
            "episode_index": int(args.episode_index),
            "default_prompt": args.default_prompt,
        }
    )

    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "rows.csv"
    overlay_path = output_dir / "action_overlay.png"
    hist_path = output_dir / "action_error_hist.png"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_rows_csv(rows, rows_path)
    _plot_action_overlay(gt_arr, pred_arr, overlay_path)
    _plot_error_hist(diff_arr, hist_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote summary to: {summary_path}")
    print(f"Wrote rows to:    {rows_path}")
    print(f"Wrote plot to:    {overlay_path}")
    print(f"Wrote plot to:    {hist_path}")


if __name__ == "__main__":
    main()
