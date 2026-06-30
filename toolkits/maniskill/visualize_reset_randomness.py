#!/usr/bin/env python3
"""Visualize ManiSkill reset randomness for RLinf wrappers.

This script is meant to answer one question:
do repeated resets actually change the initial state, or are we replaying the
same scene over and over?

It creates a ManiSkill env through RLinf's wrapper path, performs repeated
resets, and saves a compact summary plus optional preview images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_hash(value: Any) -> str:
    arr = _to_numpy(value)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _extract_image(obs: dict[str, Any], key: str) -> np.ndarray | None:
    value = obs.get(key)
    if value is None:
        return None
    arr = _to_numpy(value)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def _save_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def build_env(args: argparse.Namespace) -> ManiskillEnv:
    cfg = OmegaConf.create(
        {
            "seed": args.seed,
            "auto_reset": False,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "use_full_state": False,
            "group_size": args.group_size,
            "use_fixed_reset_state_ids": args.use_fixed_reset_state_ids,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "fps": 10,
                "record_every": 1,
                "video_base_dir": "",
            },
            "reward_mode": "only_success",
            "wrap_obs_mode": "rlt_openpi_joint",
            "rlt_intervention": None,
            "init_params": {
                "id": args.env_id,
                "num_envs": args.num_envs,
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "sim_backend": "gpu",
                "reward_mode": "sparse",
                "sim_config": {"sim_freq": 100, "control_freq": 10},
                "max_episode_steps": args.max_episode_steps,
                "sensor_configs": {"width": args.width, "height": args.height},
                "render_mode": "all",
            },
        }
    )

    env = ManiskillEnv(
        cfg=cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PegInsertionSideWideClearance-v1")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num-resets", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--use-fixed-reset-state-ids",
        action="store_true",
        help="Keep RLinf's reset_state_ids path enabled.",
    )
    parser.add_argument(
        "--legacy-reseed-every-reset",
        action="store_true",
        help="Emulate the old behavior by restoring the same seed before every reset.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)

    summary_path = args.out_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for reset_idx in range(args.num_resets):
            if args.legacy_reseed_every_reset:
                obs, info = env.reset(seed=env.seed)
            else:
                obs, info = env.reset()

            main_img = _extract_image(obs, "main_images")
            wrist_img = _extract_image(obs, "wrist_images")
            states = obs.get("states")
            task_desc = obs.get("task_descriptions")

            row = {
                "reset_idx": reset_idx,
                "seed": int(env.seed),
                "main_hash": _tensor_hash(main_img) if main_img is not None else None,
                "wrist_hash": _tensor_hash(wrist_img) if wrist_img is not None else None,
                "state_hash": _tensor_hash(states) if states is not None else None,
                "prompt": task_desc[0] if isinstance(task_desc, list) else task_desc,
                "episode_info": {
                    k: bool(v.item()) if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                    for k, v in (info.get("episode", {}) if isinstance(info, dict) else {}).items()
                },
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

            if main_img is not None:
                _save_image(args.out_dir / f"reset_{reset_idx:03d}_main.png", main_img)
            if wrist_img is not None:
                _save_image(args.out_dir / f"reset_{reset_idx:03d}_wrist.png", wrist_img)

            print(
                f"reset={reset_idx} main={row['main_hash']} "
                f"wrist={row['wrist_hash']} state={row['state_hash']}"
            )


if __name__ == "__main__":
    main()
