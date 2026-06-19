#!/usr/bin/env python3
"""Replay realworld RLT EE-action LeRobot demos on the original PegInsertionEnv."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_repo_paths() -> Path:
    script_path = Path(__file__).resolve()
    rlinf_root = script_path.parents[1]
    if str(rlinf_root) not in sys.path:
        sys.path.insert(0, str(rlinf_root))
    os.environ.setdefault("EMBODIED_PATH", str(rlinf_root / "examples/embodiment"))
    return rlinf_root


_bootstrap_repo_paths()


STATE_DIM = 34
ACTION_DIM = 7


def _quat_xyzw_to_rpy(quat: np.ndarray) -> list[float]:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0:
        raise ValueError(f"Invalid zero quaternion: {quat}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [float(roll), float(pitch), float(yaw)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int]:
    if hasattr(dataset, "episode_data_index") and dataset.episode_data_index is not None:
        data_index = dataset.episode_data_index
        if episode_index < len(data_index["from"]):
            return (
                int(data_index["from"][episode_index].item()),
                int(data_index["to"][episode_index].item()),
            )

    indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if int(_to_numpy(sample["episode_index"]).reshape(-1)[0]) == episode_index:
            indices.append(idx)
    if not indices:
        raise RuntimeError(f"Episode {episode_index} not found in dataset")
    return min(indices), max(indices) + 1


def _load_episode(dataset_path: Path, episode_index: int) -> list[dict[str, Any]]:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'lerobot'. Run on the same machine/env used for "
            "realworld data collection."
        ) from exc

    dataset = LeRobotDataset(str(dataset_path), download_videos=False)
    start, end = _episode_bounds(dataset, episode_index)
    rows = [dataset[idx] for idx in range(start, end)]
    if not rows:
        raise RuntimeError(f"Episode {episode_index} is empty")
    return rows


def _row_array(row: dict[str, Any], names: tuple[str, ...], *, dim: int) -> np.ndarray:
    for name in names:
        if name in row:
            value = row[name]
            break
    else:
        if isinstance(row.get("observation"), dict):
            for name in names:
                if name in row["observation"]:
                    value = row["observation"][name]
                    break
            else:
                raise KeyError(f"Dataset row has none of {names}. Keys: {list(row)}")
        else:
            raise KeyError(f"Dataset row has none of {names}. Keys: {list(row)}")

    arr = _to_numpy(value).astype(np.float64).reshape(-1)
    if arr.shape[0] != dim:
        raise RuntimeError(f"Expected {dim}D for {names}, got {arr.shape}")
    return arr


def _row_state(row: dict[str, Any]) -> np.ndarray:
    return _row_array(row, ("state", "observation.state"), dim=STATE_DIM)


def _row_action(row: dict[str, Any]) -> np.ndarray:
    return _row_array(row, ("actions", "action"), dim=ACTION_DIM)


def _obs_tcp_pose(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["state"]["tcp_pose"], dtype=np.float64).reshape(-1)


def _obs_gripper(obs: dict[str, Any]) -> float:
    state = obs["state"]
    if "gripper" in state:
        return float(np.asarray(state["gripper"]).reshape(-1)[0])
    if "gripper_position" in state:
        return float(np.asarray(state["gripper_position"]).reshape(-1)[0])
    return 0.0


def _make_env(args: argparse.Namespace):
    from rlinf.envs.realworld.franka.tasks.peg_insertion_env import PegInsertionEnv
    from rlinf.scheduler.hardware.robots.franka import FrankaConfig, FrankaHWInfo

    camera_infos = [
        {
            "name": "main_camera",
            "serial_number": args.main_camera_serial,
            "camera_type": args.main_camera_type,
        },
        {
            "name": "wrist_camera",
            "serial_number": args.wrist_camera_serial,
            "camera_type": args.wrist_camera_type,
        },
    ]
    hardware_info = FrankaHWInfo(
        type="Franka",
        model="Franka",
        config=FrankaConfig(
            node_rank=args.node_rank,
            robot_ip=args.robot_ip,
            camera_infos=camera_infos,
            gripper_type=args.gripper_type,
            gripper_connection=args.gripper_connection,
            disable_validate=True,
        ),
    )
    override_cfg = {
        "is_dummy": False,
        "task_description": args.task_description,
        "target_ee_pose": args.target_ee_pose,
        "joint_reset_qpos": args.joint_reset_qpos,
        "enable_gripper_penalty": False,
        "reward_threshold": args.reward_threshold,
        "max_num_steps": args.max_steps,
        "enable_camera_player": False,
    }
    return PegInsertionEnv(
        override_cfg=override_cfg,
        worker_info=None,
        hardware_info=hardware_info,
        env_idx=0,
    )


def _parse_float_list(raw: str, *, length: int, name: str) -> list[float]:
    values = [float(x) for x in raw.split(",") if x.strip()]
    if len(values) != length:
        raise argparse.ArgumentTypeError(
            f"{name} must have {length} comma-separated floats"
        )
    return values


def _patch_video_player_stop(env: Any) -> None:
    player = getattr(env, "camera_player", None)
    if player is None or hasattr(player, "stop"):
        return

    def _stop() -> None:
        if getattr(player, "is_running", False):
            player.queue.put(None)

    player.stop = _stop


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay backfilled realworld EE actions on PegInsertionEnv."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--yes", action="store_true", help="Skip destructive-test confirmation.")

    parser.add_argument("--robot-ip", default=os.environ.get("RLT_REALWORLD_ROBOT_IP"))
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("RLINF_NODE_RANK", "0")))
    parser.add_argument("--gripper-type", default=os.environ.get("RLT_REALWORLD_GRIPPER_TYPE", "robotiq"))
    parser.add_argument("--gripper-connection", default=os.environ.get("RLT_REALWORLD_GRIPPER_CONNECTION", "/dev/ttyUSB0"))
    parser.add_argument("--main-camera-serial", default=os.environ.get("RLT_REALWORLD_MAIN_CAMERA_SERIAL"))
    parser.add_argument("--main-camera-type", default=os.environ.get("RLT_REALWORLD_MAIN_CAMERA_TYPE", "realsense"))
    parser.add_argument("--wrist-camera-serial", default=os.environ.get("RLT_REALWORLD_WRIST_CAMERA_SERIAL"))
    parser.add_argument("--wrist-camera-type", default=os.environ.get("RLT_REALWORLD_WRIST_CAMERA_TYPE", "lumos"))
    parser.add_argument(
        "--joint-reset-qpos",
        type=lambda raw: _parse_float_list(raw, length=7, name="joint_reset_qpos"),
        default=None,
        help="Comma-separated 7D safe reset qpos. Defaults to first row state[1:8].",
    )
    parser.add_argument(
        "--target-ee-pose",
        type=lambda raw: _parse_float_list(raw, length=6, name="target_ee_pose"),
        default=None,
        help="Comma-separated 6D target ee pose used by reward/reset.",
    )
    parser.add_argument(
        "--reward-threshold",
        type=lambda raw: _parse_float_list(raw, length=6, name="reward_threshold"),
        default=[0.015, 0.015, 0.03, 0.2, 0.2, 0.2],
    )
    parser.add_argument("--task-description", default="insert the peg in the hole")
    args = parser.parse_args()

    required = {
        "--robot-ip": args.robot_ip,
        "--main-camera-serial": args.main_camera_serial,
        "--wrist-camera-serial": args.wrist_camera_serial,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required args/env vars: {missing}")
    return args


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    rows = _load_episode(dataset_path, args.episode_index)
    selected_rows = rows[args.start : args.start + args.max_steps * args.stride : args.stride]
    if not selected_rows:
        raise RuntimeError("No rows selected for replay.")
    first_state = _row_state(selected_rows[0])

    if args.joint_reset_qpos is None:
        args.joint_reset_qpos = first_state[1:8].astype(float).tolist()
    if args.target_ee_pose is None:
        args.target_ee_pose = (
            first_state[18:21].astype(float).tolist()
            + _quat_xyzw_to_rpy(first_state[21:25])
        )

    first_action = _row_action(selected_rows[0])
    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "episode_index": args.episode_index,
                "start": args.start,
                "num_selected_rows": len(selected_rows),
                "stride": args.stride,
                "mode": "actions[0:7] -> PegInsertionEnv EE delta action",
                "joint_reset_qpos": np.round(args.joint_reset_qpos, 6).tolist(),
                "target_ee_pose": np.round(args.target_ee_pose, 6).tolist(),
            },
            indent=2,
        )
    )
    print(f"first ee_action={np.round(first_action, 4).tolist()}")

    if not args.yes:
        print("\nDESTRUCTIVE TEST: this will command the real Franka arm.")
        input("Press ENTER to continue, or Ctrl+C to abort...")

    env = _make_env(args)
    _patch_video_player_stop(env)
    try:
        obs, _ = env.reset()
        current_tcp = _obs_tcp_pose(obs)
        print(
            "after reset "
            f"tcp_xyz={np.round(current_tcp[:3], 4).tolist()} "
            f"gripper={_obs_gripper(obs):.4f}"
        )

        for step_idx, row in enumerate(selected_rows, start=1):
            action = _row_action(row).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_tcp = _obs_tcp_pose(obs)
            if step_idx == 1 or step_idx % args.log_every == 0:
                print(
                    f"step={step_idx} reward={reward:.4f} "
                    f"term={terminated} trunc={truncated} "
                    f"ee_action={np.round(action, 4).tolist()} "
                    f"tcp_xyz={np.round(current_tcp[:3], 4).tolist()} "
                    f"gripper={_obs_gripper(obs):.4f}"
                )
            if terminated or truncated:
                print("stopped: env terminated/truncated")
                break
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
