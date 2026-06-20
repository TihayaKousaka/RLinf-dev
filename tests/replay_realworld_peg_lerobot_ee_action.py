#!/usr/bin/env python3
"""Replay current realworld RLT 19D-state/7D-EE-action demos on PegInsertionEnv."""

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


REPO_ROOT = _bootstrap_repo_paths()


STATE_DIM = 19
ACTION_DIM = 7
DEFAULT_EVAL_CONFIG = (
    REPO_ROOT / "examples/embodiment/config/rlt_realworld_ee_pi05_sft_eval.yaml"
)
DEFAULT_ENV_CONFIG = (
    REPO_ROOT
    / "examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml"
)


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


def _row_array(
    row: dict[str, Any],
    names: tuple[str, ...],
    *,
    dim: int | None = None,
    allowed_dims: tuple[int, ...] | None = None,
) -> np.ndarray:
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
    if dim is not None and arr.shape[0] != dim:
        raise RuntimeError(f"Expected {dim}D for {names}, got {arr.shape}")
    if allowed_dims is not None and arr.shape[0] not in allowed_dims:
        raise RuntimeError(f"Expected one of {allowed_dims}D for {names}, got {arr.shape}")
    return arr


def _row_state(row: dict[str, Any]) -> np.ndarray:
    return _row_array(
        row,
        ("state", "observation.state"),
        dim=STATE_DIM,
    )


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


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {path} requires PyYAML. Install it or pass all replay "
            "calibration values explicitly."
        ) from exc
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping YAML at {path}, got {type(data).__name__}")
    return data


def _first_hardware_config(config: dict[str, Any]) -> dict[str, Any]:
    for node_group in config.get("cluster", {}).get("node_groups", []) or []:
        hardware = node_group.get("hardware") or {}
        configs = hardware.get("configs") or []
        if hardware.get("type") == "Franka" and configs:
            return configs[0] or {}
    return {}


def _camera_config(
    hardware_config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    for camera in hardware_config.get("camera_infos", []) or []:
        if camera.get("name") == name:
            return camera
    return {}


def _eval_override_config(eval_config: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    override = dict(env_config.get("override_cfg") or {})
    override.update(eval_config.get("env", {}).get("eval", {}).get("override_cfg") or {})
    return override


def _configured_max_steps(eval_config: dict[str, Any], env_config: dict[str, Any]) -> int:
    eval_cfg = _env_eval_config(eval_config)
    eval_override = eval_cfg.get("override_cfg") or {}
    env_override = env_config.get("override_cfg") or {}
    value = (
        eval_override.get(
            "max_num_steps",
            eval_cfg.get(
                "max_episode_steps",
                env_override.get(
                    "max_num_steps",
                    env_config.get("max_episode_steps", 20),
                ),
            ),
        )
    )
    return int(value)


def _env_eval_config(eval_config: dict[str, Any]) -> dict[str, Any]:
    return eval_config.get("env", {}).get("eval", {}) or {}


def _apply_config_defaults(args: argparse.Namespace) -> None:
    eval_config = _load_yaml(args.eval_config)
    env_config = _load_yaml(args.env_config)
    hardware_config = _first_hardware_config(eval_config)
    override_cfg = _eval_override_config(eval_config, env_config)
    args.robot_ip = args.robot_ip or hardware_config.get("robot_ip")
    args.node_rank = args.node_rank if args.node_rank is not None else hardware_config.get("node_rank")
    args.gripper_type = args.gripper_type or hardware_config.get("gripper_type")
    args.gripper_connection = args.gripper_connection or hardware_config.get("gripper_connection")

    main_camera = _camera_config(hardware_config, args.main_camera_name)
    wrist_camera = _camera_config(hardware_config, args.wrist_camera_name)
    args.main_camera_serial = args.main_camera_serial or main_camera.get("serial_number")
    args.main_camera_type = args.main_camera_type or main_camera.get("camera_type")
    args.wrist_camera_serial = args.wrist_camera_serial or wrist_camera.get("serial_number")
    args.wrist_camera_type = args.wrist_camera_type or wrist_camera.get("camera_type")

    args.joint_reset_qpos = args.joint_reset_qpos or override_cfg.get("joint_reset_qpos")
    args.target_ee_pose = args.target_ee_pose or override_cfg.get("target_ee_pose")
    args.reward_threshold = args.reward_threshold or override_cfg.get("reward_threshold")
    args.task_description = args.task_description or override_cfg.get(
        "task_description",
        env_config.get("default_prompt"),
    )
    if args.max_steps is None:
        args.max_steps = _configured_max_steps(eval_config, env_config)

    args.node_rank = int(args.node_rank if args.node_rank is not None else 0)
    args.gripper_type = args.gripper_type or "robotiq"
    args.gripper_connection = args.gripper_connection or "/dev/ttyUSB0"
    args.main_camera_type = args.main_camera_type or "realsense"
    args.wrist_camera_type = args.wrist_camera_type or "lumos"
    args.reward_threshold = args.reward_threshold or [0.015, 0.015, 0.03, 0.2, 0.2, 0.2]
    args.task_description = args.task_description or "insert the peg in the hole"


def _validate_calibration_args(args: argparse.Namespace) -> None:
    required = {
        "--robot-ip": args.robot_ip,
        "--main-camera-serial": args.main_camera_serial,
        "--wrist-camera-serial": args.wrist_camera_serial,
        "--joint-reset-qpos": args.joint_reset_qpos,
        "--target-ee-pose": args.target_ee_pose,
    }
    missing = [name for name, value in required.items() if value is None or value == ""]
    if missing:
        raise SystemExit(
            f"Missing required args/config values: {missing}. "
            f"Checked eval config {args.eval_config} and env config {args.env_config}."
        )

    if len(args.joint_reset_qpos) != 7:
        raise SystemExit(f"--joint-reset-qpos must be 7D, got {args.joint_reset_qpos}")
    if len(args.target_ee_pose) != 6:
        raise SystemExit(f"--target-ee-pose must be 6D, got {args.target_ee_pose}")
    if len(args.reward_threshold) != 6:
        raise SystemExit(f"--reward-threshold must be 6D, got {args.reward_threshold}")
    if args.start < 0:
        raise SystemExit(f"--start must be non-negative, got {args.start}")
    if args.max_steps <= 0:
        raise SystemExit(f"--max-steps must be positive, got {args.max_steps}")
    if args.stride <= 0:
        raise SystemExit(f"--stride must be positive, got {args.stride}")
    if args.log_every <= 0:
        raise SystemExit(f"--log-every must be positive, got {args.log_every}")


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
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG,
        help="Eval YAML used to fill robot/camera/task calibration defaults.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=DEFAULT_ENV_CONFIG,
        help="Env YAML used as fallback for task calibration defaults.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum replay steps. Defaults to eval override max_num_steps, "
            "then env.eval.max_episode_steps, then env YAML max_num_steps."
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--yes", action="store_true", help="Skip destructive-test confirmation.")

    parser.add_argument("--robot-ip", default=None)
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ["RLINF_NODE_RANK"])
        if "RLINF_NODE_RANK" in os.environ
        else None,
    )
    parser.add_argument("--gripper-type", default=None)
    parser.add_argument("--gripper-connection", default=None)
    parser.add_argument("--main-camera-name", default="main_camera")
    parser.add_argument("--wrist-camera-name", default="wrist_camera")
    parser.add_argument("--main-camera-serial", default=None)
    parser.add_argument("--main-camera-type", default=None)
    parser.add_argument(
        "--wrist-camera-serial",
        default=None,
    )
    parser.add_argument("--wrist-camera-type", default=None)
    parser.add_argument(
        "--joint-reset-qpos",
        type=lambda raw: _parse_float_list(raw, length=7, name="joint_reset_qpos"),
        default=None,
        help=(
            "Comma-separated 7D safe reset qpos. Defaults to eval/env YAML. "
            "This is not read from the dataset state."
        ),
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
        default=None,
    )
    parser.add_argument("--task-description", default=None)
    args = parser.parse_args()
    _apply_config_defaults(args)
    _validate_calibration_args(args)
    return args


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    rows = _load_episode(dataset_path, args.episode_index)
    selected_rows = rows[args.start : args.start + args.max_steps * args.stride : args.stride]
    if not selected_rows:
        raise RuntimeError("No rows selected for replay.")
    first_state = _row_state(selected_rows[0])
    state_dim = first_state.shape[0]
    first_action = _row_action(selected_rows[0])
    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "eval_config": str(args.eval_config),
                "env_config": str(args.env_config),
                "episode_index": args.episode_index,
                "episode_rows": len(rows),
                "start": args.start,
                "requested_max_steps": args.max_steps,
                "num_selected_rows": len(selected_rows),
                "stride": args.stride,
                "state_dim": state_dim,
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
