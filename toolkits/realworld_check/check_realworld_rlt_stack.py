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

"""Unified preflight checks for real-world Franka EE-action RLT.

The checks are intentionally read-only with respect to robot motion: Franka is
queried for controller readiness/state, cameras are opened for frame capture,
GELLO is sampled, and the keyboard event device is inspected.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}$")
_PLACEHOLDER_VALUES = {
    "",
    "ROBOT_IP",
    "MAIN_CAMERA_SERIAL",
    "WRIST_CAMERA_SERIAL",
    "TARGET_EE_POSE",
    "TARGET_POS",
    "RESET_JOINT_QPOS",
    "CRITICAL_PHASE_RESET_JOINT_QPOS",
    "FULL_TASK_RESET_JOINT_QPOS",
}
_PLACEHOLDER_TOKENS = sorted(_PLACEHOLDER_VALUES - {""})
_DEFAULT_ROBOT_IP = "172.16.0.2"
_DEFAULT_GRIPPER_CONNECTION = "/dev/ttyUSB0"
_DEFAULT_MAIN_CAMERA_SERIAL = "141722070657"
_DEFAULT_MAIN_CAMERA_TYPE = "realsense"
_DEFAULT_WRIST_CAMERA_SERIAL = (
    "usb-XVisio_Technology_XVisio_vSLAM_250801DR48FB26001216-video-index0"
)
_DEFAULT_WRIST_CAMERA_TYPE = "lumos"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: list[str]


@dataclass
class LoadedConfig:
    path: Path
    data: Any | None
    loader: str | None
    error: str | None = None


@dataclass
class CameraSpec:
    name: str
    serial_number: str | None
    camera_type: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _repo_root() / "examples/embodiment/config/rlt_stage2_realworld_ee.yaml"


def _default_env_config_path() -> Path:
    return (
        _repo_root()
        / "examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight-check the real-world RLT Franka EE-action stack."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Stage2 YAML config path.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=_default_env_config_path(),
        help="Real-world task env YAML config path.",
    )
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("RLT_REALWORLD_ROBOT_IP")
        or os.environ.get("FRANKA_ROBOT_IP")
        or _DEFAULT_ROBOT_IP,
        help="Franka IP.",
    )
    parser.add_argument(
        "--end-effector-type",
        default="robotiq_gripper",
        choices=["franka_gripper", "robotiq_gripper", "ruiyan_hand"],
        help="Mounted Franka end-effector type.",
    )
    parser.add_argument(
        "--gripper-connection",
        default=os.environ.get("RLT_REALWORLD_GRIPPER_CONNECTION")
        or _DEFAULT_GRIPPER_CONNECTION,
        help="Serial port for Robotiq gripper.",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="Serial port for Ruiyan hand when --end-effector-type=ruiyan_hand.",
    )
    parser.add_argument(
        "--hand-baudrate",
        type=int,
        default=460800,
        help="Ruiyan hand serial baudrate.",
    )
    parser.add_argument(
        "--hand-motor-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Ruiyan hand motor IDs.",
    )
    parser.add_argument(
        "--franka-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for Franka controller readiness.",
    )
    parser.add_argument(
        "--main-camera-serial",
        default=os.environ.get("RLT_REALWORLD_MAIN_CAMERA_SERIAL")
        or _DEFAULT_MAIN_CAMERA_SERIAL,
        help="Override main camera serial/device id.",
    )
    parser.add_argument(
        "--main-camera-type",
        default=os.environ.get("RLT_REALWORLD_MAIN_CAMERA_TYPE")
        or _DEFAULT_MAIN_CAMERA_TYPE,
        help="Override main camera type.",
    )
    parser.add_argument(
        "--wrist-camera-serial",
        default=os.environ.get("RLT_REALWORLD_WRIST_CAMERA_SERIAL")
        or _DEFAULT_WRIST_CAMERA_SERIAL,
        help="Override wrist camera serial/device id.",
    )
    parser.add_argument(
        "--wrist-camera-type",
        default=os.environ.get("RLT_REALWORLD_WRIST_CAMERA_TYPE")
        or _DEFAULT_WRIST_CAMERA_TYPE,
        help="Override wrist camera type.",
    )
    parser.add_argument(
        "--camera-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
        help="Camera capture resolution used by the preflight check.",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=15,
        help="Requested camera FPS used by the preflight check.",
    )
    parser.add_argument(
        "--camera-frames",
        type=int,
        default=20,
        help="Number of frames to read from each camera.",
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for each camera frame.",
    )
    parser.add_argument(
        "--gello-port",
        default=os.environ.get("RLT_REALWORLD_GELLO_PORT"),
        help="GELLO serial port.",
    )
    parser.add_argument(
        "--gello-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the first GELLO reading.",
    )
    parser.add_argument(
        "--keyboard-device",
        default=os.environ.get("RLINF_KEYBOARD_DEVICE"),
        help="Keyboard /dev/input/eventX path.",
    )
    parser.add_argument(
        "--skip-franka",
        action="store_true",
        help="Skip Franka controller readiness/state check.",
    )
    parser.add_argument(
        "--skip-cameras",
        action="store_true",
        help="Skip main/wrist camera capture checks.",
    )
    parser.add_argument(
        "--skip-gello",
        action="store_true",
        help="Skip GELLO serial/readiness check.",
    )
    parser.add_argument(
        "--skip-keyboard",
        action="store_true",
        help="Skip keyboard event-device check.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Return exit code 0 even if a check fails.",
    )
    return parser.parse_args()


def _result(
    name: str,
    status: str,
    message: str,
    details: list[str] | None = None,
) -> CheckResult:
    return CheckResult(name=name, status=status, message=message, details=details or [])


def _load_config(path: Path) -> LoadedConfig:
    if not path.exists():
        return LoadedConfig(path=path, data=None, loader=None, error="file not found")

    try:
        from omegaconf import OmegaConf

        return LoadedConfig(path=path, data=OmegaConf.load(path), loader="omegaconf")
    except ModuleNotFoundError as exc:
        omega_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - depends on local YAML content.
        return LoadedConfig(
            path=path,
            data=None,
            loader=None,
            error=f"OmegaConf failed: {type(exc).__name__}: {exc}",
        )

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as file:
            return LoadedConfig(path=path, data=yaml.safe_load(file), loader="yaml")
    except ModuleNotFoundError as exc:
        return LoadedConfig(
            path=path,
            data=None,
            loader=None,
            error=f"{omega_error}; PyYAML unavailable: {type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - depends on local YAML content.
        return LoadedConfig(
            path=path,
            data=None,
            loader=None,
            error=f"PyYAML failed: {type(exc).__name__}: {exc}",
        )


def _select(config: LoadedConfig | None, key: str, default: Any = None) -> Any:
    if config is None or config.data is None:
        return default
    if config.loader == "omegaconf":
        try:
            from omegaconf import OmegaConf

            return _resolve_value(OmegaConf.select(config.data, key, default=default))
        except Exception:
            return default

    current = config.data
    try:
        for part in key.split("."):
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                return default
        return _resolve_value(current)
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def _resolve_value(value: Any) -> Any:
    if isinstance(value, str):
        match = _OC_ENV_PATTERN.match(value)
        if match:
            env_name = match.group(1)
            fallback = match.group(2)
            return os.environ.get(env_name, fallback)
    return value


def _is_placeholder(value: Any) -> bool:
    value = _resolve_value(value)
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return (
            stripped in _PLACEHOLDER_VALUES
            or (stripped.startswith("<") and stripped.endswith(">"))
        )
    return False


def _format_value(value: Any) -> str:
    value = _resolve_value(value)
    if value is None:
        return "<missing>"
    return str(value)


def _flag_enabled(value: Any) -> bool:
    value = _resolve_value(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _text_placeholder_details(path: Path) -> list[str]:
    if not path.exists():
        return [f"{path}: file not found"]
    text = path.read_text(encoding="utf-8")
    return [f"{path}: contains {token}" for token in _PLACEHOLDER_TOKENS if token in text]


def _camera_specs(args: argparse.Namespace, main_config: LoadedConfig) -> list[CameraSpec]:
    main_serial = args.main_camera_serial or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.camera_infos.0.serial_number",
    )
    main_type = args.main_camera_type or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.camera_infos.0.camera_type",
        "realsense",
    )
    wrist_serial = args.wrist_camera_serial or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.camera_infos.1.serial_number",
    )
    wrist_type = args.wrist_camera_type or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.camera_infos.1.camera_type",
        "lumos",
    )
    return [
        CameraSpec("main_camera", main_serial, str(main_type or "realsense")),
        CameraSpec("wrist_camera", wrist_serial, str(wrist_type or "lumos")),
    ]


def _hardware_gripper_connection(
    args: argparse.Namespace,
    main_config: LoadedConfig,
) -> Any:
    return args.gripper_connection or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.gripper_connection",
    )


def _check_config(
    args: argparse.Namespace,
    main_config: LoadedConfig,
    env_config: LoadedConfig,
) -> CheckResult:
    details: list[str] = []
    failures: list[str] = []
    warnings: list[str] = []

    for loaded in (main_config, env_config):
        if loaded.error:
            warnings.append(f"{loaded.path}: YAML parser unavailable ({loaded.error})")
        else:
            details.append(f"{loaded.path}: parsed with {loaded.loader}")

    for placeholder in _text_placeholder_details(env_config.path):
        failures.append(placeholder)

    for key in (
        "override_cfg.target_ee_pose",
        "override_cfg.joint_reset_qpos",
    ):
        value = _select(env_config, key)
        if env_config.data is not None and _is_placeholder(value):
            failures.append(f"{env_config.path}: {key} is {_format_value(value)}")

    robot_ip = args.robot_ip or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.robot_ip",
    )
    if _is_placeholder(robot_ip):
        failures.append("Franka IP is missing or still a placeholder")
    else:
        details.append(f"Franka IP: {_format_value(robot_ip)}")

    gripper_connection = _hardware_gripper_connection(args, main_config)
    if args.end_effector_type == "robotiq_gripper":
        if _is_placeholder(gripper_connection):
            failures.append("Robotiq gripper connection is missing or placeholder")
        else:
            details.append(f"Robotiq gripper connection: {gripper_connection}")

    use_gello = _flag_enabled(_select(main_config, "env.train.use_gello", False))
    gello_port = args.gello_port or _select(main_config, "env.train.gello_port")
    if use_gello and _is_placeholder(gello_port):
        failures.append("GELLO is enabled but gello_port is missing or placeholder")
    elif use_gello and str(gello_port) == str(gripper_connection):
        failures.append(
            f"GELLO port {gello_port} conflicts with Robotiq gripper connection"
        )
    elif use_gello:
        details.append(f"GELLO port: {_format_value(gello_port)}")
    else:
        details.append("GELLO intervention: disabled")

    for spec in _camera_specs(args, main_config):
        if _is_placeholder(spec.serial_number):
            failures.append(f"{spec.name} serial/device id is missing or placeholder")
        else:
            details.append(
                f"{spec.name}: type={spec.camera_type} serial={spec.serial_number}"
            )

    keyboard_device = args.keyboard_device
    if keyboard_device:
        details.append(f"Keyboard device: {keyboard_device}")
    else:
        warnings.append(
            "RLINF_KEYBOARD_DEVICE is not set; keyboard check will scan /dev/input"
        )

    gello_mode = _select(main_config, "env.train.gello_action_mode")
    if gello_mode is not None and str(gello_mode) != "ee_delta":
        warnings.append(f"env.train.gello_action_mode is {gello_mode!r}, not ee_delta")

    keyboard_wrapper = _select(main_config, "env.train.keyboard_reward_wrapper")
    if keyboard_wrapper is not None and str(keyboard_wrapper) != "single_stage":
        warnings.append(
            "env.train.keyboard_reward_wrapper is "
            f"{keyboard_wrapper!r}, not single_stage"
        )

    task_mode = _select(env_config, "task_mode")
    if task_mode == "full_task":
        details.append("Keyboard critical-phase key is required for full_task mode")
    elif keyboard_wrapper is None:
        details.append("Keyboard input: not required by current env config")

    if failures:
        return _result("config", "FAIL", "configuration has blocking issues", failures)
    if warnings:
        return _result("config", "WARN", "configuration parsed with warnings", warnings)
    return _result("config", "PASS", "configuration values look ready", details)


def _check_franka(args: argparse.Namespace, main_config: LoadedConfig) -> CheckResult:
    if args.skip_franka:
        return _result("franka", "SKIP", "skipped by --skip-franka")

    robot_ip = args.robot_ip or _select(
        main_config,
        "cluster.node_groups.1.hardware.configs.0.robot_ip",
    )
    if _is_placeholder(robot_ip):
        return _result("franka", "FAIL", "Franka IP is missing or placeholder")
    gripper_connection = _hardware_gripper_connection(args, main_config)
    if args.end_effector_type == "ruiyan_hand" and args.hand_port is None:
        return _result(
            "franka",
            "FAIL",
            "--hand-port is required for --end-effector-type=ruiyan_hand",
        )
    if args.end_effector_type == "robotiq_gripper" and _is_placeholder(
        gripper_connection
    ):
        return _result(
            "franka",
            "FAIL",
            "--gripper-connection is required for --end-effector-type=robotiq_gripper",
        )

    try:
        import numpy as np

        from rlinf.envs.realworld.franka.franka_controller import FrankaController
    except Exception as exc:
        return _result(
            "franka",
            "FAIL",
            "failed to import Franka controller dependencies",
            [f"{type(exc).__name__}: {exc}"],
        )

    end_effector_config: dict[str, Any] = {}
    if args.end_effector_type == "ruiyan_hand":
        end_effector_config = {
            "port": args.hand_port,
            "baudrate": args.hand_baudrate,
            "motor_ids": tuple(args.hand_motor_ids),
        }

    controller = None
    try:
        controller = FrankaController.launch_controller(
            robot_ip=str(robot_ip),
            end_effector_type=args.end_effector_type,
            end_effector_config=end_effector_config,
            gripper_connection=str(gripper_connection)
            if gripper_connection is not None
            else None,
        )
        start = time.time()
        robot_up = False
        while time.time() - start < args.franka_timeout:
            robot_up = bool(controller.is_robot_up().wait()[0])
            if robot_up:
                break
            time.sleep(0.5)
        if not robot_up:
            return _result(
                "franka",
                "FAIL",
                "Franka controller did not become ready",
                [f"waited {args.franka_timeout:.1f}s for robot_ip={robot_ip}"],
            )

        state = controller.get_state().wait()[0]
        joint_pos = np.asarray(getattr(state, "arm_joint_position", []), dtype=float)
        joint_vel = np.asarray(getattr(state, "arm_joint_velocity", []), dtype=float)
        tcp_pose = np.asarray(getattr(state, "tcp_pose", []), dtype=float)
        failures = []
        if joint_pos.shape != (7,) or not np.isfinite(joint_pos).all():
            failures.append(f"arm_joint_position invalid: shape={joint_pos.shape}")
        if joint_vel.shape != (7,) or not np.isfinite(joint_vel).all():
            failures.append(f"arm_joint_velocity invalid: shape={joint_vel.shape}")
        if tcp_pose.shape != (7,) or not np.isfinite(tcp_pose).all():
            failures.append(f"tcp_pose invalid: shape={tcp_pose.shape}")
        if failures:
            return _result("franka", "FAIL", "Franka state is invalid", failures)

        details = [
            f"robot_ip={robot_ip}",
            f"joint_pos={np.array2string(joint_pos, precision=3, suppress_small=True)}",
            f"tcp_pose={np.array2string(tcp_pose, precision=3, suppress_small=True)}",
        ]
        return _result("franka", "PASS", "controller ready and state readable", details)
    except Exception as exc:
        return _result(
            "franka",
            "FAIL",
            "Franka controller check failed",
            [f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if controller is not None:
            try:
                controller.stop_impedance().wait()
            except Exception:
                pass


def _available_camera_serials(camera_type: str) -> list[str]:
    camera_type = camera_type.lower()
    if camera_type in ("realsense", "rs"):
        from rlinf.envs.realworld.common.camera.realsense_camera import RealSenseCamera

        return sorted(RealSenseCamera.get_device_serial_numbers())
    if camera_type == "zed":
        from rlinf.envs.realworld.common.camera.zed_camera import ZEDCamera

        return sorted(ZEDCamera.get_device_serial_numbers())
    if camera_type == "lumos":
        from rlinf.envs.realworld.common.camera.lumos_camera import LumosCamera

        return sorted(LumosCamera.get_device_serial_numbers())
    return []


def _check_one_camera(
    spec: CameraSpec,
    *,
    resolution: tuple[int, int],
    fps: int,
    frame_count: int,
    timeout: float,
) -> CheckResult:
    if _is_placeholder(spec.serial_number):
        return _result(
            f"camera:{spec.name}",
            "FAIL",
            "camera serial/device id is missing or placeholder",
        )

    try:
        import numpy as np

        from rlinf.envs.realworld.common.camera import CameraInfo, create_camera
    except Exception as exc:
        return _result(
            f"camera:{spec.name}",
            "FAIL",
            "failed to import camera dependencies",
            [f"{type(exc).__name__}: {exc}"],
        )

    try:
        available = _available_camera_serials(spec.camera_type)
    except Exception as exc:
        available = []
        available_error = f"{type(exc).__name__}: {exc}"
    else:
        available_error = None

    details = [
        f"type={spec.camera_type}",
        f"serial={spec.serial_number}",
        f"requested_resolution={resolution[0]}x{resolution[1]}",
        f"requested_fps={fps}",
    ]
    if available:
        details.append(f"available={available}")
    elif available_error:
        details.append(f"device listing unavailable: {available_error}")

    camera = None
    try:
        camera = create_camera(
            CameraInfo(
                name=spec.name,
                serial_number=str(spec.serial_number),
                camera_type=spec.camera_type,
                resolution=resolution,
                fps=fps,
            )
        )
        camera.open()
        frames = []
        start = time.time()
        for _ in range(frame_count):
            frames.append(camera.get_frame(timeout=timeout))
        elapsed = max(time.time() - start, 1e-6)
        last_frame = np.asarray(frames[-1])
        if last_frame.ndim < 2 or last_frame.size == 0:
            return _result(
                f"camera:{spec.name}",
                "FAIL",
                "captured frame is empty or malformed",
                details + [f"last_frame_shape={last_frame.shape}"],
            )
        measured_fps = frame_count / elapsed
        details.extend(
            [
                f"frames={frame_count}",
                f"last_frame_shape={last_frame.shape}",
                f"measured_fps={measured_fps:.2f}",
            ]
        )
        return _result(
            f"camera:{spec.name}",
            "PASS",
            "camera opened and frames were captured",
            details,
        )
    except Exception as exc:
        return _result(
            f"camera:{spec.name}",
            "FAIL",
            "camera capture check failed",
            details + [f"{type(exc).__name__}: {exc}"],
        )
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass


def _check_cameras(
    args: argparse.Namespace,
    main_config: LoadedConfig,
) -> list[CheckResult]:
    if args.skip_cameras:
        return [_result("cameras", "SKIP", "skipped by --skip-cameras")]

    resolution = (int(args.camera_resolution[0]), int(args.camera_resolution[1]))
    return [
        _check_one_camera(
            spec,
            resolution=resolution,
            fps=int(args.camera_fps),
            frame_count=max(int(args.camera_frames), 1),
            timeout=float(args.camera_timeout),
        )
        for spec in _camera_specs(args, main_config)
    ]


def _check_gello(args: argparse.Namespace, main_config: LoadedConfig) -> CheckResult:
    if args.skip_gello:
        return _result("gello", "SKIP", "skipped by --skip-gello")

    use_gello = _flag_enabled(_select(main_config, "env.train.use_gello", False))
    if not use_gello:
        return _result("gello", "SKIP", "env.train.use_gello is false")

    port = args.gello_port or _select(main_config, "env.train.gello_port")
    if _is_placeholder(port):
        return _result("gello", "FAIL", "GELLO port is missing or placeholder")
    port = str(port)
    gripper_connection = _hardware_gripper_connection(args, main_config)
    if gripper_connection is not None and port == str(gripper_connection):
        return _result(
            "gello",
            "FAIL",
            f"GELLO port conflicts with Robotiq gripper connection: {port}",
        )
    if not os.path.exists(port):
        return _result("gello", "FAIL", f"GELLO port does not exist: {port}")

    try:
        import numpy as np

        from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert
    except Exception as exc:
        return _result(
            "gello",
            "FAIL",
            "failed to import GELLO dependencies",
            [f"{type(exc).__name__}: {exc}"],
        )

    try:
        gello = GelloExpert(port=port)
        start = time.time()
        while time.time() - start < args.gello_timeout:
            if gello.ready:
                break
            time.sleep(0.05)
        if not gello.ready:
            return _result(
                "gello",
                "FAIL",
                "GELLO did not produce a reading",
                [f"port={port}", f"timeout={args.gello_timeout:.1f}s"],
            )
        target_pos, target_quat, gripper = gello.get_action()
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)
        target_quat = np.asarray(target_quat, dtype=float).reshape(-1)
        gripper = np.asarray(gripper, dtype=float).reshape(-1)
        failures = []
        if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
            failures.append(f"target_pos invalid: shape={target_pos.shape}")
        if target_quat.shape != (4,) or not np.isfinite(target_quat).all():
            failures.append(f"target_quat invalid: shape={target_quat.shape}")
        if gripper.size != 1 or not np.isfinite(gripper).all():
            failures.append(f"gripper action invalid: shape={gripper.shape}")
        if not getattr(gello, "thread", None) or not gello.thread.is_alive():
            failures.append("GELLO background read thread is not alive")
        if failures:
            return _result("gello", "FAIL", "GELLO reading is invalid", failures)
        details = [
            f"port={port}",
            f"target_pos={np.array2string(target_pos, precision=3, suppress_small=True)}",
            f"target_quat={np.array2string(target_quat, precision=3, suppress_small=True)}",
            f"gripper={np.array2string(gripper, precision=3, suppress_small=True)}",
        ]
        return _result("gello", "PASS", "GELLO is readable", details)
    except Exception as exc:
        return _result(
            "gello",
            "FAIL",
            "GELLO check failed",
            [f"{type(exc).__name__}: {exc}"],
        )


def _keyboard_device_supports(device: Any, ecodes: Any, key_names: tuple[str, ...]) -> bool:
    capabilities = device.capabilities(verbose=False)
    supported_key_codes = set(capabilities.get(ecodes.EV_KEY, []))
    return all(getattr(ecodes, key_name) in supported_key_codes for key_name in key_names)


def _keyboard_required(main_config: LoadedConfig, env_config: LoadedConfig) -> bool:
    return bool(
        _select(main_config, "env.train.keyboard_reward_wrapper") is not None
        or _select(env_config, "keyboard_reward_wrapper") is not None
        or _select(env_config, "task_mode") == "full_task"
    )


def _check_keyboard(
    args: argparse.Namespace,
    main_config: LoadedConfig,
    env_config: LoadedConfig,
) -> CheckResult:
    if args.skip_keyboard:
        return _result("keyboard", "SKIP", "skipped by --skip-keyboard")
    if not _keyboard_required(main_config, env_config):
        return _result("keyboard", "SKIP", "keyboard is not required by env config")

    try:
        from evdev import InputDevice, ecodes, list_devices
    except Exception as exc:
        return _result(
            "keyboard",
            "FAIL",
            "failed to import evdev",
            [f"{type(exc).__name__}: {exc}"],
        )

    required = ("KEY_A", "KEY_B", "KEY_C", "KEY_V")
    candidates = [args.keyboard_device] if args.keyboard_device else sorted(list_devices())
    permission_denied: list[str] = []
    inspected: list[str] = []

    for path in candidates:
        if not path:
            continue
        try:
            device = InputDevice(path)
        except PermissionError:
            permission_denied.append(path)
            continue
        except FileNotFoundError:
            inspected.append(f"{path}: not found")
            continue
        except OSError as exc:
            inspected.append(f"{path}: {exc}")
            continue

        try:
            name = getattr(device, "name", "")
            inspected.append(f"{path}: {name}")
            if _keyboard_device_supports(device, ecodes, required):
                return _result(
                    "keyboard",
                    "PASS",
                    "keyboard event device is readable and supports a/b/c/v",
                    [f"device={path}", f"name={name}"],
                )
        finally:
            device.close()

    details = inspected
    if permission_denied:
        details.append(f"permission denied: {permission_denied}")
    return _result(
        "keyboard",
        "FAIL",
        "no readable keyboard device supports a/b/c/v",
        details,
    )


def _print_results(results: list[CheckResult]) -> None:
    width = max(len(result.name) for result in results) if results else 0
    for result in results:
        print(f"[{result.status:<4}] {result.name:<{width}}  {result.message}")
        for detail in result.details:
            print(f"       - {detail}")

    counts = {status: 0 for status in ("PASS", "WARN", "FAIL", "SKIP")}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(
        "\nSummary: "
        f"PASS={counts['PASS']} WARN={counts['WARN']} "
        f"FAIL={counts['FAIL']} SKIP={counts['SKIP']}"
    )


def main() -> int:
    args = _parse_args()
    main_config = _load_config(args.config)
    env_config = _load_config(args.env_config)

    results = [
        _check_config(args, main_config, env_config),
        _check_franka(args, main_config),
        *_check_cameras(args, main_config),
        _check_gello(args, main_config),
        _check_keyboard(args, main_config, env_config),
    ]
    _print_results(results)

    if any(result.status == "FAIL" for result in results) and not args.continue_on_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
