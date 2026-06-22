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

"""Minimal preflight checks for realworld RLT env wiring.

This script intentionally checks only the local RLT-specific additions:

1. ``task_mode`` must be ``full_task``.
2. ``critical_phase_key`` must be ``v``.
3. ``main_image_key`` / ``wrist_image_key`` must match the realworld RLT env.
4. A readable keyboard event device must support ``KEY_V``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_env_config_path() -> Path:
    return (
        _repo_root()
        / "examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check only the RLT-specific realworld env wiring."
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=_default_env_config_path(),
        help="Real-world RLT env YAML config path.",
    )
    parser.add_argument(
        "--keyboard-device",
        default=os.environ.get("RLINF_KEYBOARD_DEVICE"),
        help="Keyboard /dev/input/eventX path. If omitted, scan /dev/input/event*.",
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
    except Exception as exc:
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
    except Exception as exc:
        return LoadedConfig(
            path=path,
            data=None,
            loader=None,
            error=f"PyYAML failed: {type(exc).__name__}: {exc}",
        )


def _select(config: LoadedConfig, key: str, default: Any = None) -> Any:
    if config.data is None:
        return default

    if config.loader == "omegaconf":
        try:
            from omegaconf import OmegaConf

            return OmegaConf.select(config.data, key, default=default)
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
        return current
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def _check_env_config(config: LoadedConfig) -> CheckResult:
    if config.error:
        return _result(
            "env-config",
            "FAIL",
            "failed to load env config",
            [f"{config.path}: {config.error}"],
        )

    failures: list[str] = []
    details = [f"{config.path}: parsed with {config.loader}"]

    task_mode = _select(config, "task_mode")
    critical_phase_key = _select(config, "critical_phase_key")
    main_image_key = _select(config, "main_image_key")
    wrist_image_key = _select(config, "wrist_image_key")

    if task_mode != "full_task":
        failures.append(f"task_mode must be 'full_task', got {task_mode!r}")
    else:
        details.append("task_mode=full_task")

    if str(critical_phase_key).lower() != "v":
        failures.append(f"critical_phase_key must be 'v', got {critical_phase_key!r}")
    else:
        details.append("critical_phase_key=v")

    if main_image_key != "main_camera":
        failures.append(
            f"main_image_key must be 'main_camera', got {main_image_key!r}"
        )
    else:
        details.append("main_image_key=main_camera")

    if wrist_image_key != "wrist_camera":
        failures.append(
            f"wrist_image_key must be 'wrist_camera', got {wrist_image_key!r}"
        )
    else:
        details.append("wrist_image_key=wrist_camera")

    if failures:
        return _result("env-config", "FAIL", "RLT env wiring does not match expectation", failures)
    return _result("env-config", "PASS", "RLT env wiring matches expectation", details)


def _keyboard_device_supports(device: Any, ecodes: Any, key_name: str) -> bool:
    capabilities = device.capabilities(verbose=False)
    supported_key_codes = set(capabilities.get(ecodes.EV_KEY, []))
    return getattr(ecodes, key_name) in supported_key_codes


def _check_keyboard(args: argparse.Namespace) -> CheckResult:
    if args.skip_keyboard:
        return _result("keyboard", "SKIP", "skipped by --skip-keyboard")

    try:
        from evdev import InputDevice, ecodes, list_devices
    except Exception as exc:
        return _result(
            "keyboard",
            "FAIL",
            "failed to import evdev",
            [f"{type(exc).__name__}: {exc}"],
        )

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
            if _keyboard_device_supports(device, ecodes, "KEY_V"):
                return _result(
                    "keyboard",
                    "PASS",
                    "keyboard event device is readable and supports KEY_V",
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
        "no readable keyboard device supports KEY_V",
        details,
    )


def _print_results(results: list[CheckResult]) -> None:
    width = max(len(result.name) for result in results) if results else 0
    for result in results:
        print(f"[{result.status:<4}] {result.name:<{width}}  {result.message}")
        for detail in result.details:
            print(f"       - {detail}")


def main() -> int:
    args = _parse_args()
    env_config = _load_config(args.env_config)

    results = [
        _check_env_config(env_config),
        _check_keyboard(args),
    ]
    _print_results(results)

    if any(result.status == "FAIL" for result in results) and not args.continue_on_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
