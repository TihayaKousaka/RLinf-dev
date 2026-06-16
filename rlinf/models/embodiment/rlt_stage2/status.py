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

"""Human-readable status helpers for RLT Stage 2 online switching."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


PHASE_WARMUP = "warmup"
PHASE_WARMUP_WAIT_ONLINE = "warmup_wait_online"
PHASE_ONLINE = "online"

PHASE_TO_ID = {
    PHASE_WARMUP: 0,
    PHASE_WARMUP_WAIT_ONLINE: 1,
    PHASE_ONLINE: 2,
}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for status payloads."""
    return datetime.now(timezone.utc).isoformat()


def resolve_training_phase(
    *,
    buffer_ready: bool,
    ready_for_online: bool,
) -> str:
    """Resolve the actor-side training phase.

    ``buffer_ready`` tracks whether enough replay data exists to train. The
    rollout online gate opens only after the configured warmup updates finish.
    """
    if not buffer_ready:
        return PHASE_WARMUP
    if not ready_for_online:
        return PHASE_WARMUP_WAIT_ONLINE
    return PHASE_ONLINE


def resolve_rollout_phase(
    *,
    ready_for_online: bool,
    student_control_rate: float,
) -> str:
    """Resolve the env-side rollout phase from rollout forward inputs."""
    if not ready_for_online:
        return PHASE_WARMUP
    if float(student_control_rate) > 0.0:
        return PHASE_ONLINE
    return PHASE_WARMUP_WAIT_ONLINE


def phase_id(phase: str) -> int:
    """Return the stable numeric id for a phase string."""
    return PHASE_TO_ID.get(phase, -1)


def metric_mean(
    metrics: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    """Return a float mean for a tensor/list/scalar metric."""
    if key not in metrics:
        return default
    value = metrics[key]
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is None:
        if value is None:
            return default
        if isinstance(value, list):
            if not value:
                return default
            return float(sum(float(item) for item in value) / len(value))
        return float(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, list):
        tensors = [
            item.detach().float().reshape(-1).cpu()
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, dtype=torch.float32).reshape(-1)
            for item in value
        ]
        tensors = [tensor for tensor in tensors if tensor.numel() > 0]
        if not tensors:
            return default
        return float(torch.cat(tensors).mean().item())
    if value is None:
        return default
    return float(value)


def write_status_json(path: str, payload: dict[str, Any]) -> None:
    """Atomically write a small JSON status payload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(tmp_path, path)
