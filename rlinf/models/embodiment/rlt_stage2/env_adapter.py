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

"""Env-worker helpers for RLT Stage 2 rollout collection."""

from __future__ import annotations

import os
from typing import Any

import torch

from .status import (
    metric_mean,
    phase_id,
    resolve_rollout_phase,
    utc_timestamp,
    write_status_json,
)


def is_rlt_stage2_td3_cfg(cfg) -> bool:
    return (
        cfg.algorithm.get("loss_type", None) == "rlt_td3"
        and cfg.actor.model.get("model_type", None) == "rlt_stage2"
    )


def build_step_obs(
    cfg,
    start_obs: dict[str, Any] | None,
    obs_list,
) -> dict[str, Any] | None:
    if start_obs is None or not isinstance(obs_list, (list, tuple)) or not obs_list:
        return None

    stride = int(cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
    if stride <= 0:
        return None

    step_obs_list = [start_obs, *obs_list]
    offsets = _sparse_step_obs_offsets(cfg, len(step_obs_list))
    step_obs: dict[str, Any] = {}
    batch_size = _infer_obs_batch_size(step_obs_list[0])
    for key in step_obs_list[0].keys():
        if not offsets:
            continue
        values = [step_obs_list[offset].get(key, None) for offset in offsets]
        first_non_none = next((value for value in values if value is not None), None)
        if first_non_none is None:
            step_obs[key] = None
        elif isinstance(first_non_none, torch.Tensor):
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent RLT step_obs key {key!r}: "
                    "tensor values contain None."
                )
            values = [
                value.to(first_non_none.device)
                if value.device != first_non_none.device
                else value
                for value in values
            ]
            step_obs[key] = torch.stack(values, dim=0)
        elif isinstance(first_non_none, list):
            step_obs[key] = values
        else:
            step_obs[key] = values
    step_obs["_rlt_step_offsets"] = torch.tensor(
        offsets,
        dtype=torch.long,
    )[:, None].expand(len(offsets), batch_size).contiguous()
    return step_obs


def emit_rollout_status(
    *,
    cfg,
    env_metrics: dict[str, torch.Tensor],
    rank: int,
    last_logged_phase: str | None,
    log_info,
) -> str | None:
    if not is_rlt_stage2_td3_cfg(cfg):
        return last_logged_phase
    ready_value = metric_mean(env_metrics, "rlt_ready_for_online")
    if ready_value is None:
        return last_logged_phase

    ready_for_online = bool(ready_value >= 0.5)
    in_critical_phase_rate = metric_mean(
        env_metrics,
        "rlt_in_critical_phase",
        default=0.0,
    )
    record_transition_rate = metric_mean(
        env_metrics,
        "rlt_record_transition",
        default=0.0,
    )
    student_control_rate = metric_mean(
        env_metrics,
        "student_control_rate",
        default=0.0,
    )
    phase = resolve_rollout_phase(
        ready_for_online=ready_for_online,
        student_control_rate=float(student_control_rate),
    )
    phase_numeric_id = phase_id(phase)

    status_like = env_metrics["rlt_ready_for_online"].detach().float().reshape(-1)
    env_metrics["rlt_status_phase_id"] = torch.full_like(
        status_like,
        float(phase_numeric_id),
    )
    env_metrics["rlt_status_ready_for_online"] = torch.full_like(
        status_like,
        float(ready_for_online),
    )
    env_metrics["rlt_status_in_critical_phase_rate"] = torch.full_like(
        status_like,
        float(in_critical_phase_rate),
    )
    env_metrics["rlt_status_record_transition_rate"] = torch.full_like(
        status_like,
        float(record_transition_rate),
    )
    env_metrics["rlt_status_student_control_rate"] = torch.full_like(
        status_like,
        float(student_control_rate),
    )

    if rank == 0 and phase != last_logged_phase:
        log_info(
            "[RLT_STATUS][env] "
            f"phase={phase} ready={int(ready_for_online)} "
            f"critical={float(in_critical_phase_rate):.2f} "
            f"record={float(record_transition_rate):.2f} "
            f"student={float(student_control_rate):.2f}"
        )
        last_logged_phase = phase

    status_dir = os.path.join(cfg.runner.logger.log_path, "status")
    write_status_json(
        os.path.join(status_dir, f"rlt_env_status_rank{rank}.json"),
        {
            "timestamp": utc_timestamp(),
            "component": "env",
            "rank": rank,
            "phase": phase,
            "phase_id": phase_numeric_id,
            "ready_for_online": ready_for_online,
            "in_critical_phase_rate": float(in_critical_phase_rate),
            "record_transition_rate": float(record_transition_rate),
            "student_control_rate": float(student_control_rate),
        },
    )
    return last_logged_phase


def _sparse_step_obs_offsets(cfg, step_count: int) -> list[int]:
    stride = int(cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
    chunk_len = int(cfg.actor.model.num_action_chunks)
    if stride <= 0 or chunk_len <= 0:
        return []

    offsets = set()
    offset = 0
    while True:
        offset = (offset + stride) % chunk_len
        if offset == 0 or offset in offsets:
            break
        if offset < step_count:
            offsets.add(offset)
    return sorted(offsets)


def _infer_obs_batch_size(obs: dict[str, Any]) -> int:
    for value in obs.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
    raise ValueError("Cannot infer RLT step_obs batch size from observation.")
