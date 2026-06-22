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

"""Rollout adapter for RLT Stage 2 TD3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np
import torch


class TransitionSource(IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


def resolve_collection_phase_id(phase: str | int | None) -> int:
    if phase is None:
        return COLLECTION_PHASE_UNKNOWN
    if isinstance(phase, int):
        return int(phase)
    phase_name = str(phase).split(":", 1)[0].lower()
    if phase_name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if phase_name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def resolve_chunk_source(source_chunk: np.ndarray) -> int:
    values = {int(value) for value in np.asarray(source_chunk).reshape(-1)}
    if not values:
        return int(TransitionSource.RL)
    if int(TransitionSource.MIXED) in values or len(values) > 1:
        return int(TransitionSource.MIXED)
    return next(iter(values))


def extract_rlt_env_metrics(
    *,
    forward_inputs: dict[str, Any] | None = None,
    env_infos: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract RLT rollout status metrics without teaching env worker RLT fields."""
    metrics: dict[str, torch.Tensor] = {}

    if isinstance(forward_inputs, dict):
        metric_keys = (
            ("intervention_flags", "expert_intervention_actual_rate"),
            ("intervention_requested", "expert_intervention_requested_rate"),
            ("ready_for_online", "rlt_ready_for_online"),
            ("in_critical_phase", "rlt_in_critical_phase"),
            ("record_transition", "rlt_record_transition"),
            ("student_control", "student_control_rate"),
        )
        for forward_key, metric_key in metric_keys:
            value = forward_inputs.get(forward_key)
            if isinstance(value, torch.Tensor):
                metrics[metric_key] = value.detach().float().reshape(-1).cpu()

    policy_info = env_infos.get("policy_info") if isinstance(env_infos, dict) else None
    deviation = policy_info.get("deviation") if isinstance(policy_info, dict) else None
    if isinstance(deviation, torch.Tensor):
        metrics["deviation_rate"] = deviation.detach().float().mean().reshape(1).cpu()

    return metrics


REQUIRED_RLT_STAGE2_FORWARD_INPUTS = (
    "x",
    "a_tilde",
    "base_a_tilde",
    "ref_chunk",
    "action",
    "action_chunk",
    "student_control",
    "intervention_flags",
    "source_chunk",
    "source",
    "collection_phase_id",
    "intervention_requested",
    "intervention_phase",
    "in_critical_phase",
    "record_transition",
    "ready_for_online",
    "online_gate_step",
)


def resolve_source_from_source_chunk(source_chunk: torch.Tensor) -> torch.Tensor:
    """Collapse per-step source labels to one chunk-level source label."""
    if not isinstance(source_chunk, torch.Tensor) or source_chunk.ndim != 2:
        raise ValueError(
            "RLT Stage2 source_chunk must have shape [B, T], got "
            f"{_shape_str(source_chunk)}."
        )
    return torch.where(
        source_chunk.eq(source_chunk[:, :1]).all(dim=1, keepdim=True),
        source_chunk[:, :1],
        torch.full(
            (source_chunk.shape[0], 1),
            int(TransitionSource.MIXED),
            dtype=torch.uint8,
            device=source_chunk.device,
        ),
    )


def require_rlt_stage2_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    batch_size: int,
    chunk_length: int,
    action_dim: int,
    context: str,
) -> dict[str, Any]:
    """Validate that rollout emitted the canonical RLT Stage 2 fields."""
    missing = [
        key for key in REQUIRED_RLT_STAGE2_FORWARD_INPUTS if key not in forward_inputs
    ]
    if missing:
        raise RuntimeError(
            f"RLT Stage2 {context} forward_inputs missing required keys: {missing}. "
            "Build them in the rollout RLT path instead of relying on silent "
            "fallback defaults."
        )

    action_chunk_dim = int(chunk_length) * int(action_dim)
    expected_shapes = {
        "x": (batch_size, None),
        "a_tilde": (batch_size, action_chunk_dim),
        "base_a_tilde": (batch_size, action_chunk_dim),
        "ref_chunk": (batch_size, action_chunk_dim),
        "action": (batch_size, action_chunk_dim),
        "action_chunk": (batch_size, action_chunk_dim),
        "student_control": (batch_size, 1),
        "intervention_flags": (batch_size, chunk_length),
        "source_chunk": (batch_size, chunk_length),
        "source": (batch_size, 1),
        "collection_phase_id": (batch_size, 1),
        "intervention_requested": (batch_size, 1),
        "intervention_phase": (batch_size, 1),
        "in_critical_phase": (batch_size, 1),
        "record_transition": (batch_size, 1),
        "ready_for_online": (batch_size, 1),
        "online_gate_step": (batch_size, 1),
    }
    for key, expected_shape in expected_shapes.items():
        _require_tensor_shape(
            forward_inputs[key],
            expected_shape,
            field_name=key,
            context=context,
        )
    return forward_inputs


def _require_tensor_shape(
    value: Any,
    expected_shape: tuple[int | None, ...],
    *,
    field_name: str,
    context: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"RLT Stage2 {context} forward_inputs[{field_name!r}] must be a "
            f"torch.Tensor, got {type(value).__name__}."
        )
    if value.ndim != len(expected_shape):
        raise ValueError(
            f"RLT Stage2 {context} forward_inputs[{field_name!r}] must have "
            f"{len(expected_shape)} dims, got shape {_shape_str(value)}."
        )
    for dim, expected in enumerate(expected_shape):
        if expected is not None and int(value.shape[dim]) != int(expected):
            raise ValueError(
                f"RLT Stage2 {context} forward_inputs[{field_name!r}] shape "
                f"mismatch: expected {expected_shape}, got {_shape_str(value)}."
            )


def _shape_str(value: Any) -> str:
    return "None" if value is None else str(tuple(getattr(value, "shape", ())))


@dataclass(frozen=True)
class RLTActionRouteInputs:
    student_actions: torch.Tensor
    base_flat: torch.Tensor
    expert_actions: torch.Tensor | None
    expert_takeover: torch.Tensor
    requested_expert_takeover: torch.Tensor
    intervention_phase: torch.Tensor
    in_critical_phase: torch.Tensor
    record_transition: torch.Tensor
    ready_for_online: bool
    online_gate_step: int
    chunk_length: int
    action_dim: int


@dataclass(frozen=True)
class RLTActionRouteResult:
    actions: torch.Tensor
    action_flat: torch.Tensor
    base_flat: torch.Tensor
    student_control: torch.Tensor
    intervention_flags: torch.Tensor
    source_chunk: torch.Tensor
    source: torch.Tensor
    collection_phase_id: torch.Tensor
    intervention_requested: torch.Tensor
    intervention_phase: torch.Tensor
    in_critical_phase: torch.Tensor
    record_transition: torch.Tensor
    ready_for_online: torch.Tensor
    online_gate_step: torch.Tensor

    def to_forward_input_updates(self) -> dict[str, torch.Tensor]:
        return {
            "base_a_tilde": self.base_flat,
            "ref_chunk": self.base_flat.detach(),
            "action": self.action_flat.detach(),
            "action_chunk": self.action_flat.detach(),
            "student_control": self.student_control[:, None],
            "intervention_flags": self.intervention_flags,
            "source_chunk": self.source_chunk,
            "source": self.source,
            "collection_phase_id": self.collection_phase_id,
            "intervention_requested": self.intervention_requested[:, None],
            "intervention_phase": self.intervention_phase[:, None],
            "in_critical_phase": self.in_critical_phase[:, None],
            "record_transition": self.record_transition[:, None],
            "ready_for_online": self.ready_for_online,
            "online_gate_step": self.online_gate_step,
        }


def route_rlt_stage2_actions(inputs: RLTActionRouteInputs) -> RLTActionRouteResult:
    """Route execution between base VLA, RLT actor, and expert correction."""
    _validate_route_inputs(inputs)

    base_actions = inputs.base_flat.reshape(
        inputs.base_flat.shape[0],
        inputs.chunk_length,
        inputs.action_dim,
    )
    actor_control = (
        torch.full(
            (inputs.student_actions.shape[0],),
            bool(inputs.ready_for_online),
            dtype=torch.bool,
            device=inputs.student_actions.device,
        )
        & inputs.in_critical_phase
    )
    actions = torch.where(
        actor_control[:, None, None],
        inputs.student_actions,
        base_actions,
    )

    intervention_flags = torch.zeros(
        (actions.shape[0], inputs.chunk_length),
        dtype=torch.bool,
        device=actions.device,
    )
    source_chunk = torch.full(
        (actions.shape[0], inputs.chunk_length),
        int(TransitionSource.BASE),
        dtype=torch.uint8,
        device=actions.device,
    )
    source_chunk[actor_control] = int(TransitionSource.RL)

    if inputs.expert_actions is not None:
        expert_mask = inputs.expert_takeover[:, None, None].to(actions.device)
        actions = torch.where(expert_mask, inputs.expert_actions, actions)
        intervention_flags[inputs.expert_takeover] = True
        source_chunk[inputs.expert_takeover] = int(TransitionSource.HUMAN)

    action_flat = actions.reshape(actions.shape[0], -1)
    return RLTActionRouteResult(
        actions=actions,
        action_flat=action_flat,
        base_flat=inputs.base_flat,
        student_control=actor_control,
        intervention_flags=intervention_flags,
        source_chunk=source_chunk,
        source=resolve_source_from_source_chunk(source_chunk),
        collection_phase_id=torch.full(
            (actions.shape[0], 1),
            COLLECTION_PHASE_ONLINE
            if inputs.ready_for_online
            else COLLECTION_PHASE_WARMUP,
            dtype=torch.uint8,
            device=actions.device,
        ),
        intervention_requested=inputs.requested_expert_takeover.to(
            actions.device,
            dtype=torch.bool,
        ),
        intervention_phase=inputs.intervention_phase.to(
            actions.device,
            dtype=torch.float32,
        ),
        in_critical_phase=inputs.in_critical_phase.to(actions.device, dtype=torch.bool),
        record_transition=inputs.record_transition.to(actions.device, dtype=torch.bool),
        ready_for_online=torch.full(
            (actions.shape[0], 1),
            bool(inputs.ready_for_online),
            dtype=torch.bool,
            device=actions.device,
        ),
        online_gate_step=torch.full(
            (actions.shape[0], 1),
            float(inputs.online_gate_step),
            dtype=torch.float32,
            device=actions.device,
        ),
    )


def _validate_route_inputs(inputs: RLTActionRouteInputs) -> None:
    if inputs.student_actions.ndim != 3:
        raise ValueError(
            "RLTActionRouter student_actions must have shape [B, T, A], got "
            f"{_shape_str(inputs.student_actions)}."
        )
    batch_size = int(inputs.student_actions.shape[0])
    expected_action_shape = (batch_size, inputs.chunk_length, inputs.action_dim)
    if tuple(inputs.student_actions.shape) != expected_action_shape:
        raise ValueError(
            "RLTActionRouter student_actions shape mismatch: expected "
            f"{expected_action_shape}, got {_shape_str(inputs.student_actions)}."
        )
    expected_flat_shape = (batch_size, inputs.chunk_length * inputs.action_dim)
    if tuple(inputs.base_flat.shape) != expected_flat_shape:
        raise ValueError(
            "RLTActionRouter base_flat shape mismatch: expected "
            f"{expected_flat_shape}, got {_shape_str(inputs.base_flat)}."
        )
    if (
        inputs.expert_actions is not None
        and tuple(inputs.expert_actions.shape) != expected_action_shape
    ):
        raise ValueError(
            "RLTActionRouter expert_actions shape mismatch: expected "
            f"{expected_action_shape}, got {_shape_str(inputs.expert_actions)}."
        )
    for name in (
        "expert_takeover",
        "requested_expert_takeover",
        "intervention_phase",
        "in_critical_phase",
        "record_transition",
    ):
        value = getattr(inputs, name)
        if tuple(value.shape) != (batch_size,):
            raise ValueError(
                f"RLTActionRouter {name} must have shape [{batch_size}], got "
                f"{_shape_str(value)}."
            )


@dataclass(frozen=True)
class RLTStage2RolloutRouteConfig:
    ready_for_online: bool
    online_gate_step: int
    allow_expert: bool
    chunk_length: int
    action_dim: int
    in_critical_phase_default: bool = True


@dataclass(frozen=True)
class RLTStage2RolloutRouteResult:
    actions: torch.Tensor
    result: dict[str, Any]
    expert_label_flag: bool


def route_rlt_stage2_rollout(
    *,
    env_obs: dict[str, Any],
    policy_info: dict[str, torch.Tensor] | None,
    student_model: Any,
    expert_model_getter: Callable[[], Any],
    model_kwargs: dict[str, Any],
    cfg: RLTStage2RolloutRouteConfig,
    student_prediction: tuple[torch.Tensor, dict[str, Any]] | None = None,
) -> RLTStage2RolloutRouteResult:
    """Run student/expert inference and build canonical RLT forward inputs."""
    if student_prediction is None:
        student_actions, result = student_model.predict_action_batch(
            env_obs=env_obs,
            **model_kwargs,
        )
    else:
        student_actions, result = student_prediction
    if "forward_inputs" not in result:
        raise RuntimeError(
            "RLT Stage2 rollout requires result['forward_inputs']; "
            "model.predict_action_batch must expose cached rollout features."
        )
    forward_inputs = result["forward_inputs"]
    if "a_tilde" not in forward_inputs:
        raise RuntimeError(
            "RLT Stage2 rollout requires forward_inputs['a_tilde']; "
            "the rollout policy must expose the base/reference action chunk."
        )
    base_flat = forward_inputs["a_tilde"].detach()
    batch_size = int(student_actions.shape[0])

    expert_takeover = _bool_policy_info(
        policy_info,
        "expert_takeover",
        batch_size=batch_size,
        device=student_actions.device,
        default=False,
    )
    requested_expert_takeover = expert_takeover
    expert_takeover = expert_takeover & cfg.ready_for_online & cfg.allow_expert
    in_critical_phase = _bool_policy_info(
        policy_info,
        "in_critical_phase",
        batch_size=batch_size,
        device=student_actions.device,
        default=cfg.in_critical_phase_default,
    )
    record_transition = _bool_policy_info(
        policy_info,
        "record_transition",
        batch_size=batch_size,
        device=student_actions.device,
        default=True,
    )
    intervention_phase = _float_policy_info(
        policy_info,
        "intervention_phase",
        batch_size=batch_size,
        device=student_actions.device,
        default=0.0,
    )

    expert_actions = None
    expert_label_flag = False
    if cfg.allow_expert and expert_takeover.any():
        expert_model = expert_model_getter()
        if getattr(expert_model, "act_as_vla_reference", False) and hasattr(
            expert_model,
            "predict_vla_reference_action_batch",
        ):
            expert_actions, _ = expert_model.predict_vla_reference_action_batch(
                env_obs=env_obs,
                **model_kwargs,
            )
        else:
            expert_actions, _ = expert_model.predict_action_batch(
                env_obs=env_obs,
                **model_kwargs,
            )
        expert_label_flag = True

    route = route_rlt_stage2_actions(
        RLTActionRouteInputs(
            student_actions=student_actions,
            base_flat=base_flat,
            expert_actions=expert_actions,
            expert_takeover=expert_takeover,
            requested_expert_takeover=requested_expert_takeover,
            intervention_phase=intervention_phase,
            in_critical_phase=in_critical_phase,
            record_transition=record_transition,
            ready_for_online=cfg.ready_for_online,
            online_gate_step=cfg.online_gate_step,
            chunk_length=cfg.chunk_length,
            action_dim=cfg.action_dim,
        )
    )
    actions = route.actions
    forward_inputs.update(route.to_forward_input_updates())
    require_rlt_stage2_forward_inputs(
        forward_inputs,
        batch_size=actions.shape[0],
        chunk_length=cfg.chunk_length,
        action_dim=cfg.action_dim,
        context="predict",
    )
    return RLTStage2RolloutRouteResult(
        actions=actions,
        result=result,
        expert_label_flag=expert_label_flag,
    )


def _bool_policy_info(
    policy_info: dict[str, torch.Tensor] | None,
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if policy_info is None or key not in policy_info:
        return torch.full(
            (batch_size,),
            bool(default),
            dtype=torch.bool,
            device=device,
        )
    value = torch.as_tensor(policy_info[key], device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(value.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return value.reshape(batch_size, -1).to(torch.bool).any(dim=1)


def _float_policy_info(
    policy_info: dict[str, torch.Tensor] | None,
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    default: float,
) -> torch.Tensor:
    if policy_info is None or key not in policy_info:
        return torch.full(
            (batch_size,),
            float(default),
            dtype=torch.float32,
            device=device,
        )
    value = torch.as_tensor(policy_info[key], device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            float(value.reshape(-1)[0].item()),
            dtype=torch.float32,
            device=device,
        )
    return value.reshape(batch_size, -1)[:, -1].to(torch.float32)
