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

"""Pluggable rollout hooks for HuggingFace embodied rollout workers."""

from __future__ import annotations

from typing import Any

import torch


class NoopHFRolloutAdapter:
    """Default rollout hooks for standard embodied policies."""

    enabled = False

    def __init__(
        self,
        *,
        cfg,
        student_model,
        expert_model_getter,
        has_expert_model_config: bool,
    ) -> None:
        del cfg, expert_model_getter, has_expert_model_config
        self.student_model = student_model

    def expert_model_path(
        self,
        configured_path: str,
        expert_ckpt_path: str | None,
    ) -> str:
        del expert_ckpt_path
        return configured_path

    def configure_expert_model(self, expert_model_config) -> None:
        del expert_model_config

    def rollout_state_dict(self) -> dict[str, torch.Tensor]:
        return self.student_model.state_dict()

    def use_dagger_beta(self) -> bool:
        return True

    def allow_bootstrap_values(self) -> bool:
        return True

    def predict(
        self,
        *,
        env_obs: dict[str, Any],
        policy_info: dict[str, torch.Tensor] | None,
        model_kwargs: dict[str, Any],
        mode: str,
        allow_expert: bool,
        update_version: int,
    ) -> Any | None:
        del env_obs, policy_info, model_kwargs, mode, allow_expert, update_version
        return None

    def encode_step_trace(
        self,
        step_obs: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        del step_obs
        return {}

    def final_forward_inputs(self, result: dict[str, Any]) -> dict[str, Any]:
        del result
        return {}


def build_hf_rollout_adapter(
    *,
    cfg,
    student_model,
    expert_model_getter,
    has_expert_model_config: bool,
):
    """Build the optional rollout adapter for the configured algorithm."""
    from rlinf.models.embodiment.rlt_stage2.rollout import (
        RLTStage2RolloutAdapter,
    )

    adapter_cls = (
        RLTStage2RolloutAdapter
        if RLTStage2RolloutAdapter.is_enabled_cfg(cfg)
        else NoopHFRolloutAdapter
    )
    return adapter_cls(
        cfg=cfg,
        student_model=student_model,
        expert_model_getter=expert_model_getter,
        has_expert_model_config=has_expert_model_config,
    )
