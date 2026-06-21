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

"""OpenPI-RLT model package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from .rollout import (
    build_rlt_step_trace_infos,
    build_rlt_step_trace_obs,
    extract_rlt_env_metrics,
    should_build_rlt_step_trace,
)


@dataclass(frozen=True)
class OpenPIRLTEnvWorkerHandler:
    """Model-side EnvWorker extension for RLT Stage 2 metadata."""

    model_cfg: DictConfig

    def should_build_step_trace(self) -> bool:
        return should_build_rlt_step_trace(self.model_cfg)

    def build_step_trace_obs(self, obs_list: Any) -> dict[str, Any] | None:
        return build_rlt_step_trace_obs(obs_list)

    def build_step_trace_infos(
        self,
        infos_list: Any,
        *,
        expected_trace_len: int | None = None,
    ) -> dict[str, Any] | None:
        return build_rlt_step_trace_infos(
            infos_list,
            expected_trace_len=expected_trace_len,
        )

    def extract_env_metrics(
        self,
        *,
        forward_inputs: dict[str, Any] | None = None,
        env_infos: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return extract_rlt_env_metrics(
            forward_inputs=forward_inputs,
            env_infos=env_infos,
        )


def get_model(cfg: DictConfig, torch_dtype=None):
    del torch_dtype
    model_type = str(cfg.get("model_type", ""))
    if model_type == "rlt_stage1":
        from .stage1_policy import RLTStage1Policy

        return RLTStage1Policy(cfg)
    if model_type == "rlt_stage2":
        from .stage2_policy import RLTStage2Policy

        return RLTStage2Policy(cfg)
    raise ValueError(
        f"Unsupported OpenPI-RLT model_type: {model_type!r}. "
        "Expected 'rlt_stage1' or 'rlt_stage2'."
    )


def get_env_worker_handler(model_cfg: DictConfig) -> OpenPIRLTEnvWorkerHandler | None:
    if model_cfg is None or str(model_cfg.get("model_type", "")) != "rlt_stage2":
        return None
    return OpenPIRLTEnvWorkerHandler(model_cfg=model_cfg)


__all__ = ["OpenPIRLTEnvWorkerHandler", "get_env_worker_handler", "get_model"]
