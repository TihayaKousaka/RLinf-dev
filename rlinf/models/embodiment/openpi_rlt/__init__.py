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

from omegaconf import DictConfig


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


__all__ = ["get_model"]
