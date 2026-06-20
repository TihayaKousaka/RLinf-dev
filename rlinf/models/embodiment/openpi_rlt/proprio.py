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

"""State utilities for OpenPI-RLT."""

from __future__ import annotations

import torch


def resolve_proprio_dim(default_dim: int, **_: object) -> int:
    """Return the configured full state dimension used by OpenPI-RLT."""
    return int(default_dim)


def select_proprio(state: torch.Tensor) -> torch.Tensor:
    """Use the full environment state as the RLT proprio feature."""
    if state.ndim != 2:
        raise ValueError(
            "OpenPI-RLT observation.state must be a 2D tensor [B, state_dim], "
            f"got shape={tuple(state.shape)}."
        )
    return state.to(dtype=torch.float32)
