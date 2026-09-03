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

"""Categorical value-distribution utilities for RLT WarpSAC."""

import torch
import torch.nn.functional as F


def categorical_q_values(
    logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Return expected Q values for categorical logits."""
    if logits.shape[-1] != support.numel():
        raise ValueError(
            "Categorical logits and support disagree: "
            f"{logits.shape[-1]} != {support.numel()}."
        )
    support = support.to(device=logits.device, dtype=logits.dtype)
    return torch.sum(torch.softmax(logits, dim=-1) * support, dim=-1)


def select_min_categorical_logits(
    logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Select each batch item's distribution from its minimum-value critic."""
    if logits.ndim != 3:
        raise ValueError(
            "Expected categorical twin-Q logits with shape [B, Q, N], got "
            f"{tuple(logits.shape)}."
        )
    if logits.shape[1] < 2:
        raise ValueError(
            f"Categorical twin-Q requires at least two critics, got {logits.shape[1]}."
        )
    critic_indices = categorical_q_values(logits, support).argmin(dim=1)
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, critic_indices]


def project_categorical_distribution(
    source_logits: torch.Tensor,
    target_atoms: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project a transformed categorical distribution onto fixed support.

    Args:
        source_logits: Source-distribution logits with shape ``[B, N]``.
        target_atoms: Transformed locations for every source atom, ``[B, N]``.
        support: Evenly spaced target support with shape ``[N]``.

    Returns:
        Projected probabilities with shape ``[B, N]``.
    """
    if source_logits.ndim != 2 or target_atoms.shape != source_logits.shape:
        raise ValueError(
            "source_logits and target_atoms must both have shape [B, N], got "
            f"{tuple(source_logits.shape)} and {tuple(target_atoms.shape)}."
        )
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("Categorical support must be one-dimensional with >= 2 atoms.")
    if source_logits.shape[-1] != support.numel():
        raise ValueError(
            "Categorical source and support disagree: "
            f"{source_logits.shape[-1]} != {support.numel()}."
        )

    work_dtype = torch.float32
    logits = source_logits.to(dtype=work_dtype)
    atoms = target_atoms.to(device=logits.device, dtype=work_dtype)
    support = support.to(device=logits.device, dtype=work_dtype)
    bin_widths = support[1:] - support[:-1]
    if not torch.all(bin_widths > 0):
        raise ValueError("Categorical support must be strictly increasing.")
    if not torch.allclose(bin_widths, bin_widths[:1]):
        raise ValueError("Categorical projection requires evenly spaced support.")

    v_min, v_max = support[0], support[-1]
    positions = (atoms.clamp(v_min, v_max) - v_min) / bin_widths[0]
    lower = positions.floor().long().clamp(0, support.numel() - 1)
    upper = positions.ceil().long().clamp(0, support.numel() - 1)
    source_probs = torch.softmax(logits, dim=-1)

    projected = torch.zeros_like(source_probs)
    lower_weight = upper.to(work_dtype) - positions
    lower_weight = lower_weight + (lower == upper).to(work_dtype)
    upper_weight = positions - lower.to(work_dtype)
    projected.scatter_add_(-1, lower, source_probs * lower_weight)
    projected.scatter_add_(-1, upper, source_probs * upper_weight)
    return projected.detach()


def categorical_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute mean cross-entropy for one or more categorical critics."""
    if logits.ndim != 3 or target_probs.ndim != 2:
        raise ValueError(
            "Expected logits [B, Q, N] and target probabilities [B, N], got "
            f"{tuple(logits.shape)} and {tuple(target_probs.shape)}."
        )
    if (
        logits.shape[0] != target_probs.shape[0]
        or logits.shape[2] != target_probs.shape[1]
    ):
        raise ValueError("Categorical logits and targets have incompatible shapes.")
    targets = target_probs[:, None, :].to(device=logits.device, dtype=torch.float32)
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    return -torch.sum(targets * log_probs, dim=-1).mean()
