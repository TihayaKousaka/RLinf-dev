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

"""Opt-in parameter-update diagnostics for unsharded ART actors."""

import torch


class ARTParameterProbe:
    """Keep a CPU reference to trainable weights and FP32 master weights.

    One reference spans both ART updates. Diagnostics intentionally transfer
    parameters to CPU; enable them for short numerical checks, not long runs.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.parameters = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            master = optimizer.state.get(parameter, {}).get("fp32_master_param")
            self.parameters.append(
                (
                    parameter,
                    parameter.detach().cpu().clone(),
                    master,
                    None if master is None else master.detach().cpu().clone(),
                )
            )

    @torch.no_grad()
    def metrics(self, prefix: str, device: torch.device) -> dict[str, float]:
        """Return cumulative change from the pre-u1 reference, by weight dtype."""
        categories = ("bf16", "fp16", "fp32", "master")
        stats = torch.zeros(len(categories), 3, dtype=torch.float64)
        for parameter, before, master, master_before in self.parameters:
            # Adam initializes state lazily: the first probe can precede the
            # creation of master parameters in optimizer.step().
            if master is None:
                master = self.optimizer.state.get(parameter, {}).get(
                    "fp32_master_param"
                )
                if master is not None:
                    master_before = before.float()
            dtype = {
                torch.bfloat16: "bf16",
                torch.float16: "fp16",
                torch.float32: "fp32",
            }[parameter.dtype]
            pairs = [(dtype, parameter, before)]
            if master is not None:
                pairs.append(("master", master, master_before))
            for category, current, reference in pairs:
                current = current.detach().cpu()
                delta = current.float() - reference.float()
                index = categories.index(category)
                stats[index, 0] += current.numel()
                stats[index, 1] += (current != reference).sum()
                stats[index, 2] += delta.square().sum(dtype=torch.float64)
        stats = stats.to(device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
            # This probe is restricted to NO_SHARD: each rank owns a replica.
            stats /= torch.distributed.get_world_size()
        result = {}
        for category, (count, changed, squared) in zip(categories, stats.tolist()):
            if count:
                result[f"{prefix}_{category}_numel"] = count
                result[f"{prefix}_{category}_changed_frac"] = changed / count
                result[f"{prefix}_{category}_update_rms"] = (squared / count) ** 0.5
        return result


class ARTZeroUpdateProbe:
    """Pool pre-u1 errors over actual training microbatches, before any step."""

    def __init__(self, device: torch.device):
        self.stats = torch.zeros(8, device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, current, old, references, coordinate_mask, terms) -> None:
        mask = torch.broadcast_to(coordinate_mask, current.shape)
        eval_velocity, train_velocity, repeat_velocity = references
        differences = (
            eval_velocity - old,
            train_velocity - eval_velocity,
            repeat_velocity - train_velocity,
            current - train_velocity,
        )
        active = terms["active"]
        self.stats += torch.stack(
            [
                *(
                    torch.where(mask, d.float().square(), 0.0).sum()
                    for d in differences
                ),
                mask.sum(),
                active.sum(),
                terms["masked"].sum(),
                (terms["movement_score"] * active).sum(),
            ]
        ).to(self.stats)

    def metrics(self) -> dict[str, float]:
        stats = self.stats.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        *squares, coordinates, active, masked, score = stats.tolist()
        prefix = "actor/zero_update"
        return {
            **{
                f"{prefix}/{name}_rms": (squared / max(coordinates, 1)) ** 0.5
                for name, squared in zip(
                    ("batch", "train_eval", "train_repeat", "grad_nograd"), squares
                )
            },
            f"{prefix}/active_count": active,
            f"{prefix}/trainref_mask_frac": masked / max(active, 1),
            f"{prefix}/trainref_score_active": score / max(active, 1),
        }


@torch.no_grad()
def art_repeat_candidate_metrics(
    endpoints, returns, action_mask, valid_groups, group_size, *, device=None
):
    """Compare two independent continuations of each of K/2 fixed candidates.

    Adjacent rows must have identical endpoints and executed-coordinate masks.
    The covariance cosine compares the two K/2-candidate reward estimates, not
    two independent groups of K candidates.
    """
    actions = endpoints.float().reshape(-1, group_size, endpoints[0].numel())
    rewards = returns.float().reshape(-1, group_size).to(actions.device)
    mask = action_mask[..., None].expand_as(endpoints).reshape_as(actions)
    valid = valid_groups.to(actions.device)
    difference = torch.where(
        mask[:, ::2] | mask[:, 1::2],
        (actions[:, ::2] - actions[:, 1::2]).abs(),
        0.0,
    ).amax(-1)
    matched = (
        valid[:, None]
        & (difference <= 1e-6)
        & (mask[:, ::2] == mask[:, 1::2]).all(-1)
        & mask[:, ::2].any(-1)
    )
    disagreement = (rewards[:, ::2] != rewards[:, 1::2]) & matched
    covariances = []
    for offset in (0, 1):
        x, r, m = actions[:, offset::2], rewards[:, offset::2], mask[:, offset::2]
        count = m.sum(1).clamp_min(1)
        mean_x = (x * m).sum(1) / count
        mean_r = (r[..., None] * m).sum(1) / count
        covariances.append(
            ((x - mean_x[:, None]) * (r[..., None] - mean_r[:, None]) * m).sum(1)
            / (count - 1).clamp_min(1)
        )
    norms = covariances[0].norm(dim=1) * covariances[1].norm(dim=1)
    signal = matched.all(1) & (norms > 1e-20)
    cosine = (covariances[0] * covariances[1]).sum(1) / norms.clamp_min(1e-20)
    stats = torch.stack(
        [
            valid.sum(),
            matched.sum(),
            disagreement.sum(),
            signal.sum(),
            cosine[signal].sum(),
            ((cosine < 0) & signal).sum(),
        ]
    ).to(device=device or actions.device, dtype=torch.float64)
    maximum = torch.where(valid[:, None], difference, 0.0).max().to(stats.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    groups, pairs, flips, signals, cosine_sum, opposite = stats.tolist()
    return {
        "actor/repeat/groups": groups,
        "actor/repeat/matched_pairs": pairs,
        "actor/repeat/endpoint_max_abs": maximum.item(),
        "actor/repeat/reward_disagreement_frac": flips / max(pairs, 1),
        "actor/repeat/both_signal_groups": signals,
        "actor/repeat/covariance_cosine": cosine_sum / max(signals, 1),
        "actor/repeat/opposite_direction_frac": opposite / max(signals, 1),
    }
