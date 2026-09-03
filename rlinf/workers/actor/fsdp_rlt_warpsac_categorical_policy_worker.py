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

import torch

from rlinf.algorithms.rlt.categorical import (
    categorical_cross_entropy,
    categorical_q_values,
    project_categorical_distribution,
    select_min_categorical_logits,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_rlt_warpsac_scalar_policy_worker import (
    RLTWarpSACScalarFSDPPolicy,
)


class RLTWarpSACCategoricalFSDPPolicy(RLTWarpSACScalarFSDPPolicy):
    """Stage-4 WarpSAC worker with a fixed-support categorical twin critic."""

    def _categorical_support(self, device: torch.device) -> torch.Tensor:
        critic_cfg = self.cfg.algorithm.get("distributional_critic", {}) or {}
        distribution_type = str(critic_cfg.get("type", "categorical"))
        if distribution_type != "categorical":
            raise ValueError(
                "rlt_warpsac_categorical requires "
                "algorithm.distributional_critic.type='categorical'."
            )
        num_bins = int(critic_cfg.get("num_bins", 101))
        v_min = float(critic_cfg.get("v_min", -5.0))
        v_max = float(critic_cfg.get("v_max", 5.0))
        if num_bins < 2 or v_min >= v_max:
            raise ValueError(
                "Invalid categorical support: require num_bins >= 2 and v_min < "
                f"v_max, got num_bins={num_bins}, v_min={v_min}, v_max={v_max}."
            )
        model_cfg = self.cfg.actor.model
        model_support = (
            str(model_cfg.get("q_distribution_type", "scalar")),
            int(model_cfg.get("q_num_bins", 101)),
            float(model_cfg.get("q_v_min", -5.0)),
            float(model_cfg.get("q_v_max", 5.0)),
        )
        algorithm_support = (distribution_type, num_bins, v_min, v_max)
        if model_support != algorithm_support:
            raise ValueError(
                "actor.model categorical support must match "
                "algorithm.distributional_critic, got "
                f"model={model_support} and algorithm={algorithm_support}."
            )
        return torch.linspace(v_min, v_max, num_bins, device=device)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        if self.cfg.algorithm.get("q_head_type", "default") == "crossq":
            raise ValueError("Stage-4 categorical WarpSAC does not support CrossQ.")

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = batch["terminations"]
        if use_simulator_transition_replay(self.cfg):
            done_source = batch["dones"]
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )
        support = self._categorical_support(actions.device)

        with torch.no_grad():
            next_actions, _, _ = self._next_actions_for_critic_target(next_obs)
            next_logits = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_actions,
                return_logits=True,
            )
            if next_logits.shape[-1] != support.numel():
                raise ValueError(
                    "Categorical target critic output does not match configured "
                    f"support: {next_logits.shape[-1]} != {support.numel()}."
                )
            selected_next_logits = select_min_categorical_logits(next_logits, support)

            reward_target = self._discounted_chunk_rewards(rewards).to(
                device=support.device, dtype=torch.float32
            )
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = float(self.cfg.algorithm.gamma) ** reward_horizon
            if bootstrap_type == "always":
                bootstrap_mask = torch.ones_like(reward_target)
            elif bootstrap_type == "standard":
                bootstrap_mask = not_done.to(dtype=reward_target.dtype)
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

            target_atoms = reward_target + (
                bootstrap_mask * bootstrap_discount * support[None, :]
            )
            target_probs = project_categorical_distribution(
                selected_next_logits,
                target_atoms,
                support,
            )

        data_logits = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
            return_logits=True,
        )
        if data_logits.shape[-1] != support.numel():
            raise ValueError(
                "Categorical online critic output does not match configured support: "
                f"{data_logits.shape[-1]} != {support.numel()}."
            )
        critic_loss = categorical_cross_entropy(data_logits, target_probs)

        with torch.no_grad():
            q_data = categorical_q_values(data_logits, support)
            q_target = torch.sum(
                target_probs * support.to(target_probs)[None, :], dim=-1
            )
            target_entropy = -torch.sum(
                target_probs * torch.log(target_probs.clamp_min(1.0e-8)), dim=-1
            ).mean()
            metrics = {
                "q_data": q_data.mean().item(),
                "q_target": q_target.mean().item(),
                "categorical_target_entropy": target_entropy.item(),
                "categorical_support_clip_low": (target_atoms < support[0])
                .to(torch.float32)
                .mean()
                .item(),
                "categorical_support_clip_high": (target_atoms > support[-1])
                .to(torch.float32)
                .mean()
                .item(),
            }
        return critic_loss, metrics
