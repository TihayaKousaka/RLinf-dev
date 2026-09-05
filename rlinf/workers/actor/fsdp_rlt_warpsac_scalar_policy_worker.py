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
import torch.nn.functional as F

from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3FSDPPolicy


class RLTWarpSACScalarFSDPPolicy(RLTTD3FSDPPolicy):
    """RLT WarpSAC critic update with the current TD3 actor.

    Stage 3 keeps the TD3 actor objective, replay data flow, and twin scalar
    Q heads unchanged. The only algorithmic change is that the bootstrap
    action is sampled from the online actor while the bootstrap value is
    evaluated by the target critic.
    """

    def _next_actions_for_critic_target(self, next_obs):
        return self.model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
        )


class RLTWarpSACFSDPPolicy(RLTWarpSACScalarFSDPPolicy):
    """Faithful scalar WarpSAC learner with the RLT data and BC contract."""

    def setup_sac_components(self):
        super().setup_sac_components()
        if self.alpha_optimizer is None:
            raise ValueError(
                "rlt_warpsac requires automatic entropy tuning; set "
                "algorithm.entropy_tuning.alpha_type to 'softplus' or 'exp'."
            )
        if not bool(self.cfg.algorithm.get("backup_entropy", True)):
            raise ValueError("rlt_warpsac requires algorithm.backup_entropy=True.")

    @staticmethod
    def _concat_obs(curr_obs: dict, next_obs: dict) -> dict:
        keys = set(curr_obs) | set(next_obs)
        if set(curr_obs) != set(next_obs):
            raise ValueError(
                "current and next RLT observations must have the same keys"
            )
        return {key: torch.cat([curr_obs[key], next_obs[key]], dim=0) for key in keys}

    def _log_prob_sum(self, log_prob: torch.Tensor) -> torch.Tensor:
        if log_prob.ndim == 1:
            log_prob = log_prob.unsqueeze(-1)
        return log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1, keepdim=True)

    def _entropy_alpha(self) -> torch.Tensor:
        return self.entropy_temp.compute_alpha().detach().to(self.torch_dtype)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        if agg_q not in {"min", "mean"}:
            raise ValueError(f"Unsupported WarpSAC critic aggregation: {agg_q!r}")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = (
            batch["dones"]
            if use_simulator_transition_replay(self.cfg)
            else batch["terminations"]
        )
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
                training=False,
            )
            next_log_prob = self._log_prob_sum(next_log_prob)
            combined_obs = self._concat_obs(curr_obs, next_obs)
            combined_actions = torch.cat([actions, next_actions], dim=0)
            target_q = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=combined_obs,
                actions=combined_actions,
                training=True,
            )
            batch_size = actions.shape[0]
            all_next_q = target_q[batch_size:]
            if agg_q == "min":
                q_next = self._min_twin_q(all_next_q)
            else:
                self._require_twin_q(all_next_q)
                q_next = all_next_q.mean(dim=-1, keepdim=True)
            q_next = q_next - self._entropy_alpha() * next_log_prob

            reward_target = self._discounted_chunk_rewards(rewards)
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            discount = float(self.cfg.algorithm.gamma) ** reward_horizon
            if bootstrap_type == "always":
                target_q_values = reward_target + discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        all_data_q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=combined_obs,
            actions=combined_actions,
            training=True,
        )
        data_q = all_data_q[: actions.shape[0]]
        target_q_values = target_q_values.to(dtype=data_q.dtype)
        critic_loss = F.mse_loss(data_q, target_q_values.expand_as(data_q))
        return critic_loss, {
            "q_data": data_q.mean().item(),
            "q_target": target_q_values.mean().item(),
            "next_log_pi": next_log_prob.mean().item(),
            "alpha": self.entropy_temp.alpha,
        }

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        if getattr(self, "qf_optimizer", None) is not None:
            self.qf_optimizer.zero_grad(set_to_none=True)

        actor_agg_q = self.cfg.algorithm.get("actor_agg_q", "min")
        if actor_agg_q not in {"min", "mean", "q1"}:
            raise ValueError(f"Unsupported WarpSAC actor aggregation: {actor_agg_q!r}")
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        actor_obs = self._concat_obs(curr_obs, next_obs)
        actions_all, log_prob_all, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=actor_obs,
            apply_reference_dropout=reference_dropout_prob > 0.0,
            reference_dropout_prob=reference_dropout_prob,
            training=True,
        )
        batch_size = batch["actions"].shape[0]
        pi = actions_all[:batch_size]
        log_pi = self._log_prob_sum(log_prob_all[:batch_size])

        all_qf_pi = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=pi,
            detach_encoder=True,
            training=False,
        )
        if actor_agg_q == "min":
            qf_pi = self._min_twin_q(all_qf_pi)
        elif actor_agg_q == "q1":
            qf_pi = self._q1(all_qf_pi)
        else:
            self._require_twin_q(all_qf_pi)
            qf_pi = all_qf_pi.mean(dim=-1, keepdim=True)

        bc_loss, bc_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=self._ref_chunk(curr_obs),
            intervene_flags=batch.get("intervene_flags"),
        )
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        alpha = self._entropy_alpha()
        actor_loss = (
            -q_weight * qf_pi.mean() + alpha * log_pi.mean() + bc_weight * bc_loss
        )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(all_qf_pi.shape[-1])
        }
        metrics.update(bc_metrics)
        metrics.update(weight_metrics)
        metrics.update(
            {
                "q_pi": qf_pi.mean().item(),
                "entropy": -log_pi.mean().item(),
                "alpha": self.entropy_temp.alpha,
                "weighted_q": (q_weight * qf_pi.mean()).detach().item(),
                "weighted_bc": (bc_weight * bc_loss).detach().item(),
                "weighted_entropy": (alpha * log_pi.mean()).detach().item(),
                "reference_dropout_prob": reference_dropout_prob,
            }
        )
        return actor_loss, -log_pi.mean(), metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        with torch.no_grad():
            _, log_prob, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=batch["curr_obs"],
                apply_reference_dropout=reference_dropout_prob > 0.0,
                reference_dropout_prob=reference_dropout_prob,
                training=False,
            )
            log_pi = self._log_prob_sum(log_prob)

        alpha = self.entropy_temp.compute_alpha()
        return -alpha * (log_pi.mean() + self.target_entropy)
