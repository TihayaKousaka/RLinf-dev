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

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class RLTSACLossMixin:
    """RLT actor/critic losses on top of RLinf SAC infrastructure."""

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _chunk_shape(self) -> tuple[int, int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        return chunk_len, action_dim

    def _algorithm_mode(self) -> str:
        loss_type = str(self.cfg.algorithm.get("loss_type", "rlt_sac")).lower()
        if loss_type == "rlt_td3":
            return "td3"
        if loss_type == "rlt_sac":
            return "sac"
        raise NotImplementedError(f"{loss_type=} is not supported by RLT worker.")

    def get_rollout_sync_version(self) -> int:
        """Expose learner update count so TD3 rollout warmup tracks updates."""
        if self._algorithm_mode() != "td3":
            return super().get_rollout_sync_version()
        return int(self.update_step)

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._flatten_chunk(obs["ref_chunk"])

    def _aggregate_q(self, all_q_values: torch.Tensor, agg_q: str) -> torch.Tensor:
        if agg_q == "min":
            q_values, _ = torch.min(all_q_values, dim=-1, keepdim=True)
            return q_values
        if agg_q == "mean":
            return torch.mean(all_q_values, dim=-1, keepdim=True)
        raise NotImplementedError(f"{agg_q=} is not supported for RLT SAC.")

    def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1)
        rewards = rewards.to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=rewards.device),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        action_chunk = self._flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
        bc_ref_chunk = self._flatten_chunk(ref_chunk).reshape(
            ref_chunk.shape[0], -1, action_dim
        )[:, :chunk_len]

        if intervene_flags is None:
            human_mask = torch.zeros(
                pi_chunk.shape[:2], dtype=torch.bool, device=pi_chunk.device
            )
        else:
            human_mask = (
                self._flatten_chunk(intervene_flags)
                .to(device=pi_chunk.device)
                .bool()
                .reshape(-1, chunk_len, action_dim)
                .any(dim=-1)
            )

        bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        bc_loss = torch.mean(bc_error)

        policy_mask = ~human_mask
        ref_error = torch.mean(torch.square(pi_chunk - bc_ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = torch.mean(human_mask.to(torch.float32)).item()
        metrics = {
            "bc_loss": bc_loss.detach().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
        }
        return bc_loss, metrics

    def _chunk_delta_loss(
        self,
        pi: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        target_chunk = self._flatten_chunk(target).reshape(-1, chunk_len, action_dim)
        if chunk_len <= 1:
            return torch.zeros((), device=pi.device, dtype=pi.dtype)
        pred_delta = pi_chunk[:, 1:, :] - pi_chunk[:, :-1, :]
        target_delta = target_chunk[:, 1:, :] - target_chunk[:, :-1, :]
        return F.mse_loss(pred_delta, target_delta)

    def _not_done(self, terminations: torch.Tensor) -> torch.Tensor:
        return ~terminations.reshape(terminations.shape[0], -1).bool().any(
            dim=-1,
            keepdim=True,
        )

    def _bootstrap_target(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        q_next: torch.Tensor,
    ) -> torch.Tensor:
        reward_target = self._discounted_chunk_rewards(rewards)
        reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
        bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "always":
            return reward_target + bootstrap_discount * q_next
        if bootstrap_type == "standard":
            return (
                reward_target
                + self._not_done(terminations) * bootstrap_discount * q_next
            )
        raise NotImplementedError(f"{bootstrap_type=} is not supported!")

    def _actor_loss_weights(self) -> tuple[float, float, float, dict[str, float]]:
        """Resolve TD3 BC/Q/delta weights with local warmup and ramp support."""
        td3_bc_cfg = self.cfg.algorithm.get("td3_bc", {})
        loss_warmup_updates = int(
            td3_bc_cfg.get(
                "actor_loss_warmup_updates",
                self.cfg.algorithm.get("actor_loss_warmup_updates", 0),
            )
        )
        ramp_updates = int(
            td3_bc_cfg.get(
                "actor_loss_ramp_updates",
                self.cfg.algorithm.get("actor_loss_ramp_updates", 0),
            )
        )
        in_warmup = int(self.update_step) < loss_warmup_updates
        warmup_bc_weight = float(
            td3_bc_cfg.get(
                "warmup_bc_weight",
                self.cfg.algorithm.get(
                    "warmup_bc_weight",
                    self.cfg.algorithm.get("bc_weight", 1.0),
                ),
            )
        )
        warmup_q_weight = float(
            td3_bc_cfg.get(
                "warmup_q_weight",
                self.cfg.algorithm.get(
                    "warmup_q_weight",
                    self.cfg.algorithm.get("q_weight", 1.0),
                ),
            )
        )
        online_bc_weight = float(
            td3_bc_cfg.get(
                "online_bc_weight",
                self.cfg.algorithm.get(
                    "online_bc_weight",
                    self.cfg.algorithm.get("bc_weight", 1.0),
                ),
            )
        )
        online_q_weight = float(
            td3_bc_cfg.get(
                "online_q_weight",
                self.cfg.algorithm.get(
                    "online_q_weight",
                    self.cfg.algorithm.get("q_weight", 1.0),
                ),
            )
        )
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        elif ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(int(self.update_step) - loss_warmup_updates + 1)
                    / float(ramp_updates),
                ),
            )
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        else:
            bc_weight = online_bc_weight
            q_weight = online_q_weight
            ramp_progress = 1.0

        delta_weight = float(
            td3_bc_cfg.get(
                "delta_weight",
                self.cfg.algorithm.get("delta_weight", 0.0),
            )
        )
        metrics = {
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "delta_weight": delta_weight,
            "actor_loss_in_warmup": float(in_warmup),
            "actor_loss_ramp_progress": ramp_progress,
        }
        return bc_weight, q_weight, delta_weight, metrics

    def _ready_for_online(self) -> bool:
        return int(self.update_step) >= int(
            self.cfg.algorithm.get("warmup_post_collect_updates", 0)
        )

    def _td3_target_actions(self, next_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_actions, _, _ = self.target_model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
                deterministic=True,
                apply_action_noise=False,
            )
            target_noise_sigma = float(
                self.cfg.algorithm.get("target_noise_sigma", 0.0)
            )
            if target_noise_sigma > 0:
                target_noise_clip = float(
                    self.cfg.algorithm.get("target_noise_clip", 0.5)
                )
                noise = torch.randn_like(next_actions) * target_noise_sigma
                noise = noise.clamp(-target_noise_clip, target_noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            return next_actions

    def _set_q_head_requires_grad(self, requires_grad: bool) -> None:
        module = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(module, "set_q_head_requires_grad"):
            module.set_q_head_requires_grad(requires_grad)
            return
        for name, param in self.model.named_parameters():
            if "q_head" in name:
                param.requires_grad_(requires_grad)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        if self._algorithm_mode() == "td3":
            return self.forward_td3_critic(batch)

        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminations = batch["terminations"].to(self.torch_dtype)
        not_done = ~terminations.reshape(terminations.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            next_actions, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
            )

            if not use_crossq:
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_actions,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )
                q_next = self._aggregate_q(all_qf_next_target, agg_q)
            else:
                _, all_qf_next = self.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_actions,
                )
                q_next = self._aggregate_q(all_qf_next.detach(), agg_q)

            reward_target = self._discounted_chunk_rewards(rewards)
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
        else:
            all_data_q_values, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        if self._algorithm_mode() == "td3":
            return self.forward_td3_actor(batch)

        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        agg_q = self.cfg.algorithm.get(
            "actor_agg_q", self.cfg.algorithm.get("agg_q", "min")
        )

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        if not use_crossq:
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        qf_pi = self._aggregate_q(all_qf_pi, agg_q)
        metrics["q_pi"] = qf_pi.mean().item()

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, rlt_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
        )
        metrics.update(rlt_metrics)

        entropy = -log_pi.mean()
        q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
        bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["q_weight"] = q_weight
        metrics["bc_weight"] = bc_weight
        metrics["reference_dropout_prob"] = reference_dropout_prob

        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        del batch
        return self.entropy_temp.compute_alpha() * 0.0

    def forward_td3_critic(self, batch):
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminations = batch["terminations"].to(self.torch_dtype)

        with torch.no_grad():
            next_actions = self._td3_target_actions(next_obs)
            all_qf_next_target = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_actions,
            )
            if self.critic_subsample_size > 0:
                sample_idx = torch.randint(
                    0,
                    all_qf_next_target.shape[-1],
                    (self.critic_subsample_size,),
                    generator=self.critic_sample_generator,
                    device=self.device,
                )
                all_qf_next_target = all_qf_next_target.index_select(
                    dim=-1,
                    index=sample_idx,
                )
            q_next = self._aggregate_q(all_qf_next_target, agg_q)
            target_q_values = self._bootstrap_target(rewards, terminations, q_next)

        all_data_q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
        )
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values,
            target_q_values.expand_as(all_data_q_values),
        )
        metrics = {
            "q_data": all_data_q_values.mean().item(),
            "q_target": target_q_values.mean().item(),
            "ready_for_online": float(self._ready_for_online()),
        }
        return critic_loss, metrics

    def forward_td3_actor(self, batch):
        agg_q = self.cfg.algorithm.get(
            "actor_agg_q",
            self.cfg.algorithm.get("agg_q", "min"),
        )
        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        self._set_q_head_requires_grad(False)
        try:
            pi, log_pi, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=curr_obs,
                deterministic=False,
                apply_action_noise=True,
                apply_reference_dropout=True,
                reference_dropout_prob=reference_dropout_prob,
            )
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                detach_encoder=True,
            )
        finally:
            self._set_q_head_requires_grad(True)
        qf_pi = self._aggregate_q(all_qf_pi, agg_q)

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
        )
        delta_loss = self._chunk_delta_loss(pi, ref_chunk)
        bc_weight, q_weight, delta_weight, weight_metrics = self._actor_loss_weights()
        actor_loss = (
            -q_weight * qf_pi.mean()
            + bc_weight * bc_loss
            + delta_weight * delta_loss
        )
        metrics.update(
            {
                f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
                for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
            }
        )
        metrics.update(weight_metrics)
        metrics["q_pi"] = qf_pi.mean().item()
        metrics["delta_loss"] = delta_loss.detach().item()
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["weighted_delta"] = (delta_weight * delta_loss).detach().item()
        metrics["reference_dropout_prob"] = reference_dropout_prob
        metrics["ready_for_online"] = float(self._ready_for_online())
        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics


class RLTSACFSDPPolicy(RLTSACLossMixin, EmbodiedSACFSDPPolicy):
    """Synchronous RLT worker with optional TD3 warmup scheduling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None

    def _is_td3_mode(self) -> bool:
        return self.cfg.algorithm.get("loss_type", "") == "rlt_td3"

    def setup_sac_components(self):
        """Initialize replay components and let TD3 warmup own sample readiness."""
        super().setup_sac_components()
        if self._is_td3_mode():
            self.buffer_dataset.min_replay_buffer_size = 1

    @staticmethod
    def _trajectory_transition_count(traj: Trajectory) -> int:
        if traj.actions is None:
            return 0
        return int(traj.actions.shape[0] * traj.actions.shape[1])

    @staticmethod
    def _trajectory_completed_episodes(traj: Trajectory) -> int:
        dones = traj.dones
        if dones is None:
            return 0
        return int(dones.reshape(dones.shape[0], dones.shape[1], -1).any(dim=-1).sum())

    async def recv_rollout_trajectories(self, input_channel):
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        if self._is_td3_mode():
            added = sum(self._trajectory_transition_count(traj) for traj in recv_list)
            completed = sum(
                self._trajectory_completed_episodes(traj) for traj in recv_list
            )
            self.transitions_since_train += added
            self.episodes_since_train += completed
            self.total_transitions_added += added
            self.total_episodes_added += completed

    def _global_td3_counters(self) -> dict[str, float]:
        summed = all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )
        minimums = all_reduce_dict(
            {
                "min_replay_size": float(self.replay_buffer.total_samples),
                "min_demo_size": float(
                    0 if self.demo_buffer is None else self.demo_buffer.total_samples
                ),
            },
            op=torch.distributed.ReduceOp.MIN,
        )
        summed.update(minimums)
        return summed

    def _td3_updates_to_run(self) -> tuple[int, dict[str, float]]:
        replay_cfg = self.cfg.algorithm.replay_buffer
        min_buffer_size = int(
            replay_cfg.get(
                "min_buffer_size",
                self.cfg.algorithm.get("warmup_min_size", 1),
            )
        )
        counters = self._global_td3_counters()
        buffer_ready = counters["min_replay_size"] >= min_buffer_size
        warmup_required_updates = int(
            self.cfg.algorithm.get("warmup_post_collect_updates", 0)
        )
        if buffer_ready and self._warmup_ready_total_transitions is None:
            self._warmup_ready_total_transitions = int(
                counters["total_transitions_added"]
            )
            self._warmup_ready_total_episodes = int(counters["total_episodes_added"])

        updates_to_run = 0
        skip_reason = 0
        desired_total_updates = 0
        if not buffer_ready:
            skip_reason = 1
        else:
            train_every_transitions = int(
                self.cfg.algorithm.get("train_every_transitions", 0)
            )
            train_every_episodes = int(self.cfg.algorithm.get("train_every_episodes", 0))
            update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
            online_transitions = max(
                int(counters["total_transitions_added"])
                - int(self._warmup_ready_total_transitions or 0),
                0,
            )
            online_episodes = max(
                int(counters["total_episodes_added"])
                - int(self._warmup_ready_total_episodes or 0),
                0,
            )
            if train_every_transitions <= 0 and train_every_episodes <= 0:
                online_cycles = online_transitions
            else:
                transition_cycles = (
                    online_transitions // train_every_transitions
                    if train_every_transitions > 0
                    else 0
                )
                episode_cycles = (
                    online_episodes // train_every_episodes
                    if train_every_episodes > 0
                    else 0
                )
                online_cycles = max(transition_cycles, episode_cycles)
            desired_total_updates = (
                warmup_required_updates + online_cycles * update_epoch
            )
            pending_updates = max(desired_total_updates - int(self.update_step), 0)
            updates_to_run = pending_updates
            max_updates = int(self.cfg.algorithm.get("max_updates_per_train_step", 0))
            if max_updates > 0:
                updates_to_run = min(updates_to_run, max_updates)
            if updates_to_run <= 0:
                skip_reason = 2

        metrics = {
            "rlt_stage2/update_step": float(self.update_step),
            "rlt_stage2/ready_for_online": float(
                int(self.update_step) >= warmup_required_updates
            ),
            "rlt_stage2/warmup_required_updates": float(warmup_required_updates),
            "rlt_stage2/desired_total_updates": float(desired_total_updates),
            "rlt_stage2/updates_to_run": float(updates_to_run),
            "rlt_stage2/skip_reason": float(skip_reason),
            "rlt_stage2/global_min_replay_size": float(counters["min_replay_size"]),
            "rlt_stage2/min_replay_buffer_size": float(min_buffer_size),
            "rlt_stage2/global_transitions_since_train": float(
                counters["transitions_since_train"]
            ),
            "rlt_stage2/global_total_transitions_added": float(
                counters["total_transitions_added"]
            ),
        }
        return updates_to_run, metrics

    def run_training(self):
        if not self._is_td3_mode():
            return super().run_training()

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        updates_to_run, schedule_metrics = self._td3_updates_to_run()
        if updates_to_run <= 0:
            mean_metric_dict = self.process_train_metrics(schedule_metrics)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            return mean_metric_dict

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        for _ in range(updates_to_run):
            metrics_data = self.update_one_epoch(train_actor=True)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        append_to_dict(metrics, schedule_metrics)
        mean_metric_dict = self.process_train_metrics(metrics)
        self.transitions_since_train = 0
        self.episodes_since_train = 0

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict


class AsyncRLTSACFSDPPolicy(RLTSACLossMixin, AsyncEmbodiedSACFSDPPolicy):
    pass
