# Copyright 2025 The RLinf Authors.
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

"""FSDP actor worker for RLT Stage 2 TD3 training."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.utils import collect_param_names_need_sync

from ...models.embodiment.rlt_stage2.components import actor_loss, critic_loss
from ...models.embodiment.rlt_stage2.schedule import (
    RLTStage2TrainingScheduler,
    RLTTrainingPlan,
    resolve_actor_loss_weights,
    write_status_json,
)
from ...models.embodiment.rlt_stage2.trajectory_adapter import (
    RLTStage2TrajectoryReplayAdapter,
)
from .embodied_offpolicy_worker import EmbodiedOffPolicyFSDPActor


class RLTStage2FSDPPolicyWorker(EmbodiedOffPolicyFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.trajectory_adapter: RLTStage2TrajectoryReplayAdapter | None = None
        self.qf_optimizer = None
        self.qf_lr_scheduler = None
        self.training_scheduler = RLTStage2TrainingScheduler()
        self.actor_only_train_model = bool(
            cfg.algorithm.get("actor_only_train_model", True)
        )
        self._rollout_sync_key_count = 0
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._last_logged_status_phase: str | None = None
        self.buffer_dataset = None
        self.buffer_dataloader = None
        self.buffer_dataloader_iter = None

    def init_worker(self):
        self.setup_model_and_optimizer()
        self._init_replay_buffer()
        self.trajectory_adapter = RLTStage2TrajectoryReplayAdapter(
            cfg=self.cfg,
        )
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> torch.nn.Module:
        from rlinf.models import get_model

        model_cfg = self.cfg.actor.model
        if bool(
            self.actor_only_train_model
        ) and model_cfg.get("rlt_stage2", None) is not None:
            from copy import deepcopy
            from omegaconf import open_dict

            model_cfg = deepcopy(model_cfg)
            with open_dict(model_cfg):
                model_cfg.rlt_stage2.load_feature_backbones = False
                model_cfg.rlt_stage2.load_rl_token_model = False

        model = get_model(model_cfg)
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path, map_location="cpu")
            model.load_state_dict(model_dict)
        return model

    def setup_model_and_optimizer(self) -> None:
        module = self.model_provider_func()
        self.param_names_need_sync = collect_param_names_need_sync(module)
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        actor_filters = {"critic": ["critic."]}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=actor_filters,
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        grad_scaler_cfg = self.cfg.actor.fsdp_config.grad_scaler
        kwargs = {}
        for key in ["init_scale", "growth_interval"]:
            value = grad_scaler_cfg.get(key, None)
            if value is not None:
                kwargs[key] = value
        self.grad_scaler = self.build_grad_scaler(
            grad_scaler_cfg.get("enabled", False),
            **kwargs,
        )

    def _init_replay_buffer(self) -> None:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        capacity = int(
            replay_cfg.get("capacity", stage2_cfg.get("buffer_capacity", 200000))
        )
        demo_cfg = self.cfg.algorithm.get("demo_buffer", None)
        demo_capacity = (
            None if demo_cfg is None else int(demo_cfg.get("capacity", capacity))
        )
        self.init_trajectory_replay_buffers(
            replay_cfg=replay_cfg,
            replay_seed=int(self.cfg.actor.get("seed", 1234)) + self._rank,
            replay_capacity=capacity,
            demo_cfg=demo_cfg,
            demo_seed=int(self.cfg.actor.get("seed", 1234)) + self._rank + 17,
            demo_capacity=demo_capacity,
        )
        if self.replay_buffer is None:
            raise RuntimeError("RLT Stage2 replay buffer failed to initialize.")

        batch_size = int(self.cfg.actor.global_batch_size // self._world_size)
        if demo_cfg is not None and batch_size % 2 != 0:
            raise ValueError(
                "RLT Stage2 demo replay sampling requires an even per-rank "
                f"global batch size, got {batch_size}."
            )
        replay_sample_size = batch_size if demo_cfg is None else batch_size // 2
        replay_window_size = int(replay_cfg.get("sample_window_size", capacity))
        if 0 < replay_window_size < replay_sample_size:
            raise ValueError(
                "RLT Stage2 replay sample_window_size must be 0 or at least the "
                f"per-update replay sample count, got {replay_window_size} < "
                f"{replay_sample_size}."
            )
        if demo_cfg is not None:
            demo_sample_size = batch_size // 2
            demo_window_size = int(demo_cfg.get("sample_window_size", demo_capacity))
            if 0 < demo_window_size < demo_sample_size:
                raise ValueError(
                    "RLT Stage2 demo sample_window_size must be 0 or at least the "
                    f"per-update demo sample count, got {demo_window_size} < "
                    f"{demo_sample_size}."
                )
        min_replay_buffer_size, min_demo_buffer_size = (
            self._replay_dataset_min_sizes(batch_size)
        )

        buffer_dataset_cls = (
            PreloadReplayBufferDataset
            if bool(replay_cfg.get("enable_preload", False))
            else ReplayBufferDataset
        )
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=batch_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=int(replay_cfg.get("prefetch_size", 10)),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

    def _replay_dataset_min_sizes(self, batch_size: int) -> tuple[int, int]:
        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        demo_cfg = self.cfg.algorithm.get("demo_buffer", None)
        replay_sample_size = int(batch_size) if demo_cfg is None else int(batch_size) // 2
        min_replay_buffer_size = max(
            int(
                replay_cfg.get(
                    "min_buffer_size",
                    self.cfg.algorithm.get("warmup_min_size", 1),
                )
            ),
            replay_sample_size,
        )
        min_demo_buffer_size = 0
        if demo_cfg is not None:
            min_demo_buffer_size = max(
                int(demo_cfg.get("min_buffer_size", 0)),
                int(batch_size) // 2,
            )
        return min_replay_buffer_size, min_demo_buffer_size

    def soft_update_target_model(self, tau: float | None = None) -> None:
        if tau is None:
            tau = float(self.cfg.algorithm.tau)
        self.model.update_target_networks(float(tau))

    def get_rollout_state_dict(self) -> dict:
        state_dict = self.model.filter_rollout_state_dict(
            self.get_model_state_dict(cpu_offload=False, full_state_dict=False)
        )
        self._rollout_sync_key_count = len(state_dict)
        return state_dict

    def get_rollout_sync_param_names(self, state_dict: dict) -> list[str]:
        return list(state_dict.keys())

    def get_rollout_sync_version(self) -> int:
        # Rollout uses this version as the number of completed TD3 updates.
        # Do not send runner global_step here, otherwise warmup/intervention
        # gates open before any actor/critic update has actually happened.
        return int(self.update_step)

    def add_rollout_trajectories(self, trajectories: list[Trajectory]) -> None:
        if self.trajectory_adapter is None or self.replay_buffer is None:
            raise RuntimeError("RLT Stage2 replay path is not initialized.")

        replay_trajectories: list[Trajectory] = []
        completed_episodes = 0
        for trajectory in trajectories:
            new_trajectories, new_episodes = (
                self.trajectory_adapter.build_replay_trajectories(trajectory)
            )
            replay_trajectories.extend(new_trajectories)
            completed_episodes += new_episodes

        if replay_trajectories:
            self.replay_buffer.add_trajectories(replay_trajectories)

        added = len(replay_trajectories)
        self.transitions_since_train += added
        self.episodes_since_train += completed_episodes
        self.total_transitions_added += added
        self.total_episodes_added += completed_episodes

    def _global_rollout_counters(self) -> dict[str, float]:
        return all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )

    def _global_min_replay_size(self) -> int:
        replay_size = (
            0 if self.replay_buffer is None else self.replay_buffer.total_samples
        )
        reduced = all_reduce_dict(
            {"min_replay_size": float(replay_size)},
            op=torch.distributed.ReduceOp.MIN,
        )
        return int(reduced["min_replay_size"])

    def _global_min_demo_size(self) -> int:
        demo_size = 0 if self.demo_buffer is None else self.demo_buffer.total_samples
        reduced = all_reduce_dict(
            {"min_demo_size": float(demo_size)},
            op=torch.distributed.ReduceOp.MIN,
        )
        return int(reduced["min_demo_size"])

    @staticmethod
    def _unwrap_rlt_forward_inputs(
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        forward_inputs = batch.get("forward_inputs", None)
        if not isinstance(forward_inputs, dict):
            raise RuntimeError(
                "RLT Stage2 replay sample is missing forward_inputs. "
                "Only RLT transition trajectories produced by "
                "RLTStage2TrajectoryReplayAdapter are supported."
            )
        return forward_inputs

    def _next_rlt_replay_batch(
        self,
        expected_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if self.buffer_dataloader_iter is None:
            raise RuntimeError("RLT Stage2 replay DataLoader is not initialized.")
        batch = self._unwrap_rlt_forward_inputs(next(self.buffer_dataloader_iter))
        first_tensor = next(
            value for value in batch.values() if isinstance(value, torch.Tensor)
        )
        actual_batch_size = int(first_tensor.shape[0])
        if actual_batch_size != int(expected_batch_size):
            raise RuntimeError(
                "RLT Stage2 replay DataLoader returned an unexpected batch size: "
                f"expected {expected_batch_size}, got {actual_batch_size}. "
                "Check replay readiness and actor.global_batch_size."
            )
        return put_tensor_device(batch, self.device)

    def _append_training_plan_metrics(
        self,
        metrics: dict[str, Any],
        *,
        plan: RLTTrainingPlan,
        global_counters: dict[str, float],
        global_min_replay_size: int,
        global_min_demo_size: int,
        should_train: bool,
        skip_reason: int,
        critic_updates_run: int = 0,
        actor_updates_run: int = 0,
    ) -> None:
        append_to_dict(
            metrics,
            self.training_scheduler.metrics(
                plan=plan,
                update_step=self.update_step,
                global_counters=global_counters,
                global_min_replay_size=global_min_replay_size,
                global_min_demo_size=global_min_demo_size,
                should_train=should_train,
                skip_reason=skip_reason,
                critic_updates_run=critic_updates_run,
                actor_updates_run=actor_updates_run,
            ),
        )
        self._emit_training_status(
            plan=plan,
            global_min_replay_size=global_min_replay_size,
            should_train=should_train,
            skip_reason=skip_reason,
            critic_updates_run=critic_updates_run,
            actor_updates_run=actor_updates_run,
            global_total_transitions_added=int(
                global_counters["total_transitions_added"]
            ),
            global_total_episodes_added=int(global_counters["total_episodes_added"]),
        )

    def _emit_training_status(
        self,
        *,
        plan: RLTTrainingPlan,
        global_min_replay_size: int,
        should_train: bool,
        skip_reason: int,
        critic_updates_run: int,
        actor_updates_run: int,
        global_total_transitions_added: int,
        global_total_episodes_added: int,
    ) -> None:
        phase = self.training_scheduler.status_phase(plan, self.update_step)
        if self._rank == 0 and phase != self._last_logged_status_phase:
            ready_for_online = self.training_scheduler.ready_for_online(
                plan,
                self.update_step,
            )
            self.log_info(
                "[RLT_STATUS][actor] "
                f"phase={phase} ready={int(ready_for_online)} "
                f"buffer_ready={int(plan.readiness.buffer_ready)} "
                f"replay={global_min_replay_size}/{plan.readiness.warmup_min_size} "
                f"update={self.update_step}/{plan.schedule.warmup_required_updates} "
                f"pending={self.training_scheduler.pending_update_budget}"
            )
            self._last_logged_status_phase = phase

        status_dir = os.path.join(self.cfg.runner.logger.log_path, "status")
        write_status_json(
            os.path.join(status_dir, f"rlt_actor_status_rank{self._rank}.json"),
            self.training_scheduler.status_payload(
                plan=plan,
                rank=self._rank,
                update_step=self.update_step,
                global_min_replay_size=global_min_replay_size,
                should_train=should_train,
                skip_reason=skip_reason,
                critic_updates_run=critic_updates_run,
                actor_updates_run=actor_updates_run,
                global_total_transitions_added=global_total_transitions_added,
                global_total_episodes_added=global_total_episodes_added,
            ),
        )

    def _append_replay_stats(self, metrics: dict[str, Any]) -> None:
        if self.replay_buffer is None:
            return
        stats = self.replay_buffer.get_stats()
        append_to_dict(
            metrics,
            {"replay_buffer/size": self.replay_buffer.total_samples},
        )
        for key, value in stats.items():
            append_to_dict(metrics, {f"replay_buffer/{key}": value})
        rlt_stats = self._rlt_replay_stats(self.replay_buffer)
        for key, value in rlt_stats.items():
            append_to_dict(metrics, {f"replay_buffer/{key}": value})
        if self.demo_buffer is None:
            return
        demo_stats = self.demo_buffer.get_stats()
        append_to_dict(metrics, {"demo_buffer/size": self.demo_buffer.total_samples})
        for key, value in demo_stats.items():
            append_to_dict(metrics, {f"demo_buffer/{key}": value})
        demo_rlt_stats = self._rlt_replay_stats(self.demo_buffer)
        for key, value in demo_rlt_stats.items():
            append_to_dict(metrics, {f"demo_buffer/{key}": value})

    @staticmethod
    def _rlt_replay_stats(replay_buffer: TrajectoryReplayBuffer) -> dict[str, float]:
        cache = replay_buffer._flat_trajectory_cache
        if cache is None:
            return {"intervention_rate": 0.0, "human_chunk_rate": 0.0}
        buffer = cache.get_buffer()
        if not isinstance(buffer, dict):
            return {"intervention_rate": 0.0, "human_chunk_rate": 0.0}
        forward_inputs = buffer.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return {"intervention_rate": 0.0, "human_chunk_rate": 0.0}
        intervention = forward_inputs.get("intervention")
        source_chunk = forward_inputs.get("source_chunk")
        intervention_rate = (
            float(intervention.detach().float().mean().item())
            if isinstance(intervention, torch.Tensor) and intervention.numel() > 0
            else 0.0
        )
        human_chunk_rate = 0.0
        if isinstance(source_chunk, torch.Tensor) and source_chunk.numel() > 0:
            source_chunk = source_chunk.detach().to(torch.uint8)
            human_or_mixed = torch.logical_or(source_chunk == 2, source_chunk == 3)
            human_chunk_rate = float(human_or_mixed.any(dim=-1).float().mean().item())
        return {
            "intervention_rate": intervention_rate,
            "human_chunk_rate": human_chunk_rate,
        }

    @Worker.timer("run_training")
    def run_training(self):
        if self.replay_buffer is None:
            return {}

        metrics: dict[str, Any] = {}
        global_counters = self._global_rollout_counters()
        global_min_replay_size = self._global_min_replay_size()
        global_min_demo_size = self._global_min_demo_size()
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        min_replay_buffer_size, min_demo_buffer_size = (
            self._replay_dataset_min_sizes(global_batch_size_per_rank)
        )
        plan = self.training_scheduler.plan(
            self.cfg,
            update_step=self.update_step,
            has_demo_buffer=self.demo_buffer is not None,
            global_counters=global_counters,
            global_min_replay_size=global_min_replay_size,
            global_min_demo_size=global_min_demo_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
        )
        schedule = plan.schedule

        if schedule.skip_reason != 0:
            self._append_replay_stats(metrics)
            self._append_training_plan_metrics(
                metrics,
                plan=plan,
                global_counters=global_counters,
                global_min_replay_size=global_min_replay_size,
                global_min_demo_size=global_min_demo_size,
                should_train=False,
                skip_reason=schedule.skip_reason,
            )
            return self.average_metrics(metrics)

        self.ensure_training_state_loaded()

        self.model.train()
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(schedule.updates_to_run):
            batch_dict = self._next_rlt_replay_batch(
                global_batch_size_per_rank,
            )
            micro_batches = self.prepare_micro_batches(
                batch_dict,
                global_batch_size_per_rank=global_batch_size_per_rank,
            )
            epoch_metrics = self._update_one_epoch(
                micro_batches,
                global_min_replay_size=global_min_replay_size,
            )
            append_to_dict(metrics, epoch_metrics)
            self.update_step += 1
            critic_updates_run += 1
            if "rlt_stage2/actor_loss" in epoch_metrics:
                actor_updates_run += 1
        self.training_scheduler.finish_updates(critic_updates_run)

        self._append_replay_stats(metrics)
        self._append_training_plan_metrics(
            metrics,
            plan=plan,
            global_counters=global_counters,
            global_min_replay_size=global_min_replay_size,
            global_min_demo_size=global_min_demo_size,
            should_train=True,
            skip_reason=0,
            critic_updates_run=critic_updates_run,
            actor_updates_run=actor_updates_run,
        )
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        return self.average_metrics(metrics)

    def _update_one_epoch(
        self,
        micro_batches: list[dict[str, torch.Tensor]],
        *,
        global_min_replay_size: int,
    ) -> dict[str, float]:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        critic_losses = []
        q1_values = []
        q2_values = []

        self.optimizer.zero_grad(set_to_none=True)
        self.qf_optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(micro_batches):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == len(micro_batches),
            )
            with backward_ctx:
                with self.amp_context:
                    td_target = self.model.compute_td_target_batch(
                        rewards=batch["rewards"].to(torch.float32),
                        dones=batch["dones"].to(torch.float32),
                        next_x=batch["next_x"].to(torch.float32),
                        next_a_tilde=batch["next_a_tilde"].to(torch.float32),
                    )
                    q1, q2 = self.model.critic_forward(
                        batch["x"].to(torch.float32),
                        batch["a"].to(torch.float32),
                    )
                    loss = (
                        critic_loss(q1, q2, td_target) / self.gradient_accumulation
                    )
                self.grad_scaler.scale(loss).backward()
            critic_losses.append(
                loss.detach().float().item() * self.gradient_accumulation
            )
            q1_values.append(q1.detach().float().mean().item())
            q2_values.append(q2.detach().float().mean().item())

        self.grad_scaler.unscale_(self.qf_optimizer)
        critic_grad_norm = self._strategy.clip_grad_norm_(self.model)
        self.grad_scaler.step(self.qf_optimizer)
        self.grad_scaler.update()
        self.qf_lr_scheduler.step()
        self.qf_optimizer.zero_grad(set_to_none=True)

        metrics = {
            "rlt_stage2/critic_loss": float(np.mean(critic_losses)),
            "critic/q1_mean": float(np.mean(q1_values)),
            "critic/q2_mean": float(np.mean(q2_values)),
            "critic/grad_norm": float(critic_grad_norm),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
        }

        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        warmup_min_size = int(
            replay_cfg.get(
                "min_buffer_size",
                self.cfg.algorithm.get("warmup_min_size", 1),
            )
        )
        replay_ready = global_min_replay_size >= warmup_min_size
        loss_weights = resolve_actor_loss_weights(self.cfg, self.update_step)
        update_actor = (
            replay_ready
            and (self.update_step + 1) % int(self.cfg.algorithm.critic_actor_ratio) == 0
        )
        if update_actor:
            actor_losses = []
            actor_q_values = []
            actor_action_ref_abs = []
            actor_bc_losses = []
            actor_bc_ref_losses = []
            actor_bc_human_losses = []
            actor_human_mask_ratios = []
            self.optimizer.zero_grad()
            self.model.set_online_critic_requires_grad(False)
            for idx, batch in enumerate(micro_batches):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == len(micro_batches),
                )
                with backward_ctx:
                    with self.amp_context:
                        actions = self.model.actor_forward(
                            batch["x"].to(torch.float32),
                            batch["a_tilde"].to(torch.float32),
                            deterministic=False,
                            apply_ref_dropout=bool(
                                stage2_cfg.get("ref_action_dropout", 0.0) > 0.0
                            ),
                            apply_action_noise=True,
                        )
                        a_tilde_flat = batch["a_tilde"].to(torch.float32)
                        action_chunk_flat = batch["action_chunk"].to(torch.float32)
                        action_ref_delta = actions - a_tilde_flat
                        chunk_len = int(self.cfg.actor.model.num_action_chunks)
                        action_dim = int(self.cfg.actor.model.action_dim)
                        actions_chunk = actions.reshape(-1, chunk_len, action_dim)
                        a_tilde_chunk = a_tilde_flat.reshape(-1, chunk_len, action_dim)
                        executed_action_chunk = action_chunk_flat.reshape(
                            -1,
                            chunk_len,
                            action_dim,
                        )
                        q_value = self.model.critic_min(
                            batch["x"].to(torch.float32),
                            actions,
                        )
                        actor_total_loss, actor_loss_metrics = actor_loss(
                            q_value=q_value,
                            a=actions_chunk,
                            a_tilde=a_tilde_chunk,
                            action_chunk=executed_action_chunk,
                            source_chunk=batch["source_chunk"].to(torch.uint8),
                            bc_weight=loss_weights.bc_weight,
                            q_weight=loss_weights.q_weight,
                            delta_weight=loss_weights.delta_weight,
                        )
                        loss = actor_total_loss / self.gradient_accumulation
                    self.grad_scaler.scale(loss).backward()
                actor_losses.append(
                    loss.detach().float().item() * self.gradient_accumulation
                )
                actor_q_values.append(q_value.detach().float().mean().item())
                actor_action_ref_abs.append(
                    action_ref_delta.detach().float().abs().mean().item()
                )
                actor_bc_losses.append(
                    float(actor_loss_metrics["bc_loss"].float().item())
                )
                actor_bc_ref_losses.append(
                    float(actor_loss_metrics["bc_ref_loss"].float().item())
                )
                actor_bc_human_losses.append(
                    float(actor_loss_metrics["bc_human_loss"].float().item())
                )
                actor_human_mask_ratios.append(
                    float(actor_loss_metrics["human_mask_ratio"].float().item())
                )
            self.model.set_online_critic_requires_grad(True)

            self.grad_scaler.unscale_(self.optimizer)
            actor_grad_norm = self._strategy.clip_grad_norm_(self.model)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            metrics.update(
                {
                    "rlt_stage2/actor_loss": float(np.mean(actor_losses)),
                    "actor/q_mean": float(np.mean(actor_q_values)),
                    "actor/grad_norm": float(actor_grad_norm),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/action_ref_abs_mean": float(np.mean(actor_action_ref_abs)),
                    "actor/bc_weight": loss_weights.bc_weight,
                    "actor/q_weight": loss_weights.q_weight,
                    "actor/bc_loss": float(np.mean(actor_bc_losses)),
                    "actor/bc_ref_loss": float(np.mean(actor_bc_ref_losses)),
                    "actor/bc_human_loss": float(np.mean(actor_bc_human_losses)),
                    "actor/human_mask_ratio": float(np.mean(actor_human_mask_ratios)),
                }
            )
        else:
            metrics.update(
                {
                    "actor/bc_weight": loss_weights.bc_weight,
                    "actor/q_weight": loss_weights.q_weight,
                }
            )

        target_update_freq = int(self.cfg.algorithm.get("target_update_freq", 1))
        if target_update_freq <= 0:
            raise ValueError(
                f"algorithm.target_update_freq must be positive, got {target_update_freq}."
            )
        if (self.update_step + 1) % target_update_freq == 0:
            self.soft_update_target_model()
        return metrics

    def _stage2_component_dir(self, base_path: str) -> str:
        return os.path.join(base_path, "rlt_stage2_components")

    def save_checkpoint(self, save_base_path, step):
        self.ensure_checkpoint_state_loaded()
        self.save_model_checkpoint(
            save_path=save_base_path,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
        )

        stage2_save_path = self._stage2_component_dir(save_base_path)
        os.makedirs(stage2_save_path, exist_ok=True)
        scheduler_state = self.training_scheduler.state_dict()
        torch.save(
            {
                "update_step": self.update_step,
                "pending_update_budget": scheduler_state["pending_update_budget"],
                "warmup_ready_total_transitions": scheduler_state[
                    "warmup_ready_total_transitions"
                ],
                "warmup_ready_total_episodes": scheduler_state[
                    "warmup_ready_total_episodes"
                ],
                "version": self.version,
                "transitions_since_train": self.transitions_since_train,
                "episodes_since_train": self.episodes_since_train,
                "total_transitions_added": self.total_transitions_added,
                "total_episodes_added": self.total_episodes_added,
            },
            os.path.join(stage2_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )
        self.save_replay_checkpoints(stage2_save_path)

    def load_checkpoint(self, load_base_path):
        self.load_model_checkpoint(
            load_path=load_base_path,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
        )

        stage2_load_path = os.path.join(
            self._stage2_component_dir(load_base_path),
            f"checkpoint_rank_{self._rank}.pt",
        )
        if os.path.exists(stage2_load_path):
            state = torch.load(stage2_load_path, map_location="cpu", weights_only=False)
            self.update_step = int(state.get("update_step", 0))
            self.training_scheduler.load_state_dict(state)
            self.version = int(state.get("version", self.update_step))
            self.transitions_since_train = int(
                state.get("transitions_since_train", 0)
            )
            self.episodes_since_train = int(state.get("episodes_since_train", 0))
            self.total_transitions_added = int(state.get("total_transitions_added", 0))
            self.total_episodes_added = int(state.get("total_episodes_added", 0))
            has_legacy_replay = (
                state.get("replay_buffer") is not None
                or state.get("demo_buffer") is not None
            )
            if has_legacy_replay:
                raise RuntimeError(
                    "Legacy RLTStage2ReplayBuffer checkpoints are not supported by "
                    "the TrajectoryReplayBuffer refactor. Load a checkpoint saved "
                    "after the refactor or restart replay collection."
                )

        self.load_replay_checkpoints(self._stage2_component_dir(load_base_path))

    def set_global_step(self, global_step: int) -> None:
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
