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

"""Shared FSDP worker base for embodied off-policy algorithms."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedOffPolicyFSDPActor(EmbodiedFSDPActor):
    """Base class for embodied off-policy FSDP actor workers.

    Subclasses own the algorithm-specific replay schema and update rule. This
    base owns the RLinf worker contract shared by SAC, RLT, and future
    off-policy embodied algorithms.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.demo_buffer = None
        self.update_step = 0

    def init_trajectory_replay_buffers(
        self,
        *,
        replay_cfg: Any,
        replay_seed: int,
        replay_capacity: int | None = None,
        replay_subdir: str = "replay_buffer",
        demo_cfg: Any | None = None,
        demo_seed: int | None = None,
        demo_capacity: int | None = None,
        demo_subdir: str = "demo_buffer",
    ) -> int:
        """Initialize standard RLinf trajectory replay buffers.

        Algorithm workers own the replay schema and update rule. This helper
        only centralizes rank-local path resolution and TrajectoryReplayBuffer
        construction so off-policy workers do not each duplicate infra code.
        """

        self.replay_buffer = self.build_trajectory_replay_buffer(
            replay_cfg,
            seed=replay_seed,
            default_subdir=replay_subdir,
            capacity=replay_capacity,
        )

        min_demo_buffer_size = 0
        if demo_cfg is None:
            self.demo_buffer = None
            return min_demo_buffer_size

        self.demo_buffer = self.build_trajectory_replay_buffer(
            demo_cfg,
            seed=replay_seed if demo_seed is None else demo_seed,
            default_subdir=demo_subdir,
            capacity=demo_capacity,
        )
        min_demo_buffer_size = int(demo_cfg.get("min_buffer_size", 0))
        if demo_cfg.get("load_path", None) is not None:
            self.demo_buffer.load_checkpoint(
                demo_cfg.load_path,
                is_distributed=True,
                local_rank=self._rank,
                world_size=self._world_size,
            )
        return min_demo_buffer_size

    def build_trajectory_replay_buffer(
        self,
        buffer_cfg: Any,
        *,
        seed: int,
        default_subdir: str,
        capacity: int | None = None,
    ) -> TrajectoryReplayBuffer:
        """Create a rank-local TrajectoryReplayBuffer from config."""

        auto_save_path = self.resolve_replay_auto_save_path(
            buffer_cfg,
            default_subdir=default_subdir,
        )
        fallback_capacity = 5 if capacity is None else int(capacity)
        return TrajectoryReplayBuffer(
            seed=int(seed),
            enable_cache=bool(buffer_cfg.get("enable_cache", True)),
            cache_size=int(buffer_cfg.get("cache_size", fallback_capacity)),
            sample_window_size=int(
                buffer_cfg.get("sample_window_size", fallback_capacity)
            ),
            auto_save=bool(buffer_cfg.get("auto_save", False)),
            auto_save_path=auto_save_path,
            trajectory_format=buffer_cfg.get("trajectory_format", "pt"),
        )

    def resolve_replay_auto_save_path(
        self,
        buffer_cfg: Any,
        *,
        default_subdir: str,
    ) -> str:
        """Resolve a rank-local replay save path using RLinf log roots."""

        auto_save_path = buffer_cfg.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = buffer_cfg.get("auto_save_dir", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path,
                default_subdir,
            )
        return os.path.join(str(auto_save_path), f"rank_{self._rank}")

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive rollout trajectories and delegate storage to the algorithm."""
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        trajectories = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            trajectories.append(trajectory)

        self.add_rollout_trajectories(trajectories)

    def add_rollout_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Add freshly collected rollout trajectories to algorithm replay."""
        raise NotImplementedError

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """Off-policy algorithms train from replay and do not compute GAE."""
        return {}

    def average_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Average scalar/list metrics across updates and data-parallel ranks."""
        mean_metric_dict: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                mean_metric_dict[key] = float(
                    np.mean(
                        [
                            item.detach().cpu().item()
                            if isinstance(item, torch.Tensor)
                            else item
                            for item in value
                        ]
                    )
                )
            elif isinstance(value, torch.Tensor):
                mean_metric_dict[key] = float(value.detach().cpu().item())
            else:
                mean_metric_dict[key] = float(value)
        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    def ensure_training_state_loaded(self) -> None:
        """Load offloaded model/optimizer state before local training."""
        if not self.enable_offload:
            return
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

    def ensure_checkpoint_state_loaded(self) -> None:
        """Load offloaded state before checkpointing and keep it resident."""
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

    def checkpoint_format(self) -> str:
        """Return the configured FSDP checkpoint format."""
        return (
            "local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp"
        )

    def save_model_checkpoint(
        self,
        *,
        save_path: str,
        model: torch.nn.Module | None = None,
        optimizers: Any | None = None,
        lr_schedulers: Any | None = None,
    ) -> None:
        """Save a model checkpoint using the standard embodied FSDP format."""
        self._strategy.save_checkpoint(
            model=self.model if model is None else model,
            optimizers=[self.optimizer] if optimizers is None else optimizers,
            lr_schedulers=[self.lr_scheduler]
            if lr_schedulers is None
            else lr_schedulers,
            save_path=save_path,
            checkpoint_format=self.checkpoint_format(),
        )

    def load_model_checkpoint(
        self,
        *,
        load_path: str,
        model: torch.nn.Module | None = None,
        optimizers: Any | None = None,
        lr_schedulers: Any | None = None,
    ) -> None:
        """Load a model checkpoint using the standard embodied FSDP format."""
        self._strategy.load_checkpoint(
            model=self.model if model is None else model,
            optimizers=[self.optimizer] if optimizers is None else optimizers,
            lr_schedulers=[self.lr_scheduler]
            if lr_schedulers is None
            else lr_schedulers,
            load_path=load_path,
            checkpoint_format=self.checkpoint_format(),
        )

    def replay_checkpoint_path(self, component_dir: str, name: str) -> str:
        """Return the rank-local replay checkpoint directory."""
        return os.path.join(component_dir, name, f"rank_{self._rank}")

    def save_replay_checkpoints(self, component_dir: str) -> None:
        """Save standard RLinf replay buffers for this rank."""
        if self.replay_buffer is not None:
            self.replay_buffer.save_checkpoint(
                self.replay_checkpoint_path(component_dir, "replay_buffer")
            )
        if self.demo_buffer is not None:
            self.demo_buffer.save_checkpoint(
                self.replay_checkpoint_path(component_dir, "demo_buffer")
            )

    def load_replay_checkpoints(self, component_dir: str) -> None:
        """Load standard RLinf replay buffers when checkpoint directories exist."""
        replay_load_path = self.replay_checkpoint_path(component_dir, "replay_buffer")
        if self.replay_buffer is not None and os.path.exists(replay_load_path):
            self.replay_buffer.load_checkpoint(replay_load_path)

        demo_load_path = self.replay_checkpoint_path(component_dir, "demo_buffer")
        if self.demo_buffer is not None and os.path.exists(demo_load_path):
            self.demo_buffer.load_checkpoint(demo_load_path)

    def prepare_micro_batches(
        self,
        batch: dict[str, torch.Tensor],
        *,
        global_batch_size_per_rank: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Split one rank-local global batch into micro-batches."""
        if global_batch_size_per_rank is None:
            global_batch_size_per_rank = (
                self.cfg.actor.global_batch_size // self._world_size
            )
        assert global_batch_size_per_rank % self.cfg.actor.micro_batch_size == 0, (
            "global batch per rank must be divisible by micro_batch_size"
        )
        micro_batch_count = (
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size
        )
        self.gradient_accumulation = micro_batch_count
        return split_dict_to_chunk(batch, micro_batch_count)
