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

"""Convert RLT Stage 2 rollout trajectories into replay trajectories."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory

from .rollout import (
    TransitionSource,
    resolve_chunk_source,
)


class RLTStage2TrajectoryReplayAdapter:
    """Builds RLinf replay trajectories from rollout trajectories.

    The adapter owns only RLT's transition semantics. Storage, sampling, and
    persistence stay in ``TrajectoryReplayBuffer``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.is_realworld = (
            str(cfg.env.train.get("env_type", "")).lower() == "realworld"
        )

    def build_replay_trajectories(self, traj: Trajectory) -> tuple[list[Trajectory], int]:
        """Returns replay trajectories and completed episode count."""

        return self._chunk_trajectory_to_transitions(traj)

    @staticmethod
    def _transition_trajectory(
        *,
        x: torch.Tensor,
        action_chunk: torch.Tensor,
        ref_chunk: torch.Tensor,
        base_chunks: torch.Tensor | None,
        rewards: torch.Tensor,
        next_x: torch.Tensor,
        next_ref_chunk: torch.Tensor,
        next_base_chunks: torch.Tensor | None,
        done: torch.Tensor | float | bool,
        intervention: torch.Tensor,
        source_chunk: torch.Tensor,
        source: int,
        collection_phase_id: int,
        success: int | bool = 0,
        intervention_flag: bool,
        episode_id: int = 0,
        step_id: int = 0,
        model_weights_id: str = "",
    ) -> Trajectory:
        chunk_length = int(rewards.reshape(-1).shape[0])
        source_chunk = source_chunk.to(torch.uint8).reshape(chunk_length)
        intervention = intervention.to(torch.float32).reshape(-1)

        return Trajectory(
            max_episode_length=1,
            model_weights_id=model_weights_id,
            rewards=torch.ones(1, 1, dtype=torch.float32),
            forward_inputs={
                "x": x.detach().to(torch.float32).reshape(1, 1, -1).cpu().contiguous(),
                "a": action_chunk.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                "a_tilde": ref_chunk.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                "action_chunk": action_chunk.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                "ref_chunk": ref_chunk.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                **(
                    {
                        "base_chunks": base_chunks.detach()
                        .to(torch.float32)
                        .reshape(1, 1, *base_chunks.shape)
                        .cpu()
                        .contiguous()
                    }
                    if base_chunks is not None
                    else {}
                ),
                "rewards": rewards.detach()
                .to(torch.float32)
                .reshape(1, 1, chunk_length)
                .cpu()
                .contiguous(),
                "next_x": next_x.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                "next_a_tilde": next_ref_chunk.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                "next_ref_chunk": next_ref_chunk.detach()
                .to(torch.float32)
                .reshape(1, 1, -1)
                .cpu()
                .contiguous(),
                **(
                    {
                        "next_base_chunks": next_base_chunks.detach()
                        .to(torch.float32)
                        .reshape(1, 1, *next_base_chunks.shape)
                        .cpu()
                        .contiguous()
                    }
                    if next_base_chunks is not None
                    else {}
                ),
                "dones": torch.as_tensor(done, dtype=torch.float32)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "intervention": intervention.reshape(1, 1, -1).cpu().contiguous(),
                "source": torch.as_tensor(int(source), dtype=torch.uint8)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "source_chunk": source_chunk.reshape(1, 1, chunk_length)
                .cpu()
                .contiguous(),
                "collection_phase_id": torch.as_tensor(
                    int(collection_phase_id),
                    dtype=torch.uint8,
                )
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "success": torch.as_tensor(int(bool(success)), dtype=torch.int8)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "intervention_flag": torch.as_tensor(
                    bool(intervention_flag),
                    dtype=torch.bool,
                )
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "episode_id": torch.as_tensor(episode_id, dtype=torch.int32)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "step_id": torch.as_tensor(step_id, dtype=torch.int32)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
            },
        )

    @staticmethod
    def _normalize_intervention_mask(
        intervention: torch.Tensor,
        *,
        action: torch.Tensor,
        chunk_len: int,
        action_dim: int,
    ) -> torch.Tensor:
        """Return an action-level flat intervention mask for one chunk."""

        mask = (
            intervention.detach()
            .to(device=action.device, dtype=torch.bool)
            .reshape(-1)
        )
        action_numel = int(action.numel())
        if mask.numel() == action_numel:
            return mask.reshape_as(action)
        if mask.numel() == chunk_len:
            return (
                mask.reshape(chunk_len, 1)
                .expand(chunk_len, action_dim)
                .reshape_as(action)
            )
        if mask.numel() == 1:
            return torch.full_like(action, bool(mask.item()), dtype=torch.bool)
        raise ValueError(
            "RLT intervention mask must be scalar, chunk_length, or action_chunk_dim, "
            f"got {tuple(intervention.shape)} for chunk_len={chunk_len}, "
            f"action_dim={action_dim}."
        )

    @staticmethod
    def _require_forward_tensor(
        traj: Trajectory,
        key: str,
        *,
        context: str,
    ) -> torch.Tensor:
        if not traj.forward_inputs or key not in traj.forward_inputs:
            raise RuntimeError(
                f"RLT Stage2 {context} replay requires forward_inputs[{key!r}]. "
                "Rollout must emit canonical RLT Stage2 fields."
            )
        value = traj.forward_inputs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"RLT Stage2 {context} forward_inputs[{key!r}] must be a "
                f"torch.Tensor, got {type(value).__name__}."
            )
        return value

    def _chunk_intervention_mask(
        self,
        traj: Trajectory,
        *,
        t: int,
        env_idx: int,
        action: torch.Tensor,
        fallback_intervention: torch.Tensor,
        chunk_len: int,
        action_dim: int,
    ) -> torch.Tensor:
        if not self.is_realworld or traj.intervene_flags is None:
            return self._normalize_intervention_mask(
                fallback_intervention,
                action=action,
                chunk_len=chunk_len,
                action_dim=action_dim,
            )

        return self._normalize_intervention_mask(
            traj.intervene_flags[t, env_idx],
            action=action,
            chunk_len=chunk_len,
            action_dim=action_dim,
        )

    def _chunk_trajectory_to_transitions(self, traj: Trajectory) -> tuple[list[Trajectory], int]:
        if traj.actions is None or not traj.forward_inputs:
            return [], 0

        traj_len = traj.actions.shape[0]
        bsz = traj.actions.shape[1]
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        replay_trajectories: list[Trajectory] = []
        completed_episodes = 0

        x_all = traj.forward_inputs.get("x")
        a_tilde_all = traj.forward_inputs.get("a_tilde")
        base_chunks_all = traj.forward_inputs.get("base_chunks")
        if x_all is None or a_tilde_all is None:
            return [], 0

        dones_all = traj.dones
        rewards_all = traj.rewards
        if dones_all is None or rewards_all is None:
            return [], 0
        intervention_flags_all = self._require_forward_tensor(
            traj,
            "intervention_flags",
            context="chunk",
        )
        source_chunk_all = self._require_forward_tensor(
            traj,
            "source_chunk",
            context="chunk",
        )
        collection_phase_id_all = self._require_forward_tensor(
            traj,
            "collection_phase_id",
            context="chunk",
        )
        record_transition_all = self._require_forward_tensor(
            traj,
            "record_transition",
            context="chunk",
        )
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
                record_transition = (
                    record_transition_all[t, env_idx]
                    .detach()
                    .to(torch.bool)
                    .reshape(-1)
                )
                if not bool(record_transition.all().item()):
                    continue
                done_idx = min(t + 1, dones_all.shape[0] - 1)
                env_done = float(dones_all[done_idx, env_idx].any().item())
                done = float(env_done > 0.0)
                action = traj.actions[t, env_idx].detach()
                intervention_mask = self._chunk_intervention_mask(
                    traj,
                    t=t,
                    env_idx=env_idx,
                    action=action,
                    fallback_intervention=intervention_flags_all[t, env_idx],
                    chunk_len=chunk_len,
                    action_dim=action_dim,
                )
                source_chunk = source_chunk_all[t, env_idx].detach().to(torch.uint8)
                collection_phase_id = int(
                    collection_phase_id_all[t, env_idx]
                    .reshape(-1)[0]
                    .detach()
                    .cpu()
                    .item()
                )

                x = x_all[t, env_idx].detach()
                a_tilde = a_tilde_all[t, env_idx].detach()
                if base_chunks_all is None:
                    base_chunks = a_tilde.unsqueeze(0)
                else:
                    base_chunks = base_chunks_all[t, env_idx].detach()
                rewards = rewards_all[t, env_idx].detach()

                if done > 0.0:
                    next_x = x
                    next_a_tilde = a_tilde
                elif t + 1 < traj_len:
                    next_x = x_all[t + 1, env_idx].detach()
                    next_a_tilde = a_tilde_all[t + 1, env_idx].detach()
                else:
                    if x_all.shape[0] <= t + 1 or a_tilde_all.shape[0] <= t + 1:
                        raise RuntimeError(
                            "RLT Stage2 rollout boundary transition is non-terminal "
                            "but missing cached final x/a_tilde. Rollout must send "
                            "the final student forward_inputs so actor training can "
                            "bootstrap without re-encoding VLA observations."
                        )
                    next_x = x_all[t + 1, env_idx].detach()
                    next_a_tilde = a_tilde_all[t + 1, env_idx].detach()
                if done > 0.0:
                    next_base_chunks = base_chunks
                elif base_chunks_all is None:
                    next_base_chunks = next_a_tilde.unsqueeze(0)
                elif t + 1 < base_chunks_all.shape[0]:
                    next_base_chunks = base_chunks_all[t + 1, env_idx].detach()
                else:
                    raise RuntimeError(
                        "RLT Stage2 EXPO boundary transition is non-terminal but "
                        "missing cached final base_chunks. Rollout must send the "
                        "final student forward_inputs so actor training can build "
                        "top-Q bootstrap candidates without re-encoding VLA "
                        "observations."
                    )

                source_chunk = source_chunk.reshape(chunk_len).clone()
                if self.is_realworld and bool(
                    intervention_mask.reshape(-1).any().item()
                ):
                    source_chunk[:] = int(TransitionSource.HUMAN)
                source = resolve_chunk_source(source_chunk.cpu().numpy())

                replay_trajectories.append(
                    self._transition_trajectory(
                        x=x,
                        action_chunk=action,
                        ref_chunk=a_tilde,
                        base_chunks=base_chunks,
                        rewards=rewards,
                        next_x=next_x,
                        next_ref_chunk=next_a_tilde,
                        next_base_chunks=next_base_chunks,
                        done=done,
                        intervention=intervention_mask,
                        source_chunk=source_chunk,
                        source=source,
                        collection_phase_id=collection_phase_id,
                        intervention_flag=bool(
                            intervention_mask.reshape(-1).any().item()
                        ),
                        step_id=t,
                        model_weights_id=traj.model_weights_id,
                    )
                )
                if env_done > 0.0:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes
