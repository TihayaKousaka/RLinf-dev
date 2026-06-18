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
    COLLECTION_PHASE_UNKNOWN,
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

    def build_replay_trajectories(self, traj: Trajectory) -> tuple[list[Trajectory], int]:
        """Returns replay trajectories and completed episode count."""

        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        if stride > 0:
            return self._stride_trajectory_to_transitions(traj, stride=stride)
        return self._chunk_trajectory_to_transitions(traj)

    @staticmethod
    def _transition_trajectory(
        *,
        x: torch.Tensor,
        action_chunk: torch.Tensor,
        ref_chunk: torch.Tensor,
        rewards: torch.Tensor,
        next_x: torch.Tensor,
        next_ref_chunk: torch.Tensor,
        done: torch.Tensor | float | bool,
        intervention: torch.Tensor,
        source_chunk: torch.Tensor,
        source: int | None,
        collection_phase_id: int | None,
        success: int | bool = 0,
        intervention_flag: bool | None = None,
        episode_id: int = 0,
        step_id: int = 0,
        model_weights_id: str = "",
    ) -> Trajectory:
        chunk_length = int(rewards.reshape(-1).shape[0])
        source_chunk = source_chunk.to(torch.uint8).reshape(chunk_length)
        resolved_source = (
            int(source) if source is not None else resolve_chunk_source(source_chunk.cpu().numpy())
        )
        intervention = intervention.to(torch.float32).reshape(-1)
        resolved_intervention = (
            bool(intervention_flag)
            if intervention_flag is not None
            else bool(
                intervention.any().item()
                or (source_chunk == int(TransitionSource.HUMAN)).any().item()
                or (source_chunk == int(TransitionSource.MIXED)).any().item()
            )
        )
        phase = (
            COLLECTION_PHASE_UNKNOWN
            if collection_phase_id is None
            else int(collection_phase_id)
        )

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
                "dones": torch.as_tensor(done, dtype=torch.float32)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "intervention": intervention.reshape(1, 1, -1).cpu().contiguous(),
                "source": torch.as_tensor(resolved_source, dtype=torch.uint8)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "source_chunk": source_chunk.reshape(1, 1, chunk_length)
                .cpu()
                .contiguous(),
                "collection_phase_id": torch.as_tensor(phase, dtype=torch.uint8)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "success": torch.as_tensor(int(bool(success)), dtype=torch.int8)
                .reshape(1, 1, 1)
                .cpu()
                .contiguous(),
                "intervention_flag": torch.as_tensor(
                    resolved_intervention,
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
        if x_all is None or a_tilde_all is None:
            return [], 0

        dones_all = traj.dones
        rewards_all = traj.rewards
        if dones_all is None or rewards_all is None:
            return [], 0
        intervention_flags_all = traj.forward_inputs.get("intervention_flags")
        if intervention_flags_all is None:
            intervention_flags_all = traj.intervene_flags
        source_chunk_all = traj.forward_inputs.get("source_chunk")
        collection_phase_id_all = traj.forward_inputs.get("collection_phase_id")
        record_transition_all = traj.forward_inputs.get("record_transition")
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
                if record_transition_all is not None:
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
                intervention_mask = torch.zeros_like(action, dtype=torch.bool)
                if intervention_flags_all is not None:
                    intervention_mask = self._normalize_intervention_mask(
                        intervention_flags_all[t, env_idx],
                        action=action,
                        chunk_len=chunk_len,
                        action_dim=action_dim,
                    )
                source_chunk = None
                source = None
                if source_chunk_all is not None:
                    source_chunk = source_chunk_all[t, env_idx].detach().to(torch.uint8)
                collection_phase_id = None
                if collection_phase_id_all is not None:
                    collection_phase_id = int(
                        collection_phase_id_all[t, env_idx]
                        .reshape(-1)[0]
                        .detach()
                        .cpu()
                        .item()
                    )

                x = x_all[t, env_idx].detach()
                a_tilde = a_tilde_all[t, env_idx].detach()
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

                if source_chunk is None:
                    step_intervention = intervention_mask.reshape(
                        chunk_len,
                        action_dim,
                    ).any(dim=-1)
                    source_chunk = torch.where(
                        step_intervention,
                        torch.full(
                            (chunk_len,),
                            int(TransitionSource.HUMAN),
                            dtype=torch.uint8,
                            device=action.device,
                        ),
                        torch.full(
                            (chunk_len,),
                            int(TransitionSource.RL),
                            dtype=torch.uint8,
                            device=action.device,
                        ),
                    )
                    source = resolve_chunk_source(source_chunk.cpu().numpy())
                else:
                    step_intervention = intervention_mask.reshape(
                        chunk_len,
                        action_dim,
                    ).any(dim=-1)
                    source_chunk = source_chunk.reshape(chunk_len).clone()
                    source_chunk[step_intervention.to(source_chunk.device)] = int(
                        TransitionSource.HUMAN
                    )
                    source = resolve_chunk_source(source_chunk.cpu().numpy())

                replay_trajectories.append(
                    self._transition_trajectory(
                        x=x,
                        action_chunk=action,
                        ref_chunk=a_tilde,
                        rewards=rewards,
                        next_x=next_x,
                        next_ref_chunk=next_a_tilde,
                        done=done,
                        intervention=intervention_mask,
                        source_chunk=source_chunk,
                        source=source,
                        collection_phase_id=collection_phase_id,
                        intervention_flag=bool(intervention_mask.reshape(-1).any().item()),
                        step_id=t,
                        model_weights_id=traj.model_weights_id,
                    )
                )
                if env_done > 0.0:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes

    def _stride_trajectory_to_transitions(
        self,
        traj: Trajectory,
        *,
        stride: int,
    ) -> tuple[list[Trajectory], int]:
        if stride <= 0:
            raise ValueError(f"RLT replay_subsample_stride must be positive, got {stride}.")
        if traj.actions is None or not traj.forward_inputs:
            return [], 0

        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        traj_len = int(traj.actions.shape[0])
        bsz = int(traj.actions.shape[1])
        replay_trajectories: list[Trajectory] = []
        completed_episodes = 0

        step_trace_all = traj.forward_inputs.get("rlt_step_trace", None)
        if not isinstance(step_trace_all, dict):
            raise RuntimeError(
                "RLT Stage2 stride replay requires forward_inputs['rlt_step_trace']. "
                "Run rollout with RLT step-trace emission enabled."
            )
        step_x_all = step_trace_all.get("x", None)
        step_ref_all = step_trace_all.get("a_tilde", None)
        step_trace_valid_all = traj.forward_inputs.get("rlt_step_trace_valid", None)
        if not isinstance(step_x_all, torch.Tensor) or not isinstance(
            step_ref_all,
            torch.Tensor,
        ):
            raise RuntimeError(
                "RLT Stage2 stride replay requires rlt_step_trace.x and "
                "rlt_step_trace.a_tilde tensors."
            )
        if (
            step_x_all.shape[0] < traj_len + 1
            or step_ref_all.shape[0] < traj_len + 1
        ):
            raise RuntimeError(
                "RLT Stage2 stride replay step trace must align with rollout "
                f"trajectory length {traj_len}+1, got x={tuple(step_x_all.shape)} "
                f"and a_tilde={tuple(step_ref_all.shape)}."
            )
        if not isinstance(step_trace_valid_all, torch.Tensor):
            raise RuntimeError(
                "RLT Stage2 stride replay requires "
                "forward_inputs['rlt_step_trace_valid']."
            )
        x_all = traj.forward_inputs.get("x")
        a_tilde_all = traj.forward_inputs.get("a_tilde")
        if not isinstance(x_all, torch.Tensor) or not isinstance(
            a_tilde_all,
            torch.Tensor,
        ):
            raise RuntimeError("RLT Stage2 stride replay requires boundary x/a_tilde.")

        dones_all = traj.dones
        rewards_all = traj.rewards
        if dones_all is None or rewards_all is None:
            return [], 0
        source_chunk_all = traj.forward_inputs.get("source_chunk")
        intervention_flags_all = traj.forward_inputs.get("intervention_flags")
        if intervention_flags_all is None:
            intervention_flags_all = traj.intervene_flags
        collection_phase_id_all = traj.forward_inputs.get("collection_phase_id")
        record_transition_all = traj.forward_inputs.get("record_transition")
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))
        step_trace_infos_all = traj.forward_inputs.get("rlt_step_trace_infos")

        def _trace_policy_scalar(
            name: str,
            *,
            trace_idx: int,
            env_idx: int,
            offset: int,
        ) -> torch.Tensor | None:
            if not isinstance(step_trace_infos_all, dict):
                return None
            policy_info = step_trace_infos_all.get("policy_info")
            if not isinstance(policy_info, dict):
                return None
            value = policy_info.get(name)
            if not isinstance(value, torch.Tensor):
                return None
            if value.shape[0] <= trace_idx or value.shape[2] <= offset:
                return None
            return value[trace_idx, env_idx, offset].detach().reshape(-1)[0]

        for env_idx in range(bsz):
            flat_actions: list[torch.Tensor] = []
            state_by_step: list[torch.Tensor] = []
            ref_by_step: list[torch.Tensor] = []
            flat_rewards: list[torch.Tensor] = []
            flat_dones: list[torch.Tensor] = []
            flat_sources: list[torch.Tensor] = []
            flat_interventions: list[torch.Tensor] = []
            flat_record: list[torch.Tensor] = []
            flat_phase: list[torch.Tensor] = []
            state_record_by_step: list[torch.Tensor] = []
            state_phase_by_step: list[torch.Tensor] = []
            for t in range(traj_len):
                absolute_start = t * chunk_len
                if len(state_by_step) < absolute_start:
                    raise RuntimeError(
                        "RLT Stage2 stride replay has a gap in reconstructed "
                        f"state trace before rollout chunk {t}."
                    )
                if record_transition_all is None:
                    boundary_record = torch.ones(
                        (),
                        dtype=torch.bool,
                        device=traj.actions.device,
                    )
                else:
                    boundary_record = (
                        record_transition_all[t, env_idx]
                        .detach()
                        .to(device=traj.actions.device, dtype=torch.bool)
                        .reshape(-1)
                        .all()
                    )
                if collection_phase_id_all is None:
                    boundary_phase = torch.as_tensor(
                        COLLECTION_PHASE_UNKNOWN,
                        dtype=torch.uint8,
                        device=traj.actions.device,
                    )
                else:
                    boundary_phase = (
                        collection_phase_id_all[t, env_idx]
                        .detach()
                        .to(device=traj.actions.device, dtype=torch.uint8)
                        .reshape(-1)[0]
                    )
                boundary_x = x_all[t, env_idx].detach()
                boundary_ref = a_tilde_all[t, env_idx].detach().reshape(-1)
                if len(state_by_step) == absolute_start:
                    state_by_step.append(boundary_x)
                    ref_by_step.append(boundary_ref)
                    state_record_by_step.append(boundary_record)
                    state_phase_by_step.append(boundary_phase)
                else:
                    state_by_step[absolute_start] = boundary_x
                    ref_by_step[absolute_start] = boundary_ref
                    state_record_by_step[absolute_start] = boundary_record
                    state_phase_by_step[absolute_start] = boundary_phase

                trace_idx = t + 1
                trace_valid = bool(
                    step_trace_valid_all[trace_idx, env_idx]
                    .detach()
                    .reshape(-1)[0]
                    .item()
                )
                if not trace_valid:
                    raise RuntimeError(
                        "RLT Stage2 stride replay is missing a valid step trace "
                        f"for rollout chunk {t}."
                    )
                action_chunk = traj.actions[t, env_idx].detach().reshape(
                    chunk_len,
                    action_dim,
                )
                step_x_trace = step_x_all[trace_idx, env_idx].detach()
                step_ref_trace = step_ref_all[trace_idx, env_idx].detach()
                if step_x_trace.shape[0] != chunk_len:
                    raise RuntimeError(
                        "RLT Stage2 stride replay step x trace length mismatch: "
                        f"expected {chunk_len}, got {tuple(step_x_trace.shape)}."
                    )
                if step_ref_trace.shape[0] != chunk_len:
                    raise RuntimeError(
                        "RLT Stage2 stride replay step ref trace length mismatch: "
                        f"expected {chunk_len}, got {tuple(step_ref_trace.shape)}."
                    )
                rewards = rewards_all[t, env_idx].detach().reshape(chunk_len)
                done_idx = min(t + 1, dones_all.shape[0] - 1)
                dones = dones_all[done_idx, env_idx].detach().reshape(chunk_len)
                intervention_mask = torch.zeros_like(action_chunk, dtype=torch.bool)
                if intervention_flags_all is not None:
                    intervention_mask = self._normalize_intervention_mask(
                        intervention_flags_all[t, env_idx],
                        action=action_chunk.reshape(-1),
                        chunk_len=chunk_len,
                        action_dim=action_dim,
                    ).reshape(chunk_len, action_dim)
                if source_chunk_all is not None:
                    source_chunk = (
                        source_chunk_all[t, env_idx].detach().to(torch.uint8).reshape(chunk_len)
                    )
                else:
                    source_chunk = torch.full(
                        (chunk_len,),
                        int(TransitionSource.RL),
                        dtype=torch.uint8,
                        device=action_chunk.device,
                    )
                source_chunk = source_chunk.clone()
                source_chunk[intervention_mask.any(dim=-1).to(source_chunk.device)] = (
                    int(TransitionSource.HUMAN)
                )

                for offset in range(chunk_len):
                    absolute_after = absolute_start + offset + 1
                    trace_record = _trace_policy_scalar(
                        "record_transition",
                        trace_idx=trace_idx,
                        env_idx=env_idx,
                        offset=offset,
                    )
                    next_record = (
                        trace_record.to(device=action_chunk.device, dtype=torch.bool)
                        if trace_record is not None
                        else boundary_record.to(
                            device=action_chunk.device,
                            dtype=torch.bool,
                        )
                    )
                    if len(state_by_step) == absolute_after:
                        state_by_step.append(step_x_trace[offset])
                        ref_by_step.append(step_ref_trace[offset].reshape(-1))
                        state_record_by_step.append(next_record)
                        state_phase_by_step.append(boundary_phase)
                    elif len(state_by_step) > absolute_after:
                        state_by_step[absolute_after] = step_x_trace[offset]
                        ref_by_step[absolute_after] = step_ref_trace[offset].reshape(-1)
                        state_record_by_step[absolute_after] = next_record
                        state_phase_by_step[absolute_after] = boundary_phase
                    else:
                        raise RuntimeError(
                            "RLT Stage2 stride replay has a gap in reconstructed "
                            f"state trace after rollout chunk {t} offset {offset}."
                        )
                    flat_actions.append(action_chunk[offset])
                    flat_rewards.append(rewards[offset])
                    flat_dones.append(dones[offset].to(torch.bool))
                    flat_sources.append(source_chunk[offset])
                    flat_interventions.append(intervention_mask[offset])
                    flat_record.append(
                        state_record_by_step[absolute_start + offset]
                    )
                    flat_phase.append(state_phase_by_step[absolute_start + offset])
                if bool(dones.any().item()) and not auto_reset:
                    break

            if not flat_actions:
                continue

            action_steps = torch.stack(flat_actions, dim=0)
            state_steps = torch.stack(state_by_step, dim=0)
            ref_chunks_by_step = torch.stack(ref_by_step, dim=0)
            reward_steps = torch.stack(flat_rewards, dim=0)
            done_steps = torch.stack(flat_dones, dim=0).to(torch.bool)
            source_steps = torch.stack(flat_sources, dim=0).to(torch.uint8)
            intervention_steps = torch.stack(flat_interventions, dim=0).to(torch.bool)
            record_steps = torch.stack(flat_record, dim=0).to(torch.bool)
            phase_steps = torch.stack(flat_phase, dim=0).to(torch.uint8)

            total_steps = int(action_steps.shape[0])
            cursor = 0
            while cursor < total_steps:
                while cursor < total_steps and not bool(record_steps[cursor].item()):
                    cursor += 1
                if cursor >= total_steps:
                    break

                segment_start = cursor
                segment_end = segment_start
                is_terminal_segment = False
                while segment_end < total_steps:
                    if not bool(record_steps[segment_end].item()):
                        break
                    segment_end += 1
                    if bool(done_steps[segment_end - 1].item()):
                        completed_episodes += 1
                        is_terminal_segment = True
                        break

                segment_len = segment_end - segment_start
                if segment_len < chunk_len:
                    cursor = segment_end
                    continue

                window_starts = list(
                    range(segment_start, segment_end - chunk_len + 1, stride)
                )
                if is_terminal_segment:
                    terminal_start = segment_end - chunk_len
                    if terminal_start not in window_starts:
                        window_starts.append(terminal_start)

                for start in window_starts:
                    end = start + chunk_len
                    action_chunk = action_steps[start:end]
                    rewards = reward_steps[start:end]
                    source_chunk = source_steps[start:end]
                    intervention = intervention_steps[start:end]
                    source = resolve_chunk_source(source_chunk.cpu().numpy())

                    done = bool(done_steps[start:end].any().item())
                    next_index = end
                    next_x = state_steps[next_index]
                    next_ref_chunk = ref_chunks_by_step[next_index]
                    ref_chunk = ref_chunks_by_step[start].reshape(
                        chunk_len,
                        action_dim,
                    )

                    phase_id = int(phase_steps[start].detach().cpu().item())
                    replay_trajectories.append(
                        self._transition_trajectory(
                            x=state_steps[start],
                            action_chunk=action_chunk.reshape(-1),
                            ref_chunk=ref_chunk.reshape(-1),
                            rewards=rewards,
                            next_x=next_x,
                            next_ref_chunk=next_ref_chunk.reshape(-1),
                            done=done,
                            intervention=intervention.reshape(-1),
                            source_chunk=source_chunk,
                            source=source,
                            collection_phase_id=phase_id,
                            intervention_flag=bool(intervention.any().item()),
                            step_id=start,
                            model_weights_id=traj.model_weights_id,
                        )
                    )

                cursor = segment_end

        return replay_trajectories, completed_episodes
