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

"""Env-side policy_info adapter for RLT Stage 2 collection."""

from __future__ import annotations

from typing import Any, Literal

import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult, RolloutResult


class RLTStage2PolicyInfoAdapter:
    """Owns RLT Stage 2 env-side state exposed to rollout workers."""

    def __init__(
        self,
        *,
        cfg,
        train_batch_size: int | None,
        eval_batch_size: int | None,
    ) -> None:
        self.cfg = cfg
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_states: list[dict[str, torch.Tensor]] = []
        self.eval_states: list[dict[str, torch.Tensor]] = []
        self.train_maniskill_controllers: list[Any | None] = []
        self.eval_maniskill_controllers: list[Any | None] = []

    def init_stage(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if not self.enabled(mode):
            return None

        batch_size = self._batch_size(mode)
        if self.env_type(mode) == "maniskill":
            controller = self._init_maniskill_controller(
                stage_id=stage_id,
                mode=mode,
                batch_size=batch_size,
                env=env,
            )
            states = self._states(mode)
            self._ensure_len(states, stage_id, {})
            states[stage_id] = controller.state
            return controller.export_policy_info(controller.state)

        state = self._init_generic_state(batch_size=batch_size, mode=mode)
        states = self._states(mode)
        self._ensure_len(states, stage_id, {})
        states[stage_id] = state
        return self.export_policy_info(state)

    def update_stage(
        self,
        *,
        infos: dict[str, Any] | None,
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        states = self._states(mode)
        if not self.enabled(mode) or infos is None or stage_id >= len(states):
            return None

        if self.env_type(mode) == "maniskill":
            controllers = self._maniskill_controllers(mode)
            if stage_id >= len(controllers) or controllers[stage_id] is None:
                controller = self._init_maniskill_controller(
                    stage_id=stage_id,
                    mode=mode,
                    batch_size=int(chunk_dones.shape[0]),
                    env=env,
                )
            else:
                controller = controllers[stage_id]
            assert controller is not None
            policy_info = controller.update(
                infos=infos,
                chunk_dones=chunk_dones,
                intervention_enabled=self.local_correction_enabled(),
            )
            self._ensure_len(states, stage_id, {})
            states[stage_id] = controller.state
            return policy_info

        return self._update_generic_stage(
            infos=infos,
            chunk_dones=chunk_dones,
            stage_id=stage_id,
            mode=mode,
        )

    def update_last_action_metadata(
        self,
        *,
        rollout_result: Any,
        intervene_flags: torch.Tensor,
    ) -> None:
        if not self.local_correction_enabled():
            return
        from rlinf.models.embodiment.rlt_stage2.rollout_result_adapter import (
            update_last_rlt_action_metadata,
        )

        update_last_rlt_action_metadata(rollout_result, intervene_flags)

    def build_step_obs(
        self,
        *,
        start_obs: dict[str, Any] | None,
        obs_list,
    ) -> dict[str, Any] | None:
        from rlinf.models.embodiment.rlt_stage2.env_adapter import build_step_obs

        return build_step_obs(self.cfg, start_obs, obs_list)

    def append_step_trace(
        self,
        *,
        rollout_accumulator: EmbodiedRolloutResult,
        rollout_result: RolloutResult,
    ) -> None:
        if rollout_result.rlt_step_trace:
            rollout_accumulator.append_rlt_step_trace(rollout_result.rlt_step_trace)

    def final_forward_inputs(self, rollout_result: RolloutResult) -> dict[str, Any]:
        return rollout_result.forward_inputs

    def collect_rollout_metrics(
        self,
        *,
        env_metrics: dict[str, list],
        rollout_result: RolloutResult,
    ) -> None:
        forward_inputs = rollout_result.forward_inputs
        intervention_flags = forward_inputs.get("intervention_flags", None)
        if intervention_flags is not None:
            actual_intervention = intervention_flags.detach().float().reshape(-1).cpu()
            env_metrics["expert_intervention_actual_rate"].append(
                actual_intervention
            )
            env_metrics["expert_takeover_rate"].append(actual_intervention)
            intervention_phase = forward_inputs.get("intervention_phase", None)
            if intervention_phase is not None:
                phase = intervention_phase.detach().float().reshape(-1).cpu()
                if phase.numel() == actual_intervention.numel():
                    env_metrics["grasp_intervention_rate"].append(
                        actual_intervention * (phase == 1).float()
                    )
                    env_metrics["insert_intervention_rate"].append(
                        actual_intervention * (phase == 2).float()
                    )

        metric_names = {
            "intervention_requested": "expert_intervention_requested_rate",
            "ready_for_online": "rlt_ready_for_online",
            "in_critical_phase": "rlt_in_critical_phase",
            "record_transition": "rlt_record_transition",
            "student_control": "student_control_rate",
        }
        for source_key, metric_key in metric_names.items():
            value = forward_inputs.get(source_key, None)
            if value is not None:
                env_metrics[metric_key].append(
                    value.detach().float().reshape(-1).cpu()
                )

    def emit_status(
        self,
        *,
        env_metrics: dict[str, torch.Tensor],
        rank: int,
        last_logged_phase: str | None,
        log_info,
    ) -> str | None:
        from rlinf.models.embodiment.rlt_stage2.env_adapter import emit_rollout_status

        return emit_rollout_status(
            cfg=self.cfg,
            env_metrics=env_metrics,
            rank=rank,
            last_logged_phase=last_logged_phase,
            log_info=log_info,
        )

    def enabled(self, mode: Literal["train", "eval"]) -> bool:
        if not self.td3_enabled():
            return False
        env_type = self.env_type(mode)
        if env_type == "realworld":
            return True
        if self.local_correction_enabled():
            return env_type == "maniskill"
        return False

    def td3_enabled(self) -> bool:
        return (
            self.cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and self.cfg.actor.model.get("model_type", None) == "rlt_stage2"
        )

    def intervention_mode(self) -> str:
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        return str(intervention_cfg.get("mode", "local_correction"))

    def intervention_enabled(self) -> bool:
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        return (
            self.td3_enabled()
            and bool(intervention_cfg.get("enable", False))
            and self.intervention_mode() in {"local_correction", "human_override"}
        )

    def local_correction_enabled(self) -> bool:
        return (
            self.intervention_enabled()
            and self.intervention_mode() == "local_correction"
        )

    def env_type(self, mode: Literal["train", "eval"]) -> str:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        return str(env_cfg.get("env_type", "")).lower()

    def _batch_size(self, mode: Literal["train", "eval"]) -> int:
        batch_size = self.train_batch_size if mode == "train" else self.eval_batch_size
        if batch_size is None:
            raise RuntimeError(f"RLT policy_info {mode} batch size is not initialized.")
        return int(batch_size)

    def _task_mode(self, mode: Literal["train", "eval"]) -> str:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        return str(env_cfg.get("task_mode", "critical_phase"))

    def _default_in_critical_phase(self, mode: Literal["train", "eval"]) -> bool:
        return self._task_mode(mode) == "critical_phase"

    def _default_record_transition(self, mode: Literal["train", "eval"]) -> bool:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        if bool(env_cfg.get("record_prefix_before_critical_phase", False)):
            return True
        return self._default_in_critical_phase(mode)

    def _init_maniskill_controller(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        batch_size: int,
        env: Any | None,
    ) -> Any:
        from rlinf.envs.maniskill.rlt_intervention import (
            ManiSkillLocalCorrectionController,
        )

        controllers = self._maniskill_controllers(mode)
        self._ensure_len(controllers, stage_id, None)
        controller = ManiSkillLocalCorrectionController(
            cfg=self.cfg,
            batch_size=batch_size,
            mode=mode,
            hole_radii=self._get_maniskill_hole_radii(env),
        )
        controllers[stage_id] = controller
        return controller

    def _init_generic_state(
        self,
        *,
        batch_size: int,
        mode: Literal["train", "eval"],
    ) -> dict[str, torch.Tensor]:
        default_in_critical_phase = self._default_in_critical_phase(mode)
        return {
            "intervention_region": torch.zeros(batch_size, dtype=torch.bool),
            "intervention_phase": torch.zeros(batch_size, dtype=torch.int64),
            "expert_takeover": torch.zeros(batch_size, dtype=torch.bool),
            "deviation": torch.zeros(batch_size, dtype=torch.bool),
            "deviation_count": torch.zeros(batch_size, dtype=torch.int64),
            "grasp_deviation_count": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_left": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_used": torch.zeros(batch_size, dtype=torch.int64),
            "prev_yz_error": torch.full((batch_size,), float("nan"), dtype=torch.float32),
            "prev_hole_x": torch.full((batch_size,), float("nan"), dtype=torch.float32),
            "in_critical_phase": torch.full(
                (batch_size,),
                default_in_critical_phase,
                dtype=torch.bool,
            ),
            "record_transition": torch.full(
                (batch_size,),
                self._default_record_transition(mode),
                dtype=torch.bool,
            ),
            "critical_phase_started": torch.full(
                (batch_size,),
                default_in_critical_phase,
                dtype=torch.bool,
            ),
        }

    def _update_generic_stage(
        self,
        *,
        infos: dict[str, Any],
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
    ) -> dict[str, torch.Tensor]:
        state = self._states(mode)[stage_id]
        device = chunk_dones.device
        done_any = chunk_dones.any(dim=1).to(device)
        batch_size = int(done_any.shape[0])
        for key, value in state.items():
            state[key] = value.to(device)

        expert_takeover = self._coerce_bool_info(
            self._lookup_info_value(infos, "expert_takeover"),
            batch_size=batch_size,
            device=device,
        )
        deviation = self._coerce_bool_info(
            self._lookup_info_value(infos, "deviation"),
            batch_size=batch_size,
            device=device,
        )
        intervention_region = self._coerce_bool_info(
            self._lookup_info_value(infos, "intervention_region"),
            batch_size=batch_size,
            device=device,
        )
        state["expert_takeover"] = torch.where(
            done_any,
            torch.zeros_like(expert_takeover),
            expert_takeover,
        )
        state["deviation"] = torch.where(done_any, torch.zeros_like(deviation), deviation)
        state["intervention_region"] = torch.where(
            done_any,
            torch.zeros_like(intervention_region),
            intervention_region,
        )
        for key in (
            "in_critical_phase",
            "record_transition",
            "critical_phase_started",
        ):
            default_value = (
                self._default_record_transition(mode)
                if key == "record_transition"
                else self._default_in_critical_phase(mode)
            )
            default_tensor = torch.full_like(state[key], default_value)
            current_value = self._coerce_bool_info(
                self._lookup_info_value(infos, key),
                batch_size=batch_size,
                device=device,
            )
            state[key] = torch.where(done_any, default_tensor, current_value)
        for key in (
            "deviation_count",
            "grasp_deviation_count",
            "takeover_left",
            "takeover_used",
        ):
            state[key] = torch.where(
                done_any,
                torch.zeros_like(state[key]),
                self._coerce_int_info(
                    self._lookup_info_value(infos, key),
                    batch_size=batch_size,
                    device=device,
                ),
            )
        state["prev_yz_error"] = torch.full_like(state["prev_yz_error"], float("nan"))
        state["prev_hole_x"] = torch.full_like(state["prev_hole_x"], float("nan"))
        state["intervention_phase"] = torch.where(
            done_any,
            torch.zeros_like(state["intervention_phase"]),
            self._coerce_int_info(
                self._lookup_info_value(infos, "intervention_phase"),
                batch_size=batch_size,
                device=device,
            ),
        )
        return self.export_policy_info(state)

    @staticmethod
    def _lookup_info_value(infos: dict[str, Any], key: str) -> Any:
        if key in infos:
            return infos[key]
        policy_info = infos.get("policy_info")
        if isinstance(policy_info, dict) and key in policy_info:
            return policy_info[key]
        final_info = infos.get("final_info")
        if isinstance(final_info, dict):
            if key in final_info:
                return final_info[key]
            final_policy_info = final_info.get("policy_info")
            if isinstance(final_policy_info, dict) and key in final_policy_info:
                return final_policy_info[key]
        return None

    @staticmethod
    def _coerce_bool_info(
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 1:
            return torch.full(
                (batch_size,),
                bool(tensor.reshape(-1)[0].item()),
                dtype=torch.bool,
                device=device,
            )
        tensor = tensor.reshape(batch_size, -1)
        return tensor.to(torch.bool).any(dim=1)

    @staticmethod
    def _coerce_int_info(
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.int64, device=device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 1:
            return torch.full(
                (batch_size,),
                int(tensor.reshape(-1)[0].item()),
                dtype=torch.int64,
                device=device,
            )
        tensor = tensor.reshape(batch_size, -1)
        return tensor[:, -1].to(torch.int64)

    @staticmethod
    def export_policy_info(
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        policy_info = {
            "expert_takeover": state["expert_takeover"][:, None],
            "deviation": state["deviation"][:, None],
            "deviation_count": state["deviation_count"].to(torch.float32)[:, None],
            "intervention_phase": state["intervention_phase"].to(torch.float32)[
                :, None
            ],
            "takeover_left": state["takeover_left"].to(torch.float32)[:, None],
            "takeover_used": state["takeover_used"].to(torch.float32)[:, None],
            "in_critical_phase": state["in_critical_phase"].to(torch.bool)[:, None],
            "record_transition": state["record_transition"].to(torch.bool)[:, None],
        }
        if "grasp_deviation_count" in state:
            policy_info["grasp_deviation_count"] = state[
                "grasp_deviation_count"
            ].to(torch.float32)[:, None]
        if "critical_phase_started" in state:
            policy_info["critical_phase_started"] = state[
                "critical_phase_started"
            ].to(torch.bool)[:, None]
        return policy_info

    def _states(self, mode: Literal["train", "eval"]) -> list[dict[str, torch.Tensor]]:
        return self.train_states if mode == "train" else self.eval_states

    def _maniskill_controllers(self, mode: Literal["train", "eval"]) -> list[Any | None]:
        return (
            self.train_maniskill_controllers
            if mode == "train"
            else self.eval_maniskill_controllers
        )

    @staticmethod
    def _unwrap_env(env: Any) -> Any:
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "unwrapped", env)

    @classmethod
    def _get_maniskill_hole_radii(cls, env: Any | None) -> torch.Tensor | None:
        if env is None:
            return None
        unwrapped = cls._unwrap_env(env)
        if hasattr(unwrapped, "box_hole_radii"):
            return unwrapped.box_hole_radii
        return None

    @staticmethod
    def _ensure_len(target: list, stage_id: int, fill_value: Any) -> None:
        while len(target) <= stage_id:
            target.append(
                fill_value.copy() if isinstance(fill_value, dict) else fill_value
            )
