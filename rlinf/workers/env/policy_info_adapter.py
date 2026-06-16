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

"""Policy-info adapter factory for env workers."""

from __future__ import annotations

from typing import Any, Literal

import torch


class NoopPolicyInfoAdapter:
    """Default adapter used when an algorithm does not emit policy_info."""

    def init_stage(self, **kwargs: Any):
        return None

    def update_stage(self, **kwargs: Any):
        return None

    def update_last_action_metadata(self, **kwargs: Any) -> None:
        return None


class RealworldPolicyInfoAdapter:
    """Env-side policy_info adapter for realworld RLT."""

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

    def init_stage(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        del env
        if not self.enabled(mode):
            return None
        batch_size = self._batch_size(mode)
        state = self._init_state(batch_size=batch_size, mode=mode)
        states = self._states(mode)
        self._ensure_len(states, stage_id, {})
        states[stage_id] = state
        return self._export_policy_info(state)

    def update_stage(
        self,
        *,
        infos: dict[str, Any] | None,
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        del env
        states = self._states(mode)
        if not self.enabled(mode) or infos is None or stage_id >= len(states):
            return None
        state = states[stage_id]
        device = chunk_dones.device
        done_any = chunk_dones.any(dim=1).to(device)
        batch_size = int(done_any.shape[0])
        for key, value in state.items():
            state[key] = value.to(device)

        expert_takeover = self._coerce_bool_info(
            self._lookup_policy_info_value(infos, "expert_takeover"),
            batch_size=batch_size,
            device=device,
        )
        deviation = self._coerce_bool_info(
            self._lookup_policy_info_value(infos, "deviation"),
            batch_size=batch_size,
            device=device,
        )
        intervention_region = self._coerce_bool_info(
            self._lookup_policy_info_value(infos, "intervention_region"),
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
                self._lookup_policy_info_value(infos, key),
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
                    self._lookup_policy_info_value(infos, key),
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
                self._lookup_policy_info_value(infos, "intervention_phase"),
                batch_size=batch_size,
                device=device,
            ),
        )
        states[stage_id] = state
        return self._export_policy_info(state)

    def update_last_action_metadata(self, **kwargs: Any) -> None:
        del kwargs
        return None

    def enabled(self, mode: Literal["train", "eval"]) -> bool:
        return self.td3_enabled() and self.env_type(mode) == "realworld"

    def td3_enabled(self) -> bool:
        return (
            self.cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and self.cfg.actor.model.get("model_type", None) == "rlt_stage2"
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

    def _init_state(
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

    @staticmethod
    def _lookup_policy_info_value(infos: dict[str, Any], key: str) -> Any:
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
    def _export_policy_info(
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        policy_info = {
            "expert_takeover": state["expert_takeover"][:, None],
            "deviation": state["deviation"][:, None],
            "deviation_count": state["deviation_count"].to(torch.float32)[:, None],
            "grasp_deviation_count": state["grasp_deviation_count"].to(torch.float32)[
                :, None
            ],
            "intervention_phase": state["intervention_phase"].to(torch.float32)[
                :, None
            ],
            "takeover_left": state["takeover_left"].to(torch.float32)[:, None],
            "takeover_used": state["takeover_used"].to(torch.float32)[:, None],
            "in_critical_phase": state["in_critical_phase"].to(torch.bool)[:, None],
            "record_transition": state["record_transition"].to(torch.bool)[:, None],
            "critical_phase_started": state["critical_phase_started"].to(torch.bool)[
                :, None
            ],
        }
        return policy_info

    def _states(self, mode: Literal["train", "eval"]) -> list[dict[str, torch.Tensor]]:
        return self.train_states if mode == "train" else self.eval_states

    @staticmethod
    def _ensure_len(target: list, stage_id: int, fill_value: Any) -> None:
        while len(target) <= stage_id:
            target.append(
                fill_value.copy() if isinstance(fill_value, dict) else fill_value
            )


def build_policy_info_adapter(cfg, train_batch_size, eval_batch_size):
    """Build an env-side policy_info adapter for algorithms that need one."""
    intervention_cfg = cfg.algorithm.get("intervention", {})
    is_rlt_stage2 = (
        cfg.algorithm.get("loss_type", None) == "rlt_td3"
        and cfg.actor.model.get("model_type", None) == "rlt_stage2"
    )
    mode = str(intervention_cfg.get("mode", "local_correction"))
    if not is_rlt_stage2 or not bool(intervention_cfg.get("enable", False)):
        return NoopPolicyInfoAdapter()

    train_env_type = str(cfg.env.get("train", {}).get("env_type", "")).lower()
    eval_env_type = str(cfg.env.eval.get("env_type", "")).lower()
    if (train_env_type == "maniskill" or eval_env_type == "maniskill") and (
        mode == "local_correction"
    ):
        from rlinf.envs.maniskill.rlt_policy_info import RLTStage2PolicyInfoAdapter

        return RLTStage2PolicyInfoAdapter(
            cfg=cfg,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
    if (train_env_type == "realworld" or eval_env_type == "realworld") and (
        mode in {"local_correction", "human_override"}
    ):
        return RealworldPolicyInfoAdapter(
            cfg=cfg,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
    return NoopPolicyInfoAdapter()
