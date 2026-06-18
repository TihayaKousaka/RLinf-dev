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

"""RLT Stage 2 policy.

This policy keeps the original Stage 2 structure:
- frozen OpenPI VLA
- frozen RL token encoder
- trainable direct Gaussian actor
- trainable twin-Q critic

The policy exposes RLinf-compatible interfaces so the existing rollout/env
pipeline can be reused. Training itself is handled by a dedicated actor worker.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .components import DirectGaussianActor, TwinQCritic, compute_td_target
from .proprio import resolve_proprio_dim
from .rl_token import RLTokenModel
from .rollout import RLTStage2RolloutRouteConfig, route_rlt_stage2_rollout
from .vla_wrapper import Stage2VLAWrapper


class RLTStage2Policy(torch.nn.Module, BasePolicy):
    ROLLOUT_SYNC_PREFIXES = ("actor.",)

    def __init__(
        self,
        cfg: DictConfig,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        stage2_cfg = cfg.rlt_stage2
        self.chunk_length = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        self.action_chunk_dim = self.chunk_length * self.action_dim
        self.proprio_mode = stage2_cfg.get("proprio_mode", None)
        if self.proprio_mode is not None:
            self.proprio_mode = str(self.proprio_mode)
        self.replay_subsample_stride = int(
            stage2_cfg.get("replay_subsample_stride", 0)
        )
        self.replay_feature_batch_size = int(
            stage2_cfg.get("replay_feature_batch_size", 0)
        )
        self.proprio_dim = resolve_proprio_dim(
            stage2_cfg,
            default_dim=self.action_dim,
        )
        self.act_as_vla_reference = bool(stage2_cfg.get("act_as_vla_reference", False))
        self.load_feature_backbones = bool(
            stage2_cfg.get("load_feature_backbones", True)
        )
        self.load_rl_token_model = bool(
            stage2_cfg.get("load_rl_token_model", self.load_feature_backbones)
        )
        self.global_step = 0
        for required_key in (
            "online_gate_updates",
            "intervention_enabled",
            "intervention_mode",
        ):
            if required_key not in stage2_cfg:
                raise ValueError(
                    "RLT Stage2 config requires "
                    f"rlt_stage2.{required_key}; do not rely on implicit rollout "
                    "defaults."
                )
        self.online_gate_updates = int(stage2_cfg.online_gate_updates)
        if self.online_gate_updates < 0:
            raise ValueError(
                "rlt_stage2.online_gate_updates must be >= 0, got "
                f"{self.online_gate_updates}."
            )
        self.intervention_enabled = bool(stage2_cfg.intervention_enabled)
        self.intervention_mode = str(stage2_cfg.intervention_mode)
        if self.intervention_enabled and self.intervention_mode not in {
            "local_correction",
            "human_override",
        }:
            raise ValueError(
                "rlt_stage2.intervention_mode must be 'local_correction' or "
                f"'human_override', got {self.intervention_mode!r}."
            )

        self.vla = None
        self.rl_token_model = None
        if self.load_feature_backbones:
            self.vla = Stage2VLAWrapper(
                model_path=cfg.model_path,
                config_name=stage2_cfg.config_name,
                norm_stats_path=stage2_cfg.get("norm_stats_path", None),
                num_images_in_input=int(stage2_cfg.get("num_images_in_input", 1)),
                num_action_chunks=self.chunk_length,
                action_dim=self.action_dim,
                num_steps=int(stage2_cfg.get("num_steps", cfg.get("num_steps", 5))),
                device=self.device,
            )

        if self.load_rl_token_model:
            self.rl_token_model = RLTokenModel(
                embedding_dim=int(stage2_cfg.get("embedding_dim", 2048)),
                encoder_layers=int(stage2_cfg.get("encoder_layers", 2)),
                encoder_heads=int(stage2_cfg.get("encoder_heads", 8)),
                decoder_layers=int(stage2_cfg.get("decoder_layers", 2)),
                decoder_heads=int(stage2_cfg.get("decoder_heads", 8)),
            ).to(self.device)
            rl_token_ckpt = torch.load(stage2_cfg.rl_token_path, map_location="cpu")
            if "model_state_dict" in rl_token_ckpt:
                rl_token_ckpt = rl_token_ckpt["model_state_dict"]
            self.rl_token_model.load_state_dict(rl_token_ckpt, strict=False)
            self.rl_token_model.eval()
            for param in self.rl_token_model.parameters():
                param.requires_grad_(False)

        embedding_dim = int(stage2_cfg.get("embedding_dim", 2048))
        self.state_dim = embedding_dim + self.proprio_dim

        self.actor = DirectGaussianActor(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=int(stage2_cfg.get("mlp_hidden_dim", 256)),
            num_hidden_layers=int(stage2_cfg.get("mlp_num_hidden_layers", 2)),
            sigma=float(stage2_cfg.get("actor_noise_sigma", 0.1)),
            ref_dropout=float(stage2_cfg.get("ref_action_dropout", 0.0)),
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        for param in self.target_actor.parameters():
            param.requires_grad_(False)

        self.critic = TwinQCritic(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=int(stage2_cfg.get("mlp_hidden_dim", 256)),
            num_hidden_layers=int(stage2_cfg.get("mlp_num_hidden_layers", 2)),
        ).to(self.device)

    @staticmethod
    def _shape_str(tensor: torch.Tensor | None) -> str:
        return "None" if tensor is None else str(tuple(getattr(tensor, "shape", ())))

    @staticmethod
    def _normalize_state_dict_key(key: str) -> str:
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    @classmethod
    def _is_rollout_sync_key(cls, key: str) -> bool:
        normalized_key = cls._normalize_state_dict_key(key)
        return any(
            normalized_key.startswith(prefix) for prefix in cls.ROLLOUT_SYNC_PREFIXES
        )

    @classmethod
    def filter_rollout_state_dict(cls, state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            normalized_key = cls._normalize_state_dict_key(key)
            if not cls._is_rollout_sync_key(normalized_key):
                continue
            if normalized_key in filtered:
                raise ValueError(
                    "Duplicate RLT Stage2 rollout sync key after normalization: "
                    f"{normalized_key}"
                )
            filtered[normalized_key] = value
        if not filtered:
            raise ValueError(
                "RLT Stage2 rollout sync state_dict is empty. Expected actor.* "
                "parameters for direct actor weight sync."
            )
        return filtered

    def rollout_state_dict(self) -> dict[str, Any]:
        return self.filter_rollout_state_dict(self.state_dict())

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def _require_feature_backbones(self, caller: str) -> None:
        if self.vla is None or self.rl_token_model is None:
            raise RuntimeError(
                f"RLT Stage2 {caller} requires VLA/RL-token feature backbones, "
                "but this policy was initialized with "
                "rlt_stage2.load_feature_backbones=False. This mode is only "
                "valid for actor/critic training on cached rollout features."
            )

    def _validate_action_chunk(self, tensor: torch.Tensor | None, *, name: str) -> None:
        expected_tail = (self.chunk_length, self.action_dim)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tuple(tensor.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"RLT Stage2 {name} shape mismatch: expected [B, "
                f"{self.chunk_length}, {self.action_dim}], got "
                f"{self._shape_str(tensor)}. Check num_action_chunks, "
                "action_dim, OpenPI action_horizon/action_env_dim, and dataset "
                "action shape."
            )

    def _validate_flat_action(self, tensor: torch.Tensor | None, *, name: str) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[1] != self.action_chunk_dim
        ):
            raise ValueError(
                f"RLT Stage2 {name} shape mismatch: expected [B, "
                f"{self.action_chunk_dim}] from "
                f"{self.chunk_length}x{self.action_dim}, got "
                f"{self._shape_str(tensor)}. Refuse to continue with an "
                "ambiguous action chunk layout."
            )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type for RLT Stage 2: {forward_type}")

    def _encode_state_and_reference(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, a_tilde_flat, _ = self._prepare_features(env_obs)
        return x, a_tilde_flat

    def _prepare_features(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._require_feature_backbones("_prepare_features")
        observation, processed_obs = self.vla.prepare_obs(env_obs)
        embeddings, pad_mask = self.vla.extract_embeddings(observation)
        z_rl = self.rl_token_model.encode(embeddings, pad_mask)
        a_tilde = self.vla.get_rl_chunk_reference(observation, self.chunk_length)
        self._validate_action_chunk(a_tilde, name="a_tilde")
        a_tilde_flat = a_tilde.reshape(a_tilde.shape[0], -1)
        self._validate_flat_action(a_tilde_flat, name="a_tilde_flat")
        state = self.vla.extract_proprio(
            observation,
            proprio_dim=self.proprio_dim,
            proprio_mode=self.proprio_mode,
        )
        x = torch.cat([z_rl.to(torch.float32), state], dim=-1)
        return x, a_tilde_flat, processed_obs

    @torch.no_grad()
    def _encode_step_trace(
        self,
        trace_obs: dict[str, Any] | None,
        *,
        tokenized_prompt: torch.Tensor | None = None,
        tokenized_prompt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(trace_obs, dict) or not trace_obs:
            return {}
        states = trace_obs.get("states", None)
        if not isinstance(states, torch.Tensor) or states.ndim < 3:
            raise ValueError(
                "RLT step trace obs must contain states with shape [B, S, ...], "
                f"got {self._shape_str(states)}."
            )
        batch_size = int(states.shape[0])
        trace_len = int(states.shape[1])
        expected_batch = batch_size * trace_len
        flat_obs: dict[str, Any] = {}
        for key, value in trace_obs.items():
            if key == "task_descriptions":
                if value is None:
                    flat_obs[key] = None
                elif isinstance(value, list):
                    flat_obs[key] = [
                        item for item in value for _ in range(trace_len)
                    ]
                else:
                    flat_obs[key] = value
                continue
            if isinstance(value, torch.Tensor):
                if value.shape[0] != batch_size or value.shape[1] != trace_len:
                    raise ValueError(
                        f"RLT step trace obs[{key!r}] must have leading shape "
                        f"[{batch_size}, {trace_len}], got {self._shape_str(value)}."
                    )
                flat_obs[key] = value.reshape(batch_size * trace_len, *value.shape[2:])
            else:
                flat_obs[key] = value
        if (tokenized_prompt is None) != (tokenized_prompt_mask is None):
            raise ValueError(
                "RLT step trace prompt reuse requires both tokenized_prompt and "
                "tokenized_prompt_mask."
            )
        if tokenized_prompt is not None and tokenized_prompt_mask is not None:
            if tokenized_prompt.shape[0] != batch_size:
                raise ValueError(
                    "RLT step trace tokenized_prompt must have batch size "
                    f"{batch_size}, got {self._shape_str(tokenized_prompt)}."
                )
            if tokenized_prompt_mask.shape[0] != batch_size:
                raise ValueError(
                    "RLT step trace tokenized_prompt_mask must have batch size "
                    f"{batch_size}, got {self._shape_str(tokenized_prompt_mask)}."
                )
            flat_obs["tokenized_prompt"] = (
                tokenized_prompt[:, None, ...]
                .expand(batch_size, trace_len, *tokenized_prompt.shape[1:])
                .reshape(expected_batch, *tokenized_prompt.shape[1:])
            )
            flat_obs["tokenized_prompt_mask"] = (
                tokenized_prompt_mask[:, None, ...]
                .expand(batch_size, trace_len, *tokenized_prompt_mask.shape[1:])
                .reshape(expected_batch, *tokenized_prompt_mask.shape[1:])
            )

        def _slice_flat_obs(start: int, end: int) -> dict[str, Any]:
            sliced: dict[str, Any] = {}
            for key, value in flat_obs.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == expected_batch:
                    sliced[key] = value[start:end]
                elif isinstance(value, list) and len(value) == expected_batch:
                    sliced[key] = value[start:end]
                else:
                    sliced[key] = value
            return sliced

        feature_batch_size = self.replay_feature_batch_size
        if feature_batch_size > 0 and expected_batch > feature_batch_size:
            x_batches: list[torch.Tensor] = []
            a_tilde_batches: list[torch.Tensor] = []
            for start in range(0, expected_batch, feature_batch_size):
                end = min(start + feature_batch_size, expected_batch)
                x_batch, a_tilde_batch, _ = self._prepare_features(
                    _slice_flat_obs(start, end)
                )
                x_batches.append(x_batch)
                a_tilde_batches.append(a_tilde_batch)
            x = torch.cat(x_batches, dim=0)
            a_tilde = torch.cat(a_tilde_batches, dim=0)
        else:
            x, a_tilde, _ = self._prepare_features(flat_obs)
        if x.shape[0] != expected_batch or a_tilde.shape[0] != expected_batch:
            raise ValueError(
                "RLT step trace feature encoder must preserve flattened batch size "
                f"{expected_batch}, got x={self._shape_str(x)} and "
                f"a_tilde={self._shape_str(a_tilde)}."
            )
        return {
            "x": x.reshape(batch_size, trace_len, -1).detach(),
            "a_tilde": a_tilde.reshape(batch_size, trace_len, -1).detach(),
        }

    @staticmethod
    def _normalize_step_record_trace(
        record_trace: torch.Tensor | None,
        *,
        fallback_record: torch.Tensor,
        chunk_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(fallback_record.shape[0])
        if not isinstance(record_trace, torch.Tensor):
            record_trace = fallback_record
        record_trace = record_trace.to(dtype=torch.bool, device=device)
        if record_trace.numel() == 1:
            return record_trace.reshape(1, 1, 1).expand(
                batch_size,
                chunk_length,
                1,
            )
        record_trace = record_trace.reshape(batch_size, -1)
        if record_trace.shape[1] == 1:
            record_trace = record_trace.expand(-1, chunk_length)
        if record_trace.shape[1] != chunk_length:
            raise ValueError(
                "RLT step trace record_transition must have shape [B], [B, 1], "
                f"[B, {chunk_length}], or [B, {chunk_length}, 1], got "
                f"{tuple(record_trace.shape)} after normalization."
            )
        return record_trace[:, :, None]

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "RLT Stage 2 does not use RLinf PPO-style default_forward."
        )

    def actor_forward(
        self,
        x: torch.Tensor,
        a_tilde: torch.Tensor,
        *,
        deterministic: bool = False,
        apply_ref_dropout: bool | None = None,
        apply_action_noise: bool | None = None,
    ) -> torch.Tensor:
        self._validate_flat_action(a_tilde, name="actor input a_tilde")
        action = self.actor(
            x,
            a_tilde,
            deterministic=deterministic,
            apply_ref_dropout=apply_ref_dropout,
            apply_action_noise=apply_action_noise,
        )
        self._validate_flat_action(action, name="actor output action_flat")
        return action

    def critic_forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_flat_action(actions, name="critic input actions")
        return self.critic(x, actions)

    def critic_min(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self._validate_flat_action(actions, name="critic_min input actions")
        return self.critic.q_min(x, actions)

    @torch.no_grad()
    def compute_td_target_batch(
        self,
        *,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_x: torch.Tensor,
        next_a_tilde: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_flat_action(
            next_a_tilde,
            name="compute_td_target_batch next_a_tilde",
        )
        stage2_cfg = self.cfg.rlt_stage2
        return compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=next_a_tilde,
            target_actor=self.target_actor,
            critic=self.critic,
            gamma=float(stage2_cfg.get("gamma", self.cfg.get("gamma", 0.99))),
            chunk_length=self.chunk_length,
        )

    @torch.no_grad()
    def update_target_networks(self, tau: float) -> None:
        for online_param, target_param in zip(
            self.actor.parameters(),
            self.target_actor.parameters(),
            strict=True,
        ):
            target_param.data.lerp_(online_param.data, tau)
        self.critic.update_targets(tau)

    def set_online_critic_requires_grad(self, requires_grad: bool) -> None:
        for module in (self.critic.q1, self.critic.q2):
            for param in module.parameters():
                param.requires_grad_(requires_grad)

    @torch.no_grad()
    def encode_obs(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_state_and_reference(env_obs)

    @torch.no_grad()
    def predict_vla_reference_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del mode, kwargs
        if self.vla is None:
            raise RuntimeError(
                "RLT Stage2 VLA reference prediction requires the VLA backbone. "
                "Do not call this method on actor-only training policies."
            )
        observation, processed_obs = self.vla.prepare_obs(env_obs)
        a_tilde = self.vla.get_rl_chunk_reference(observation, self.chunk_length)
        self._validate_action_chunk(a_tilde, name="expert vla_reference actions")
        action_flat = a_tilde.reshape(a_tilde.shape[0], -1)
        self._validate_flat_action(action_flat, name="expert vla_reference action_flat")
        zeros = torch.zeros(
            action_flat.shape[0], 1, device=action_flat.device, dtype=torch.float32
        )
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {
                "action": action_flat.detach(),
                "a_tilde": action_flat.detach(),
                "tokenized_prompt": processed_obs["tokenized_prompt"].detach(),
                "tokenized_prompt_mask": processed_obs[
                    "tokenized_prompt_mask"
                ].detach(),
            },
        }
        return a_tilde, result

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        env_infos: dict[str, Any] | None = None,
        allow_expert: bool = True,
        expert_model_getter=None,
        enable_rlt_route: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs

        x, a_tilde, processed_obs = self._prepare_features(env_obs)
        deterministic = mode == "eval"
        if deterministic or self.act_as_vla_reference:
            self.actor.eval()
        if self.act_as_vla_reference:
            action_flat = a_tilde
        else:
            action_flat = self.actor_forward(
                x,
                a_tilde,
                deterministic=deterministic,
            )
        actions = action_flat.reshape(
            action_flat.shape[0],
            self.chunk_length,
            self.action_dim,
        )
        self._validate_action_chunk(actions, name="predict_action_batch actions")
        zeros = torch.zeros(
            action_flat.shape[0], 1, device=action_flat.device, dtype=torch.float32
        )
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {
                "action": action_flat.detach(),
                "x": x.detach(),
                "a_tilde": a_tilde.detach(),
                "tokenized_prompt": processed_obs["tokenized_prompt"].detach(),
                "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"].detach(),
            },
        }
        if not enable_rlt_route:
            return actions, result

        policy_info = None
        if isinstance(env_infos, dict) and isinstance(env_infos.get("policy_info"), dict):
            policy_info = env_infos["policy_info"]
        ready_for_online = self.global_step >= self.online_gate_updates
        route = route_rlt_stage2_rollout(
            env_obs=env_obs,
            policy_info=policy_info,
            student_model=self,
            expert_model_getter=expert_model_getter,
            model_kwargs={"mode": mode, "enable_rlt_route": False},
            cfg=RLTStage2RolloutRouteConfig(
                ready_for_online=ready_for_online,
                online_gate_step=self.online_gate_updates,
                intervention_enabled=self.intervention_enabled,
                allow_expert=(
                    mode == "train"
                    and allow_expert
                    and expert_model_getter is not None
                    and self.intervention_enabled
                ),
                chunk_length=self.chunk_length,
                action_dim=self.action_dim,
            ),
            student_prediction=(actions, result),
        )
        route.result["expert_label_flag"] = route.expert_label_flag
        forward_inputs = route.result["forward_inputs"]
        if self.replay_subsample_stride > 0 and isinstance(env_infos, dict):
            encoded_step_trace = self._encode_step_trace(
                env_infos.get("rlt_step_trace_obs", None),
                tokenized_prompt=processed_obs["tokenized_prompt"],
                tokenized_prompt_mask=processed_obs["tokenized_prompt_mask"],
            )
            if encoded_step_trace and encoded_step_trace["x"].shape[1] != self.chunk_length:
                raise ValueError(
                    "RLT step trace length must match chunk_length "
                    f"{self.chunk_length}, got {encoded_step_trace['x'].shape[1]}."
                )
            if encoded_step_trace:
                forward_inputs["rlt_step_trace"] = encoded_step_trace
                forward_inputs["rlt_step_trace_valid"] = torch.ones(
                    (action_flat.shape[0], 1),
                    dtype=torch.bool,
                    device=action_flat.device,
                )
            trace_infos = env_infos.get("rlt_step_trace_infos", None)
            policy_trace_infos = (
                trace_infos.get("policy_info", {})
                if isinstance(trace_infos, dict)
                else {}
            )
            record_trace = (
                policy_trace_infos.get("record_transition")
                if isinstance(policy_trace_infos, dict)
                else None
            )
            record_trace = self._normalize_step_record_trace(
                record_trace,
                fallback_record=forward_inputs["record_transition"],
                chunk_length=self.chunk_length,
                device=action_flat.device,
            )
            forward_inputs["rlt_step_trace_infos"] = {
                "policy_info": {
                    "record_transition": record_trace.detach(),
                },
            }
        if self.replay_subsample_stride > 0 and "rlt_step_trace" not in forward_inputs:
            # The first rollout step has no previous chunk trace. Keep a typed
            # placeholder so trajectory stacking preserves the trace schema; the
            # adapter only consumes entries marked valid.
            forward_inputs["rlt_step_trace"] = {
                "x": x[:, None, :].expand(-1, self.chunk_length, -1).detach(),
                "a_tilde": a_tilde[:, None, :].expand(
                    -1,
                    self.chunk_length,
                    -1,
                ).detach(),
            }
            forward_inputs["rlt_step_trace_valid"] = torch.zeros(
                (action_flat.shape[0], 1),
                dtype=torch.bool,
                device=action_flat.device,
            )
        if (
            self.replay_subsample_stride > 0
            and "rlt_step_trace_infos" not in forward_inputs
        ):
            record_trace = self._normalize_step_record_trace(
                None,
                fallback_record=forward_inputs["record_transition"],
                chunk_length=self.chunk_length,
                device=action_flat.device,
            )
            forward_inputs["rlt_step_trace_infos"] = {
                "policy_info": {
                    "record_transition": record_trace.detach(),
                },
            }
        return route.actions, route.result
