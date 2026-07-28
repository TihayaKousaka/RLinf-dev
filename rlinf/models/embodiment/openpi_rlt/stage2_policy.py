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
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.openpi import build_openpi_rlt_backbone
from rlinf.utils.pytree import register_pytree_dataclasses

from .components import (
    DirectGaussianActor,
    MultiQCritic,
    TwinQCritic,
    compute_td_target,
)
from .proprio import resolve_proprio_dim, select_proprio
from .rl_token import RLTokenModel
from .rollout import RLTStage2RolloutRouteConfig, route_rlt_stage2_rollout


class RLTStage2Policy(torch.nn.Module, BasePolicy):
    ROLLOUT_SYNC_PREFIXES = ("actor.", "critic.q1.", "critic.q2.", "critic.q_heads.")
    accepts_rollout_context = True

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
        if "proprio_dim" not in stage2_cfg:
            raise ValueError(
                "RLT Stage2 requires rlt_stage2.proprio_dim to match the full "
                "environment state dimension."
            )
        self.proprio_dim = resolve_proprio_dim(
            default_dim=int(stage2_cfg.proprio_dim),
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

        self.policy_mode = str(stage2_cfg.get("policy_mode", "td3")).lower()
        if self.policy_mode not in {"td3", "expo_ft"}:
            raise ValueError(
                "rlt_stage2.policy_mode must be 'td3' or 'expo_ft', got "
                f"{self.policy_mode!r}."
            )
        self.num_q_heads = int(stage2_cfg.get("num_q_heads", 10))
        self.num_base_candidates = int(stage2_cfg.get("num_base_candidates", 8))
        self.num_edit_samples = int(
            stage2_cfg.get("num_edit_samples", self.num_base_candidates)
        )
        self.num_min_qs = int(stage2_cfg.get("num_min_qs", 2))
        if self.num_q_heads <= 0:
            raise ValueError(
                f"rlt_stage2.num_q_heads must be positive, got {self.num_q_heads}."
            )
        if self.num_base_candidates <= 0:
            raise ValueError(
                "rlt_stage2.num_base_candidates must be positive, got "
                f"{self.num_base_candidates}."
            )
        if self.num_edit_samples < 0:
            raise ValueError(
                "rlt_stage2.num_edit_samples must be non-negative, got "
                f"{self.num_edit_samples}."
            )
        if self.num_min_qs <= 0 or self.num_min_qs > self.num_q_heads:
            raise ValueError(
                "rlt_stage2.num_min_qs must be in [1, num_q_heads], got "
                f"{self.num_min_qs} for {self.num_q_heads} heads."
            )

        self.vla = None
        self.rl_token_model = None
        if self.load_feature_backbones:
            self.vla = build_openpi_rlt_backbone(
                model_path=cfg.model_path,
                config_name=stage2_cfg.config_name,
                num_images_in_input=int(stage2_cfg.get("num_images_in_input", 1)),
                num_action_chunks=self.chunk_length,
                action_dim=self.action_dim,
                num_steps=int(stage2_cfg.get("num_steps", cfg.get("num_steps", 5))),
                device=self.device,
                freeze=True,
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

        actor_hidden_dim = int(stage2_cfg.get("mlp_hidden_dim", 256))
        actor_num_hidden_layers = int(stage2_cfg.get("mlp_num_hidden_layers", 2))
        ref_dropout = float(stage2_cfg.get("ref_action_dropout", 0.0))
        self.actor = DirectGaussianActor(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=actor_hidden_dim,
            num_hidden_layers=actor_num_hidden_layers,
            sigma=float(stage2_cfg.get("actor_noise_sigma", 0.1)),
            ref_dropout=ref_dropout,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        for param in self.target_actor.parameters():
            param.requires_grad_(False)

        if self.policy_mode == "expo_ft":
            self.critic = MultiQCritic(
                state_dim=self.state_dim,
                action_chunk_dim=self.action_chunk_dim,
                hidden_dim=actor_hidden_dim,
                num_hidden_layers=actor_num_hidden_layers,
                num_q_heads=self.num_q_heads,
            ).to(self.device)
        else:
            self.critic = TwinQCritic(
                state_dim=self.state_dim,
                action_chunk_dim=self.action_chunk_dim,
                hidden_dim=actor_hidden_dim,
                num_hidden_layers=actor_num_hidden_layers,
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
                "or online critic parameters for direct actor weight sync."
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
        raise NotImplementedError(
            f"Unsupported forward_type for RLT Stage 2: {forward_type}"
        )

    def _encode_state_and_reference(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, x, a_tilde_flat = self._prepare_features(env_obs)
        return x, a_tilde_flat

    def _prepare_features(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[Any, torch.Tensor, torch.Tensor]:
        self._require_feature_backbones("_prepare_features")
        observation, _ = self.vla.prepare_rlt_observation(env_obs)
        embeddings, pad_mask = self.vla.extract_rlt_prefix_embeddings(
            observation,
            dtype=torch.float32,
        )
        z_rl = self.rl_token_model.encode(embeddings, pad_mask)
        a_tilde = self.vla.predict_rlt_reference_action(observation, self.chunk_length)
        self._validate_action_chunk(a_tilde, name="a_tilde")
        a_tilde_flat = a_tilde.reshape(a_tilde.shape[0], -1)
        self._validate_flat_action(a_tilde_flat, name="a_tilde_flat")
        try:
            state = select_proprio(
                observation.state,
                proprio_dim=self.proprio_dim,
            ).to(device=self.device)
        except ValueError as exc:
            raise ValueError(
                "RLT Stage2 full-state proprio dimension mismatch: "
                f"config proprio_dim={self.proprio_dim}, "
                f"observation.state dim={observation.state.shape[1]}."
            ) from exc
        x = torch.cat([z_rl.to(torch.float32), state], dim=-1)
        return observation, x, a_tilde_flat

    def _normalize_base_chunks(
        self,
        base_chunks: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if base_chunks.ndim == 2:
            base_chunks = base_chunks[:, None, :]
        if (
            not isinstance(base_chunks, torch.Tensor)
            or base_chunks.ndim != 3
            or base_chunks.shape[-1] != self.action_chunk_dim
        ):
            raise ValueError(
                f"RLT Stage2 {name} shape mismatch: expected [B, N, "
                f"{self.action_chunk_dim}], got {self._shape_str(base_chunks)}."
            )
        return base_chunks

    @staticmethod
    def _repeat_batch_tree(tree: Any, repeats: int, batch_size: int) -> Any:
        register_pytree_dataclasses(tree)
        return tree_map(
            lambda value: (
                value.repeat_interleave(repeats, dim=0)
                if torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == int(batch_size)
                else value
            ),
            tree,
        )

    @torch.no_grad()
    def _sample_base_chunks(self, observation: Any) -> torch.Tensor:
        self._require_feature_backbones("_sample_base_chunks")
        batch_size = int(observation.state.shape[0])
        repeated_observation = self._repeat_batch_tree(
            observation,
            self.num_base_candidates,
            batch_size,
        )
        outputs = self.vla.sample_actions(
            repeated_observation,
            mode="eval",
            compute_values=False,
        )
        action_dict = self.vla.output_transform(
            {"actions": outputs["actions"], "state": repeated_observation.state}
        )
        base_chunks = action_dict["actions"].to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(batch_size, self.num_base_candidates, -1)
        return self._normalize_base_chunks(base_chunks, name="base_chunks")

    def sample_q_indices(self, mode: str) -> torch.Tensor | None:
        if self.policy_mode != "expo_ft":
            return None
        if self.num_min_qs >= self.num_q_heads:
            return None
        if mode == "eval":
            return torch.arange(self.num_min_qs, device=self.device)
        return torch.randperm(self.num_q_heads, device=self.device)[: self.num_min_qs]

    def _flatten_candidate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None]:
        if actions.ndim == 2:
            self._validate_flat_action(actions, name="critic actions")
            return x, actions, None
        if actions.ndim != 3 or actions.shape[-1] != self.action_chunk_dim:
            raise ValueError(
                "RLT Stage2 critic actions must have shape [B, A] or [B, N, A], "
                f"got {self._shape_str(actions)}."
            )
        batch_size, num_candidates, _ = actions.shape
        repeated_x = x[:, None, :].expand(-1, num_candidates, -1).reshape(
            batch_size * num_candidates,
            -1,
        )
        repeated_actions = actions.reshape(batch_size * num_candidates, -1)
        return repeated_x, repeated_actions, (batch_size, num_candidates)

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
        use_target: bool = False,
    ) -> torch.Tensor:
        self._validate_flat_action(a_tilde, name="actor input a_tilde")
        actor = self.target_actor if use_target else self.actor
        action = actor(
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
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.policy_mode == "expo_ft":
            return self.critic_values_forward(x, actions, use_target=False)
        self._validate_flat_action(actions, name="critic input actions")
        return self.critic(x, actions)

    def critic_values_forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        *,
        use_target: bool = False,
        q_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.policy_mode != "expo_ft":
            raise RuntimeError("critic_values_forward is only used by EXPO-FT.")
        x, actions, candidate_shape = self._flatten_candidate_actions(x, actions)
        critic = self.critic.target_q_values if use_target else self.critic.forward
        q_values = critic(x, actions)
        if candidate_shape is not None:
            batch_size, num_candidates = candidate_shape
            q_values = q_values.reshape(batch_size, num_candidates, -1)
        if q_indices is not None:
            q_values = q_values.index_select(dim=-1, index=q_indices)
        return q_values

    def critic_min(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        *,
        use_target: bool = False,
        q_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.policy_mode == "expo_ft":
            q_values = self.critic_values_forward(
                x,
                actions,
                use_target=use_target,
                q_indices=q_indices,
            )
            return q_values.min(dim=-1, keepdim=True).values
        self._validate_flat_action(actions, name="critic_min input actions")
        if use_target:
            return self.critic.target_q_min(x, actions)
        return self.critic.q_min(x, actions)

    @torch.no_grad()
    def _best_of_n_actor_forward(
        self,
        x: torch.Tensor,
        a_tilde: torch.Tensor,
        *,
        best_of_n: int,
    ) -> torch.Tensor:
        if best_of_n <= 1:
            return self.actor_forward(x, a_tilde, deterministic=False)

        batch_size = int(x.shape[0])
        repeated_x = x.repeat_interleave(best_of_n, dim=0)
        repeated_a_tilde = a_tilde.repeat_interleave(best_of_n, dim=0)
        candidate_actions = self.actor_forward(
            repeated_x,
            repeated_a_tilde,
            deterministic=False,
            apply_ref_dropout=False,
            apply_action_noise=True,
        )
        q_values = self.critic_min(repeated_x, candidate_actions).reshape(
            batch_size,
            best_of_n,
        )
        best_indices = q_values.argmax(dim=1)
        candidate_actions = candidate_actions.reshape(
            batch_size,
            best_of_n,
            self.action_chunk_dim,
        )
        return candidate_actions[
            torch.arange(batch_size, device=candidate_actions.device),
            best_indices,
        ]

    @torch.no_grad()
    def select_top_q_actions(
        self,
        x: torch.Tensor,
        base_chunks: torch.Tensor,
        *,
        q_indices: torch.Tensor | None = None,
        use_target: bool = False,
        deterministic_candidates: bool = False,
    ) -> torch.Tensor:
        base_chunks = self._normalize_base_chunks(base_chunks, name="base_chunks")
        batch_size, num_base_candidates, _ = base_chunks.shape
        num_edit_candidates = min(self.num_edit_samples, num_base_candidates)

        candidate_chunks = [base_chunks]
        if num_edit_candidates > 0:
            edit_base_chunks = base_chunks[:, :num_edit_candidates, :]
            repeated_x = x[:, None, :].expand(-1, num_edit_candidates, -1).reshape(
                batch_size * num_edit_candidates,
                -1,
            )
            repeated_base = edit_base_chunks.reshape(
                batch_size * num_edit_candidates,
                -1,
            )
            edited_chunks = self.actor_forward(
                repeated_x,
                repeated_base,
                deterministic=deterministic_candidates,
                apply_ref_dropout=False,
                apply_action_noise=not deterministic_candidates,
                use_target=use_target,
            ).reshape(batch_size, num_edit_candidates, -1)
            candidate_chunks.append(edited_chunks)

        all_candidates = torch.cat(candidate_chunks, dim=1)
        candidate_scores = self.critic_values_forward(
            x,
            all_candidates,
            use_target=use_target,
            q_indices=q_indices,
        ).min(dim=-1).values
        selected_indices = candidate_scores.argmax(dim=1)
        return all_candidates[
            torch.arange(batch_size, device=all_candidates.device),
            selected_indices,
        ]

    @torch.no_grad()
    def compute_td_target_batch(
        self,
        *,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_x: torch.Tensor,
        next_a_tilde: torch.Tensor,
        next_base_chunks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stage2_cfg = self.cfg.rlt_stage2
        gamma = float(stage2_cfg.get("gamma", self.cfg.get("gamma", 0.99)))
        if self.policy_mode != "expo_ft":
            self._validate_flat_action(
                next_a_tilde,
                name="compute_td_target_batch next_a_tilde",
            )
            return compute_td_target(
                rewards=rewards,
                dones=dones,
                next_x=next_x,
                next_a_tilde=next_a_tilde,
                target_actor=self.target_actor,
                critic=self.critic,
                gamma=gamma,
                chunk_length=self.chunk_length,
            )

        self._validate_flat_action(
            next_a_tilde,
            name="compute_td_target_batch next_a_tilde",
        )
        if next_base_chunks is None:
            next_base_chunks = next_a_tilde[:, None, :]
        next_base_chunks = self._normalize_base_chunks(
            next_base_chunks,
            name="compute_td_target_batch next_base_chunks",
        )
        q_indices = self.sample_q_indices(mode="train")
        next_actions = self.select_top_q_actions(
            next_x,
            next_base_chunks,
            q_indices=q_indices,
            use_target=True,
            deterministic_candidates=False,
        )
        next_q = self.critic_min(
            next_x,
            next_actions,
            use_target=True,
            q_indices=q_indices,
        )
        discount_powers = gamma ** torch.arange(
            self.chunk_length, device=rewards.device, dtype=rewards.dtype
        )
        chunk_return = (rewards * discount_powers).sum(dim=-1, keepdim=True)
        bootstrap = (gamma**self.chunk_length) * (1.0 - dones) * next_q
        return chunk_return + bootstrap

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
        if self.policy_mode == "expo_ft":
            critic_modules = self.critic.q_heads
        else:
            critic_modules = (self.critic.q1, self.critic.q2)
        for module in critic_modules:
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
        observation, _ = self.vla.prepare_rlt_observation(env_obs)
        a_tilde = self.vla.predict_rlt_reference_action(observation, self.chunk_length)
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
            },
        }
        return a_tilde, result

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        env_cfg: Any | None = None,
        env_infos: dict[str, Any] | None = None,
        allow_expert: bool = True,
        expert_model_getter=None,
        enable_rlt_route: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs

        observation, x, a_tilde = self._prepare_features(env_obs)
        deterministic = mode == "eval"
        ready_for_online = self.global_step >= self.online_gate_updates
        is_maniskill_train = (
            mode == "train"
            and env_cfg is not None
            and str(env_cfg.get("env_type", "")) == "maniskill"
        )
        if deterministic or self.act_as_vla_reference:
            self.actor.eval()
        base_chunks = None
        if self.act_as_vla_reference:
            action_flat = a_tilde
        elif self.policy_mode == "expo_ft":
            base_chunks = self._sample_base_chunks(observation)
            q_indices = self.sample_q_indices(mode)
            action_flat = self.select_top_q_actions(
                x,
                base_chunks,
                q_indices=q_indices,
                deterministic_candidates=deterministic,
            )
        elif deterministic:
            action_flat = self.actor_forward(
                x,
                a_tilde,
                deterministic=True,
            )
        else:
            best_of_n = (
                max(1, int(self.cfg.rlt_stage2.get("best_of_n", 1)))
                if is_maniskill_train and ready_for_online and not deterministic
                else 1
            )
            action_flat = self._best_of_n_actor_forward(
                x,
                a_tilde,
                best_of_n=best_of_n,
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
            },
        }
        if base_chunks is not None:
            result["forward_inputs"]["base_chunks"] = base_chunks.detach()
        if not enable_rlt_route:
            return actions, result

        policy_info = None
        if isinstance(env_infos, dict) and isinstance(
            env_infos.get("policy_info"),
            dict,
        ):
            policy_info = env_infos["policy_info"]
        is_realworld_train = (
            mode == "train"
            and env_cfg is not None
            and str(env_cfg.get("env_type", "")) == "realworld"
        )
        route = route_rlt_stage2_rollout(
            env_obs=env_obs,
            policy_info=policy_info,
            student_model=self,
            expert_model_getter=expert_model_getter,
            model_kwargs={"mode": mode, "enable_rlt_route": False},
            cfg=RLTStage2RolloutRouteConfig(
                ready_for_online=ready_for_online,
                online_gate_step=self.online_gate_updates,
                allow_expert=(
                    mode == "train"
                    and allow_expert
                    and expert_model_getter is not None
                    and self.intervention_enabled
                ),
                chunk_length=self.chunk_length,
                action_dim=self.action_dim,
                in_critical_phase_default=not is_realworld_train,
                record_transition_default=not is_realworld_train,
            ),
            student_prediction=(actions, result),
        )
        route.result["expert_label_flag"] = route.expert_label_flag
        return route.actions, route.result

    def get_rollout_policy_mode(
        self,
        *,
        mode: Literal["train", "eval"],
        env_cfg: Any | None = None,
    ) -> str:
        """Return the policy mode used by the generic HF rollout worker."""
        if mode == "eval" and env_cfg is not None:
            return str(env_cfg.get("policy_mode", mode))
        return mode
