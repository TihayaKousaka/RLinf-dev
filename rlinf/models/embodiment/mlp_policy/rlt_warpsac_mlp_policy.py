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
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class WarpBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d with the per-call training override used by WarpSAC."""

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [B, C] input, got {tuple(x.shape)}")
        if training is None:
            training = self.training
        use_batch_stats = bool(training) or not self.track_running_stats
        if use_batch_stats and x.shape[0] <= 1:
            raise ValueError(
                "WarpSAC BatchNorm requires more than one sample in training mode."
            )
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            use_batch_stats,
            self.momentum,
            self.eps,
        )


class WarpLinear(nn.Linear):
    """Orthogonally initialized linear layer from the FlashSAC backbone."""

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class WarpSACBlock(nn.Module):
    """FlashSAC residual MLP block."""

    def __init__(
        self, hidden_dim: int, expansion: int = 4, use_bias: bool = False
    ) -> None:
        super().__init__()
        expanded_dim = hidden_dim * expansion
        self.w1 = WarpLinear(hidden_dim, expanded_dim, bias=use_bias)
        self.w2 = WarpLinear(expanded_dim, hidden_dim, bias=use_bias)
        self.norm1 = WarpBatchNorm1d(expanded_dim)
        self.norm2 = WarpBatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.w1(x), training=training))
        x = F.relu(self.norm2(self.w2(x), training=training))
        return residual + x


class WarpSACEncoder(nn.Module):
    """FlashSAC input embedding, residual blocks and RMS normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = WarpBatchNorm1d(input_dim)
        self.input_proj = WarpLinear(input_dim, hidden_dim, bias=use_bias)
        self.blocks = nn.ModuleList(
            [WarpSACBlock(hidden_dim, use_bias=use_bias) for _ in range(num_blocks)]
        )
        self.output_norm = nn.RMSNorm(hidden_dim, eps=1.0e-6)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        x = self.input_proj(self.input_norm(x, training=training))
        for block in self.blocks:
            x = block(x, training=training)
        return self.output_norm(x)


class WarpSACActor(nn.Module):
    """Tanh-squashed diagonal Gaussian actor used by WarpSAC."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        # Keep the actor feature extractor out of the generic ``encoder``
        # critic-parameter filter used by EmbodiedSACFSDPPolicy.
        self.backbone = WarpSACEncoder(
            input_dim, hidden_dim, num_blocks, use_bias=use_bias
        )
        self.fc_mean = WarpLinear(hidden_dim, action_dim, bias=use_bias)
        self.fc_log_std = WarpLinear(hidden_dim, action_dim, bias=use_bias)

    def forward(
        self, x: torch.Tensor, training: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x, training=training)
        return self.fc_mean(features), self.fc_log_std(features)

    def _transform_log_std(self, raw_log_std: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(raw_log_std)
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            bounded + 1.0
        )

    def sample_and_log_prob(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self(x, training=training)
        log_std = self._transform_log_std(raw_log_std)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
            log_prob = torch.zeros_like(mean)
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + std * noise
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(pre_tanh)
            log_prob = log_prob - 2.0 * (
                pre_tanh.new_tensor(2.0).log() - pre_tanh - F.softplus(-2.0 * pre_tanh)
            )
        return pre_tanh.tanh(), log_prob


class WarpSACQNetwork(nn.Module):
    """Single scalar Q network with the FlashSAC residual backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = WarpSACEncoder(
            input_dim, hidden_dim, num_blocks, use_bias=use_bias
        )
        self.output = WarpLinear(hidden_dim, 1, bias=use_bias)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        return self.output(self.encoder(x, training=training)).squeeze(-1)


class WarpSACTwinQCritic(nn.Module):
    """Twin scalar Q critic used by the faithful Stage-5 learner."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        input_dim = int(state_dim) + int(action_dim)
        self.q1 = WarpSACQNetwork(input_dim, hidden_dim, num_blocks, use_bias)
        self.q2 = WarpSACQNetwork(input_dim, hidden_dim, num_blocks, use_bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        training: bool | None = None,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return torch.stack(
            [self.q1(x, training=training), self.q2(x, training=training)], dim=-1
        )


class RLTWarpSACMLPPolicy(nn.Module, BasePolicy):
    """RLT policy with the stochastic actor and FlashSAC-style networks."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        actor_hidden_dim: int = 128,
        actor_num_blocks: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_blocks: int = 2,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        if not add_q_head:
            raise ValueError("RLTWarpSACMLPPolicy requires add_q_head=True.")
        if q_head_type != "default":
            raise ValueError(
                "RLTWarpSACMLPPolicy only supports q_head_type='default', got "
                f"{q_head_type!r}."
            )

        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.step_action_dim = int(action_dim)
        self.chunk_len = int(num_action_chunks)
        self.ref_chunk_len = (
            self.chunk_len
            if ref_num_action_chunks is None
            else int(ref_num_action_chunks)
        )
        if self.ref_chunk_len < self.chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{self.ref_chunk_len} < {self.chunk_len}."
            )

        self.action_dim = self.step_action_dim
        self.num_action_chunks = self.chunk_len
        self.flat_action_dim = self.chunk_len * self.step_action_dim
        self.state_dim = self.z_dim + self.proprio_dim
        self.actor = WarpSACActor(
            self.state_dim + self.flat_action_dim,
            self.flat_action_dim,
            hidden_dim=actor_hidden_dim,
            num_blocks=actor_num_blocks,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            use_bias=use_bias,
        )
        self.q_head = WarpSACTwinQCritic(
            self.state_dim,
            self.flat_action_dim,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            use_bias=use_bias,
        )
        self.torch_compile_enabled = False

    def preprocess_env_obs(self, env_obs: dict) -> dict:
        device = next(self.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _state(self, obs: dict) -> torch.Tensor:
        return torch.cat(
            [self._flatten_batch(obs["z_rl"]), self._flatten_batch(obs["proprio"])],
            dim=-1,
        )

    def _ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        return ref_chunk[:, : self.chunk_len].reshape(ref_chunk.shape[0], -1)

    @staticmethod
    def _drop_reference(ref_chunk: torch.Tensor, dropout_prob: float) -> torch.Tensor:
        if dropout_prob <= 0.0:
            return ref_chunk
        keep = torch.rand((ref_chunk.shape[0], 1), device=ref_chunk.device) >= float(
            dropout_prob
        )
        return ref_chunk * keep.to(dtype=ref_chunk.dtype)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "RLTWarpSACMLPPolicy does not use PPO-style default_forward."
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ_Q:
            return self.crossq_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def sac_forward(
        self,
        obs: dict,
        *,
        deterministic: bool = False,
        training: bool | None = None,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del kwargs
        ref_chunk = self._ref_chunk(obs)
        if apply_reference_dropout:
            ref_chunk = self._drop_reference(ref_chunk, reference_dropout_prob)
        actor_input = torch.cat([self._state(obs), ref_chunk], dim=-1)
        action, log_prob = self.actor.sample_and_log_prob(
            actor_input, deterministic=deterministic, training=training
        )
        return action, log_prob, None

    def sac_q_forward(
        self,
        obs: dict,
        actions: torch.Tensor,
        *,
        detach_encoder: bool = False,
        training: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        state = self._state(obs)
        if detach_encoder:
            state = state.detach()
        return self.q_head(
            state,
            self._flatten_batch(actions),
            training=training,
        )

    def crossq_q_forward(
        self,
        obs: dict,
        actions: torch.Tensor,
        next_obs: dict | None = None,
        next_actions: torch.Tensor | None = None,
        detach_encoder: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        data_q = self.sac_q_forward(
            obs=obs,
            actions=actions,
            detach_encoder=detach_encoder,
        )
        if next_obs is None or next_actions is None:
            return data_q, data_q.new_zeros(data_q.shape)
        next_q = self.sac_q_forward(
            obs=next_obs,
            actions=next_actions,
            detach_encoder=detach_encoder,
        )
        return data_q, next_q

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs)
        action, chunk_logprobs, _ = self.sac_forward(
            obs,
            deterministic=(mode == "eval"),
            training=False,
        )
        chunk_actions = self._format_chunk_actions(action)
        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)
        return chunk_actions, {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(action[..., :1]),
            "forward_inputs": forward_inputs,
        }


__all__ = [
    "RLTWarpSACMLPPolicy",
    "WarpSACTwinQCritic",
]
