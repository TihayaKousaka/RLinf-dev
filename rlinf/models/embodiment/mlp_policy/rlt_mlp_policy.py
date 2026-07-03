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
from torch.distributions.normal import Normal

from rlinf.models.embodiment.modules.utils import make_mlp
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


def _make_relu_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> nn.Sequential:
    return nn.Sequential(
        *make_mlp(
            in_channels=input_dim,
            mlp_channels=[
                *[hidden_dim for _ in range(num_hidden_layers)],
                output_dim,
            ],
            act_builder=nn.ReLU,
            last_act=False,
        )
    )


class DirectRLTQHead(nn.Module):
    """Ablation-compatible twin-Q MLP head for RLT Stage2."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_q_heads: int,
    ):
        super().__init__()
        if int(num_q_heads) <= 0:
            raise ValueError(f"num_q_heads must be positive, got {num_q_heads}.")
        self.qs = nn.ModuleList(
            [
                _make_relu_mlp(
                    input_dim=state_dim + action_dim,
                    output_dim=1,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                for _ in range(int(num_q_heads))
            ]
        )

    def forward(self, state_features, action_features, **kwargs):
        del kwargs
        q_input = torch.cat([state_features, action_features], dim=-1)
        return torch.cat([q(q_input) for q in self.qs], dim=-1)


class RLTMLPPolicy(MLPPolicy):
    """MLP policy for RLT actor/critic heads.

    Actor input follows RLT: reference action chunk, RL token feature, and
    proprioceptive state. Critic input follows RLT: action chunk, RL token
    feature, and proprioceptive state. The actor is a Gaussian policy over
    action chunks, matching the RLT Stage2 objective.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        num_q_heads: int = 2,
        rlt_head_type: str = "sac",
        actor_noise_sigma: float = 0.003,
    ):
        if not add_q_head:
            raise ValueError("RLTMLPPolicy requires add_q_head=True for RL training.")
        rlt_head_type = str(rlt_head_type)
        if rlt_head_type not in {"sac", "direct"}:
            raise ValueError(
                f"Unsupported RLT MLP head type: {rlt_head_type!r}. "
                "Expected 'sac' or 'direct'."
            )
        z_dim = int(z_dim)
        proprio_dim = int(proprio_dim)
        step_action_dim = int(action_dim)
        chunk_len = int(num_action_chunks)
        flat_action_dim = chunk_len * step_action_dim

        actor_obs_dim = z_dim + proprio_dim + flat_action_dim
        critic_obs_dim = z_dim + proprio_dim

        if rlt_head_type == "direct":
            if q_head_type != "default":
                raise ValueError(
                    "RLT direct MLP head only supports q_head_type='default', "
                    f"got {q_head_type!r}."
                )
            nn.Module.__init__(self)
            self.obs_dim = actor_obs_dim
            self.critic_obs_dim = critic_obs_dim
            self.action_dim = flat_action_dim
            self.num_action_chunks = 1
            self.independent_std = False
            self.final_tanh = False
            self.action_scale = None
            self.torch_compile_enabled = False
            self.cuda_graph_manager = None
            self.direct_actor = _make_relu_mlp(
                input_dim=actor_obs_dim,
                output_dim=flat_action_dim,
                hidden_dim=int(hidden_dim),
                num_hidden_layers=int(num_hidden_layers),
            )
            self.q_head = DirectRLTQHead(
                state_dim=critic_obs_dim,
                action_dim=flat_action_dim,
                hidden_dim=int(hidden_dim),
                num_hidden_layers=int(num_hidden_layers),
                num_q_heads=int(num_q_heads),
            )
        else:
            super().__init__(
                obs_dim=actor_obs_dim,
                action_dim=flat_action_dim,
                num_action_chunks=1,
                add_value_head=False,
                add_q_head=add_q_head,
                q_head_type=q_head_type,
                hidden_dim=hidden_dim,
                num_q_heads=num_q_heads,
                critic_obs_dim=critic_obs_dim,
            )
        self.rlt_head_type = rlt_head_type
        self.actor_noise_sigma = float(actor_noise_sigma)
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.step_action_dim = step_action_dim
        self.chunk_len = chunk_len
        self.flat_action_dim = flat_action_dim

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed = {}
        for key, value in env_obs.items():
            processed[key] = value.to(device) if torch.is_tensor(value) else value
        return processed

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["z_rl"])

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["proprio"])

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = obs["ref_chunk"]
        if ref_chunk.dim() == 3:
            ref_chunk = ref_chunk[:, : self.chunk_len]
        else:
            ref_chunk = self._flatten_batch(ref_chunk).reshape(
                ref_chunk.shape[0], -1, self.step_action_dim
            )
            ref_chunk = ref_chunk[:, : self.chunk_len]
        return ref_chunk.reshape(ref_chunk.shape[0], -1)

    def _maybe_drop_reference(
        self,
        ref_chunk: torch.Tensor,
        reference_dropout_prob: float,
    ) -> torch.Tensor:
        if reference_dropout_prob <= 0:
            return ref_chunk
        keep_prob = 1.0 - float(reference_dropout_prob)
        keep_mask = (
            torch.rand((ref_chunk.shape[0], 1), device=ref_chunk.device) < keep_prob
        )
        return ref_chunk * keep_mask.to(dtype=ref_chunk.dtype)

    def _actor_state(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        ref_chunk = self._get_ref_chunk(obs)
        if apply_reference_dropout:
            ref_chunk = self._maybe_drop_reference(ref_chunk, reference_dropout_prob)
        # Match the original RLT Stage2 actor input order: x=[z_rl, proprio],
        # then the VLA reference action chunk.
        return torch.cat([self._get_z(obs), self._get_proprio(obs), ref_chunk], dim=-1)

    def _critic_state(self, obs: dict) -> torch.Tensor:
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def _direct_action_noise_sigma(
        self,
        deterministic: bool,
        action_noise_sigma: float,
    ) -> float:
        if float(action_noise_sigma) > 0.0:
            return float(action_noise_sigma)
        if deterministic:
            return 0.0
        return float(self.actor_noise_sigma)

    def sac_forward(
        self,
        obs,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
        action_noise_sigma: float = 0.0,
        action_noise_clip: float | None = None,
        **kwargs,
    ):
        del kwargs
        actor_state = self._actor_state(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        if self.rlt_head_type == "direct":
            action = self.direct_actor(actor_state)
            noise_sigma = self._direct_action_noise_sigma(
                deterministic,
                action_noise_sigma,
            )
            if noise_sigma > 0.0:
                noise = torch.randn_like(action) * noise_sigma
                if action_noise_clip is not None and float(action_noise_clip) > 0.0:
                    noise = noise.clamp(
                        min=-float(action_noise_clip),
                        max=float(action_noise_clip),
                    )
                action = action + noise
            action = action.clamp(-1.0, 1.0)
            return action, torch.zeros_like(action), None

        feat = self.backbone(actor_state)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd(feat)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = self.logstd_range[0] + 0.5 * (
            self.logstd_range[1] - self.logstd_range[0]
        ) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        raw_action = action_mean if deterministic else probs.rsample()

        action_normalized = torch.tanh(raw_action)
        action = action_normalized * self.action_scale + self.action_bias
        if action_noise_sigma > 0.0:
            noise = torch.randn_like(action) * float(action_noise_sigma)
            if action_noise_clip is not None and float(action_noise_clip) > 0.0:
                noise = noise.clamp(
                    min=-float(action_noise_clip),
                    max=float(action_noise_clip),
                )
            action = (action + noise).clamp(-1.0, 1.0)

        chunk_logprobs = probs.log_prob(raw_action)
        chunk_logprobs = chunk_logprobs - torch.log(
            self.action_scale * (1 - action_normalized.pow(2)) + 1e-6
        )

        return action, chunk_logprobs, None

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        del shared_feature
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        return self.q_head(critic_state, self._flatten_batch(actions))

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        del shared_feature
        critic_state = self._critic_state(obs)
        next_critic_state = (
            self._critic_state(next_obs) if next_obs is not None else None
        )
        if detach_encoder:
            critic_state = critic_state.detach()
            if next_critic_state is not None:
                next_critic_state = next_critic_state.detach()
        return self.q_head(
            critic_state,
            self._flatten_batch(actions),
            next_state_features=next_critic_state,
            next_action_features=(
                self._flatten_batch(next_actions) if next_actions is not None else None
            ),
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def set_q_head_requires_grad(self, requires_grad: bool) -> None:
        for param in self.q_head.parameters():
            param.requires_grad_(requires_grad)

    def sft_forward(self, data, **kwargs):
        obs = data["obs"] if "obs" in data else data
        target_actions = self._flatten_batch(
            data["action"] if "action" in data else data["actions"]
        )
        actor_state = self._actor_state(obs)
        if self.rlt_head_type == "direct":
            pred_actions = self.direct_actor(actor_state).clamp(-1.0, 1.0)
        else:
            pred_actions = self.actor_mean(self.backbone(actor_state))
        return F.mse_loss(pred_actions, target_actions, reduction="none")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        sampling_mode: str | None = None,
        action_noise_sigma: float = 0.0,
        action_noise_clip: float | None = None,
        **kwargs,
    ):
        del calculate_logprobs, calculate_values
        obs = self.preprocess_env_obs(env_obs=env_obs)
        if sampling_mode is None:
            sampling_mode = "deterministic" if mode == "eval" else "sac_sample"
        if sampling_mode not in {"sac_sample", "deterministic", "td3_action_noise"}:
            raise ValueError(f"Unsupported RLT sampling_mode: {sampling_mode!r}")
        deterministic = mode == "eval" or sampling_mode != "sac_sample"
        noise_sigma = (
            float(action_noise_sigma)
            if sampling_mode == "td3_action_noise"
            else 0.0
        )
        action, chunk_logprobs, _ = self.sac_forward(
            obs,
            deterministic=deterministic,
            action_noise_sigma=noise_sigma,
            action_noise_clip=action_noise_clip,
        )
        chunk_actions = self._format_chunk_actions(action)

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(chunk_logprobs[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
