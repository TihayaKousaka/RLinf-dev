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

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


@dataclass
class Cosmos3DiffusionConfig:
    model_path: str = ""
    auxiliary_model_path: str | None = None
    local_files_only: bool = True
    resolution: list[int] = field(default_factory=lambda: [256, 256])
    num_frames: int = 5
    fps: float = 24.0
    num_steps: int = 10
    eval_num_steps: int = 20
    max_sequence_length: int = 512
    guidance_scale: float = 1.0
    eval_guidance_scale: float = 1.0
    timestep_fraction: float = 1.0
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
            "o_proj_moe_gen",
        ]
    )

    def __post_init__(self) -> None:
        if len(self.resolution) != 2 or any(v <= 0 or v % 32 for v in self.resolution):
            raise ValueError(
                "Cosmos3 resolution must contain two positive multiples of 32"
            )
        if self.num_frames <= 0 or (self.num_frames - 1) % 4:
            raise ValueError("Cosmos3 num_frames must be positive and equal to 4k+1")
        if self.num_steps <= 0 or self.eval_num_steps <= 0 or self.fps <= 0:
            raise ValueError("Cosmos3 sampling steps and fps must be positive")
        if self.max_sequence_length <= 0 or self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError(
                "Cosmos3 sequence length, LoRA rank and alpha must be positive"
            )
        if not 0 < self.timestep_fraction <= 1:
            raise ValueError("Cosmos3 timestep_fraction must be in (0, 1]")


class Cosmos3DiffusionPolicy(torch.nn.Module, BasePolicy):
    """Text-to-video Cosmos3 policy using normalized flow time and video rewards."""

    _no_split_modules = ["MoTDecoderLayer"]
    _fsdp_wrap_embeddings = False

    def __init__(self, config: Cosmos3DiffusionConfig, omni: torch.nn.Module):
        super().__init__()
        self.config = config
        self.model_path = config.model_path
        self.omni = omni
        self._gradient_checkpointing_enabled = False

    @property
    def device(self) -> torch.device:
        return next(self.omni.net.parameters()).device

    def _update_device(self) -> None:
        self.omni.tensor_kwargs["device"] = self.device
        self.omni.tensor_kwargs_fp32["device"] = self.device

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs) -> dict[str, Any]:
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        raise NotImplementedError("Cosmos3 diffusion training supports ForwardType.NFT")

    def default_forward(self, **kwargs):
        raise NotImplementedError("Cosmos3 diffusion requires NFT training")

    def _velocity(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        from cosmos_framework.data.generator.sequence_packing.sequence import (
            SequencePlan,
        )
        from cosmos_framework.model.generator.utils.data_and_condition import (
            GenerationDataClean,
        )

        self._update_device()
        if latents.ndim != 5:
            raise ValueError(
                f"Expected Cosmos3 latents [B,C,T,H,W], got {latents.shape}"
            )
        tokens = [
            row[mask.bool()].tolist()
            for row, mask in zip(token_ids, token_mask, strict=True)
        ]
        plans = [SequencePlan(has_text=True, has_vision=True) for _ in tokens]
        clean = GenerationDataClean(
            batch_size=latents.shape[0],
            is_image_batch=self.config.num_frames == 1,
            x0_tokens_vision=[x.unsqueeze(0) for x in latents],
            fps_vision=torch.full(
                (latents.shape[0],), self.config.fps, device=self.device
            ),
        )
        velocity = self.omni._get_velocity(
            noise_x=[item.flatten() for item in latents],
            timestep=timesteps.reshape(-1).float() * 1000.0,
            text_tokens=tokens,
            sequence_plans=plans,
            gen_data_clean=clean,
            has_noisy_actions=False,
        )
        return torch.stack(
            [v.reshape_as(x) for v, x in zip(velocity, latents, strict=True)]
        )

    def nft_forward(self, forward_inputs: dict, nft_inputs: dict, **kwargs) -> dict:
        x_t = nft_inputs["x_t"].to(self.device)
        # RLinf stores the latent temporal axis before the channel axis.
        latents = x_t.movedim(1, 2)
        t = nft_inputs["timesteps"].to(self.device).reshape(-1)
        v = self._guided_velocity(
            latents, t, forward_inputs, self.config.guidance_scale
        )
        return {"v_theta": v.movedim(2, 1)}

    def _guided_velocity(self, latents, timesteps, inputs, guidance):
        v = self._velocity(
            latents, timesteps, inputs["token_ids"], inputs["token_mask"]
        )
        if guidance != 1.0:
            unconditional = self._velocity(
                latents,
                timesteps,
                inputs["negative_token_ids"],
                inputs["negative_token_mask"],
            )
            v = unconditional + guidance * (v - unconditional)
        return v

    @torch.no_grad()
    def _encode_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        cond, uncond = self.omni._get_inference_text_tokens(
            {self.omni.input_caption_key: prompts}, False, None
        )
        result = {}
        for prefix, rows in (("", cond), ("negative_", uncond)):
            if any(len(row) > self.config.max_sequence_length for row in rows):
                raise ValueError(
                    "Cosmos3 prompt exceeds max_sequence_length; increase the configured limit"
                )
            ids = torch.zeros(
                len(rows),
                self.config.max_sequence_length,
                dtype=torch.long,
                device=self.device,
            )
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for index, row in enumerate(rows):
                ids[index, : len(row)] = torch.tensor(row, device=self.device)
                mask[index, : len(row)] = True
            result[prefix + "token_ids"] = ids
            result[prefix + "token_mask"] = mask
        return result

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs, mode="train", compute_values=False, **kwargs
    ):
        self._update_device()
        prompts = env_obs["task_descriptions"]
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        inputs = self._encode_prompts(prompts)
        height, width = self.config.resolution
        latent_frames = (self.config.num_frames - 1) // 4 + 1
        shape = (
            len(prompts),
            self.omni.config.state_ch,
            latent_frames,
            height // 16,
            width // 16,
        )
        latents = torch.randn(
            shape,
            device=self.device,
            dtype=torch.float32,
            generator=kwargs.get("generator"),
        )
        is_eval = mode == "eval"
        steps = self.config.eval_num_steps if is_eval else self.config.num_steps
        guidance = (
            self.config.eval_guidance_scale if is_eval else self.config.guidance_scale
        )
        schedule = torch.linspace(1, 0, steps + 1, device=self.device)
        for i in range(steps):
            v = self._guided_velocity(
                latents, schedule[i].expand(len(prompts)), inputs, guidance
            )
            latents = latents + (schedule[i + 1] - schedule[i]) * v.float()
        # Native Wan tokenizer decoding returns RGB pixels in [-1, 1].
        videos = (
            self.omni.decode(latents)
            .float()
            .add(1)
            .div(2)
            .clamp(0, 1)
            .permute(0, 2, 1, 3, 4)
        )
        if is_eval:
            return videos, {"prev_values": None}
        inputs["nft_x0"] = latents.movedim(2, 1).detach()
        inputs["nft_noise_level"] = torch.zeros(len(prompts), device=self.device)
        return videos, {
            "prev_logprobs": torch.zeros(
                len(prompts),
                max(1, int(steps * self.config.timestep_fraction)),
                device=self.device,
            ),
            "prev_values": None,
            "forward_inputs": inputs,
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        if self._gradient_checkpointing_enabled:
            return
        from cosmos_framework.configs.base.defaults.activation_checkpointing import (
            ActivationCheckpointingConfig,
        )
        from cosmos_framework.model.generator.mot.parallelize_unified_mot import (
            apply_ac,
        )

        apply_ac(
            self.omni.net.language_model, ActivationCheckpointingConfig(mode="full")
        )
        self._gradient_checkpointing_enabled = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def export_config(self) -> dict[str, Any]:
        options = asdict(self.config)
        options.pop("model_path")
        return {"model_path": self.model_path, "cosmos3_diffusion": options}


def validate_cosmos3_diffusion_cfg(cfg) -> None:
    """Validate the video-level NFT contract before workers are launched."""
    from omegaconf import OmegaConf

    only_eval = (
        cfg.runner.get("only_eval", False)
        or cfg.runner.get("task_type") == "embodied_eval"
    )
    model = cfg.rollout.model if only_eval else cfg.actor.model
    options = Cosmos3DiffusionConfig(
        model_path=str(model.model_path),
        **OmegaConf.to_container(model.cosmos3_diffusion, resolve=True),
    )
    expected_chunks = (options.num_frames - 1) // 4 + 1
    if model.num_action_chunks != expected_chunks:
        raise ValueError(
            f"Cosmos3 num_action_chunks must be {expected_chunks} for {options.num_frames} frames"
        )
    if model.precision != "bf16":
        raise ValueError("Cosmos3 diffusion currently requires precision=bf16")
    if not only_eval:
        algorithm = cfg.algorithm
        expected = {
            "loss_type": "embodied_nft",
            "nft_gather_type": "chunk_level",
            "nft_target_space": "x0",
            "nft_xcur_source": "resample",
            "adv_type": "grpo_video",
            "advantage_mode": "video",
        }
        for key, value in expected.items():
            if algorithm.get(key) != value:
                raise ValueError(f"Cosmos3 video OCR requires algorithm.{key}={value}")
        if algorithm.group_size < 2:
            raise ValueError("Cosmos3 OCR RL requires algorithm.group_size >= 2")
    for split in ("eval",) if only_eval else ("train", "eval"):
        env = cfg.env.get(split)
        if env is not None:
            if (
                env.env_type != "diffusion"
                or env.reward.get("reward_type") != "video_level"
            ):
                raise ValueError(
                    f"Cosmos3 requires env.{split}.env_type=diffusion and video_level rewards"
                )
            if env.reward.get("frame_interval") != 4:
                raise ValueError(
                    f"Cosmos3 requires env.{split}.reward.frame_interval=4"
                )
