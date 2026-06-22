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

"""RLT Stage 1 policy for RL-token training."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.openpi import build_openpi_rlt_backbone

from .rl_token import RLTokenModel


class RLTStage1Policy(torch.nn.Module, BasePolicy):
    def __init__(
        self,
        cfg: DictConfig,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self._latest_stage1_metrics: dict[str, torch.Tensor] = {}

        stage1_cfg = cfg.rlt_stage1

        vla = build_openpi_rlt_backbone(
            model_path=cfg.model_path,
            config_name=stage1_cfg.config_name,
            norm_stats_path=stage1_cfg.get("norm_stats_path", None),
            num_images_in_input=int(stage1_cfg.get("num_images_in_input", 2)),
            num_action_chunks=int(cfg.num_action_chunks),
            action_dim=int(cfg.action_dim),
            num_steps=int(stage1_cfg.get("num_steps", 5)),
            device=self.device,
            freeze=True,
        )
        # Keep the frozen VLA out of PyTorch's module tree so FSDP only flattens
        # the trainable RL-token module. The backbone is already on-device and
        # remains accessible through ``self.vla`` for feature extraction.
        object.__setattr__(self, "vla", vla)

        self.rl_token_model = RLTokenModel(
            embedding_dim=int(stage1_cfg.get("embedding_dim", 2048)),
            encoder_layers=int(stage1_cfg.get("encoder_layers", 2)),
            encoder_heads=int(stage1_cfg.get("encoder_heads", 8)),
            decoder_layers=int(stage1_cfg.get("decoder_layers", 2)),
            decoder_heads=int(stage1_cfg.get("decoder_heads", 8)),
        ).to(self.device)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"Unsupported forward_type for RLT Stage 1: {forward_type}"
        )

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        """Expose only RL-token parameters to FSDP/optimizer.

        The VLA backbone is a frozen feature extractor in Stage 1. Keeping it out
        of FSDP flattening avoids mixing its bf16 parameters with the fp32
        RL-token module while preserving the same forward computation.
        """

        rl_token_prefix = f"{prefix}.rl_token_model" if prefix else "rl_token_model"
        yield from self.rl_token_model.named_parameters(
            prefix=rl_token_prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def parameters(self, recurse: bool = True):
        yield from self.rl_token_model.parameters(recurse=recurse)

    def trainable_parameters(self):
        return self.rl_token_model.parameters()

    def sft_forward(self, data: dict[str, Any], **kwargs) -> dict[str, torch.Tensor]:
        observation = data["observation"]

        with torch.no_grad():
            z, pad_mask = self.vla.extract_rlt_prefix_embeddings(
                observation,
                dtype=torch.float32,
            )
        l_ro, z_rl, z_hat = self.rl_token_model(
            z.to(self.device), pad_mask.to(self.device)
        )
        metrics = {
            "loss": l_ro,
            "l_ro": l_ro.detach(),
            "z_rl": z_rl.detach(),
            "z_hat": z_hat.detach(),
        }
        self._latest_stage1_metrics = {
            key: value for key, value in metrics.items() if key != "loss"
        }
        return metrics

    def default_forward(self, **kwargs):
        raise NotImplementedError("RLT Stage 1 does not use default_forward.")

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError("RLT Stage 1 does not expose rollout actions.")
