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
        self.alpha = float(stage1_cfg.get("alpha", 0.0))
        self.rl_token_micro_batch_size = int(
            stage1_cfg.get("rl_token_micro_batch_size", 0) or 0
        )
        joint_finetune = self.alpha > 0.0

        vla = build_openpi_rlt_backbone(
            model_path=cfg.model_path,
            config_name=stage1_cfg.config_name,
            num_images_in_input=int(stage1_cfg.get("num_images_in_input", 2)),
            num_action_chunks=int(cfg.num_action_chunks),
            action_dim=int(cfg.action_dim),
            num_steps=int(stage1_cfg.get("num_steps", 5)),
            device=self.device,
            freeze=not joint_finetune,
        )
        if joint_finetune:
            self.vla = vla
        else:
            # Keep the frozen VLA out of PyTorch's module tree so FSDP only
            # flattens the trainable RL-token module.
            object.__setattr__(self, "vla", vla)

        self.rl_token_model = RLTokenModel(
            embedding_dim=int(stage1_cfg.get("embedding_dim", 2048)),
            encoder_layers=int(stage1_cfg.get("encoder_layers", 2)),
            encoder_heads=int(stage1_cfg.get("encoder_heads", 8)),
            decoder_layers=int(stage1_cfg.get("decoder_layers", 2)),
            decoder_heads=int(stage1_cfg.get("decoder_heads", 8)),
        ).to(self.device)
        # Keep the RL-token block as its own FSDP wrap unit so joint Stage 1
        # finetuning does not flatten its float32 parameters together with the
        # OpenPI backbone's bfloat16 parameters.
        self.rl_token_model._fsdp_wrap_name = "rl_token_model"

    @property
    def _no_split_modules(self) -> list[str]:
        if self.alpha <= 0.0:
            return []
        return list(getattr(self.vla, "_no_split_modules", []))

    @property
    def _no_split_names(self) -> list[str]:
        names = ["rl_token_model"]
        if self.alpha > 0.0:
            names.extend(getattr(self.vla, "_no_split_names", []))
        return list(dict.fromkeys(names))

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"Unsupported forward_type for RLT Stage 1: {forward_type}"
        )

    def trainable_parameters(self):
        if self.alpha > 0.0:
            return self.parameters()
        return self.rl_token_model.parameters()

    def sft_forward(self, data: dict[str, Any], **kwargs) -> dict[str, torch.Tensor]:
        observation = data["observation"]

        if self.alpha > 0.0:
            with torch.no_grad():
                z, pad_mask = self.vla.extract_rlt_prefix_embeddings(
                    observation,
                    dtype=torch.float32,
                )
            vla_loss = self.vla(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": data["actions"]},
            )
        else:
            with torch.no_grad():
                z, pad_mask = self.vla.extract_rlt_prefix_embeddings(
                    observation,
                    dtype=torch.float32,
                )
            vla_loss = None

        l_ro, z_rl, z_hat = self.compute_rl_token_loss(z, pad_mask)
        loss = l_ro if vla_loss is None else l_ro + self.alpha * vla_loss
        metrics = {
            "loss": loss,
            "l_ro": l_ro.detach(),
            "z_rl": z_rl.detach(),
            "z_hat": z_hat.detach(),
        }
        if vla_loss is not None:
            metrics["vla_loss"] = vla_loss.detach()
            metrics["alpha"] = torch.as_tensor(self.alpha, device=l_ro.device)
        self._latest_stage1_metrics = {
            key: value for key, value in metrics.items() if key != "loss"
        }
        return metrics

    def compute_rl_token_loss(
        self,
        z: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        micro_batch_size = self.rl_token_micro_batch_size
        if micro_batch_size <= 0 or micro_batch_size >= z.shape[0]:
            return self.rl_token_model(z, pad_mask)

        loss_sum = z.new_zeros(())
        num_valid = z.new_zeros(())
        z_rl_chunks = []
        z_hat_chunks = []
        for start in range(0, z.shape[0], micro_batch_size):
            end = min(start + micro_batch_size, z.shape[0])
            chunk_loss_sum, chunk_num_valid, chunk_z_rl, chunk_z_hat = (
                self.rl_token_model.loss_sum(z[start:end], pad_mask[start:end])
            )
            loss_sum = loss_sum + chunk_loss_sum
            num_valid = num_valid + chunk_num_valid.to(loss_sum.device)
            z_rl_chunks.append(chunk_z_rl.detach())
            z_hat_chunks.append(chunk_z_hat.detach())

        loss = loss_sum / num_valid.clamp(min=1.0)
        return loss, torch.cat(z_rl_chunks, dim=0), torch.cat(z_hat_chunks, dim=0)

    def default_forward(self, **kwargs):
        raise NotImplementedError("RLT Stage 1 does not use default_forward.")

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError("RLT Stage 1 does not expose rollout actions.")
