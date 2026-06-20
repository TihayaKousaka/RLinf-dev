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

"""Thin OpenPI loader used by the OpenPI-RLT policies."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf, open_dict

from rlinf.models.embodiment.openpi import get_model as get_openpi_model


def build_openpi_rlt_backbone(
    *,
    model_path: str,
    config_name: str,
    norm_stats_path: str | None = None,
    num_images_in_input: int = 2,
    num_action_chunks: int = 10,
    action_dim: int = 8,
    num_steps: int = 5,
    device: torch.device | str = "cuda",
    freeze: bool = True,
) -> torch.nn.Module:
    """Build an OpenPI backbone with the existing upstream loader path."""
    cfg_dict = {
        "model_type": "openpi",
        "model_path": model_path,
        "precision": None,
        "is_lora": False,
        "load_to_device": True,
        "openpi": {
            "config_name": config_name,
            "num_images_in_input": int(num_images_in_input),
            "action_chunk": int(num_action_chunks),
            "num_steps": int(num_steps),
            "action_env_dim": int(action_dim),
            "train_expert_only": False,
            "add_value_head": False,
        },
    }
    if norm_stats_path is not None:
        cfg_dict["openpi_data"] = {"norm_stats_path": norm_stats_path}

    cfg = OmegaConf.create(cfg_dict)
    with open_dict(cfg):
        cfg.num_action_chunks = int(num_action_chunks)
        cfg.action_dim = int(action_dim)

    model = get_openpi_model(cfg, torch_dtype=None)
    model.to(torch.device(device))
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model
