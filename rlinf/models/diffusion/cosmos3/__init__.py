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

import typing
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from rlinf.models.diffusion.cosmos3.policy import (
    Cosmos3DiffusionConfig,
    Cosmos3DiffusionPolicy,
)


def get_model(cfg: DictConfig, torch_dtype=None) -> Cosmos3DiffusionPolicy:
    """Load a native Cosmos3 DCP checkpoint with RLinf owning FSDP."""
    # Cosmos currently imports the Python 3.12 spelling on Python 3.11.
    if not hasattr(typing, "override"):
        from typing_extensions import override

        typing.override = override

    import cosmos_framework.model.generator.omni_mot_model as omni_module
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
    from cosmos_framework.utils import checkpoint_db
    from cosmos_framework.utils.flags import Device
    from huggingface_hub import hf_hub_download, snapshot_download

    config = Cosmos3DiffusionConfig(
        model_path=str(cfg.model_path),
        **OmegaConf.to_container(cfg.cosmos3_diffusion, resolve=True),
    )
    path = Path(config.model_path).expanduser().resolve()
    if (path / "model" / ".metadata").is_file():
        path = path / "model"
    if not (path / ".metadata").is_file() or not (path / "config.json").is_file():
        raise ValueError(
            f"Expected a Cosmos3 DCP directory with .metadata and config.json: {path}"
        )
    auxiliary = (
        Path(config.auxiliary_model_path).expanduser().resolve()
        if config.auxiliary_model_path
        else None
    )

    def local_file(checkpoint):
        if auxiliary is not None and checkpoint.repository == "nvidia/Cosmos3-Nano":
            file = auxiliary / checkpoint.filename
            if not file.is_file():
                raise FileNotFoundError(file)
            return str(file)
        return hf_hub_download(
            checkpoint.repository,
            checkpoint.filename,
            revision=checkpoint.revision,
            repo_type=checkpoint.repository_type.value,
            local_files_only=config.local_files_only,
        )

    def local_dir(checkpoint):
        if auxiliary is not None and checkpoint.repository == "nvidia/Cosmos3-Nano":
            directory = auxiliary / checkpoint.subdirectory
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            return str(directory)
        directory = snapshot_download(
            checkpoint.repository,
            revision=checkpoint.revision,
            repo_type=checkpoint.repository_type.value,
            local_files_only=config.local_files_only,
        )
        return str(Path(directory) / checkpoint.subdirectory)

    # Tokenizer assets use the same Hub cache as the checkpoint conversion.
    original_file = checkpoint_db.CheckpointFileHf._download
    original_dir = checkpoint_db.CheckpointDirHf._download
    original_device = omni_module.DEVICE
    checkpoint_db.CheckpointFileHf._download = local_file
    checkpoint_db.CheckpointDirHf._download = local_dir
    omni_module.DEVICE = Device.CPU
    try:
        hf_config = Cosmos3OmniConfig.from_pretrained(path)
        wrapper = Cosmos3OmniModel.from_pretrained_dcp(path, config=hf_config)
    finally:
        checkpoint_db.CheckpointFileHf._download = original_file
        checkpoint_db.CheckpointDirHf._download = original_dir
        omni_module.DEVICE = original_device
    omni = wrapper.model
    # CPU materialization uses to_empty and skips native CUDA initialization;
    # these nonpersistent buffers are absent from DCP and must be rebuilt.
    for module in omni.net.modules():
        if "inv_freq" in module._non_persistent_buffers_set:
            module.init_weights(buffer_device="cpu")
        if "_timestep_frequencies" in module._non_persistent_buffers_set:
            module.register_buffer(
                "_timestep_frequencies",
                module._build_timestep_frequencies(
                    module.frequency_embedding_size, 10000, device="cpu"
                ),
                persistent=False,
            )
    omni.requires_grad_(False)
    if config.use_lora:
        from peft import LoraConfig, inject_adapter_in_model

        omni.net = inject_adapter_in_model(
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.target_modules,
                init_lora_weights="gaussian",
            ),
            omni.net,
        )
    else:
        from rlinf.models.embodiment.cosmos3 import _apply_freeze

        omni.net.requires_grad_(True)
        _apply_freeze(omni, config.target_modules)
    # FSDP2 requires parameters to have at least one dimension.
    from rlinf.models.embodiment.cosmos3 import _promote_scalar_params_to_1d

    _promote_scalar_params_to_1d(omni)
    return Cosmos3DiffusionPolicy(config, omni)


__all__ = ["Cosmos3DiffusionConfig", "Cosmos3DiffusionPolicy", "get_model"]
