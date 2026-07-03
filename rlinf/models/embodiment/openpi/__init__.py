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
# openpi model configs

import os

import torch
from omegaconf import DictConfig


def _load_torch_state_dict(path: str) -> dict:
    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = state_dict.get(key)
            if isinstance(nested, dict):
                return nested
    return state_dict


def _load_safetensors_state_dict(path: str) -> dict:
    import safetensors

    if os.path.isdir(path):
        weight_paths = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".safetensors")
        )
        if not weight_paths:
            weight_paths = [os.path.join(path, "model.safetensors")]
    else:
        weight_paths = [path]

    all_state_dict = {}
    for weight_path in weight_paths:
        all_state_dict.update(safetensors.torch.load_file(weight_path, device="cpu"))
    return all_state_dict


def _checkpoint_weight_path(path: str) -> str | None:
    if not os.path.isdir(path):
        return None
    candidates = (
        os.path.join(path, "model_state_dict", "full_weights.pt"),
        os.path.join(path, "actor", "model_state_dict", "full_weights.pt"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_openpi_state_dict(path: str) -> dict:
    weight_path = _checkpoint_weight_path(path)
    if weight_path is not None:
        return _load_torch_state_dict(weight_path)

    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            return _load_safetensors_state_dict(path)
        return _load_torch_state_dict(path)

    return _load_safetensors_state_dict(path)


def _configured_rlt_module_path(cfg: DictConfig) -> str | None:
    for container in (cfg, getattr(cfg, "openpi", None)):
        if container is None:
            continue
        path = container.get("rlt_module_path", None)
        if path:
            return str(path)
        path = container.get("rlt_token_path", None)
        if path:
            return str(path)
    return None


def _extract_rlt_module_state_dict(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected dict checkpoint for rlt_module, got {type(state_dict)}")

    for nested_key in ("rlt_module", "rl_token", "rl_token_state_dict"):
        nested = state_dict.get(nested_key)
        if isinstance(nested, dict):
            state_dict = nested
            break

    prefixes = (
        "rlt_module.",
        "module.rlt_module.",
        "_orig_mod.rlt_module.",
        "model.rlt_module.",
    )
    extracted = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                break

    if extracted:
        return extracted
    if any(key.startswith(("encoder.", "decoder.")) for key in state_dict):
        return state_dict

    raise ValueError(
        "Could not find rlt_module weights. Expected keys like "
        "'rlt_module.encoder.*' or direct 'encoder.*'/'decoder.*' keys."
    )


def _is_legacy_rlt_module_state_dict(state_dict: dict) -> bool:
    return any(
        key.startswith(("encoder.e_rl", "encoder.transformer.", "decoder.h_phi"))
        for key in state_dict
    )


def _replace_with_legacy_rlt_module(model, rlt_state_dict: dict) -> None:
    from rlinf.models.embodiment.modules.rlt_token_transformer import (
        LegacyRLTTokenTransformer,
    )

    e_rl = rlt_state_dict.get("encoder.e_rl", None)
    e_rl_shape = tuple(e_rl.shape) if torch.is_tensor(e_rl) else None
    model.rlt_module = LegacyRLTTokenTransformer(
        input_dim=int(model.config.rlt_input_dim),
        embed_dim=int(model.config.rlt_embed_dim),
        num_rl_tokens=int(model.config.rlt_num_rl_tokens),
        prefix_seq_len=int(model.config.rlt_prefix_seq_len),
        num_layers=int(model.config.rlt_num_layers),
        num_heads=int(model.config.rlt_num_heads),
        mlp_ratio=float(model.config.rlt_mlp_ratio),
        e_rl_shape=e_rl_shape,
    )


def _load_rlt_module_override(model, cfg: DictConfig) -> None:
    rlt_module_path = _configured_rlt_module_path(cfg)
    if not rlt_module_path:
        return
    if not hasattr(model, "rlt_module"):
        raise ValueError(
            f"rlt_module_path={rlt_module_path!r} was set, but model has no rlt_module."
        )

    checkpoint_dir = rlt_module_path
    if not os.path.exists(checkpoint_dir):
        import openpi.shared.download as download

        checkpoint_dir = download.maybe_download(rlt_module_path)

    state_dict = _load_openpi_state_dict(checkpoint_dir)
    rlt_state_dict = _extract_rlt_module_state_dict(state_dict)
    if _is_legacy_rlt_module_state_dict(rlt_state_dict):
        _replace_with_legacy_rlt_module(model, rlt_state_dict)
    strict = bool(getattr(cfg, "strict_rlt_module_load", True))
    incompatible = model.rlt_module.load_state_dict(rlt_state_dict, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "Failed to load rlt_module override strictly: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def get_model(cfg: DictConfig, torch_dtype=None):
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    model.load_state_dict(_load_openpi_state_dict(checkpoint_dir), strict=False)
    _load_rlt_module_override(model, cfg)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
