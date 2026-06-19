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
import dataclasses
import json
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import realworld_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRealworldDataConfig(DataConfigFactory):
    """Data configuration for RLinf-collected realworld LeRobot datasets."""

    default_prompt: str | None = None
    extra_delta_transform: bool = False
    norm_stats_path: str | None = None

    def _load_explicit_norm_stats(self):
        if not self.norm_stats_path:
            return None

        norm_stats_path = pathlib.Path(self.norm_stats_path).expanduser()
        if not norm_stats_path.exists():
            raise FileNotFoundError(
                f"Explicit norm stats file not found: {norm_stats_path}"
            )

        if norm_stats_path.is_dir():
            return _normalize.load(norm_stats_path)

        raw_text = norm_stats_path.read_text()
        if hasattr(_normalize, "deserialize_json"):
            return _normalize.deserialize_json(raw_text)

        raw_data = json.loads(raw_text)
        return raw_data.get("norm_stats", raw_data)

    def generate_observations(
        image: np.ndarray, state: np.ndarray, prompt: str
    ) -> dict:
        """Creates an input example for the realworld policy."""
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image": "extra_view_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                realworld_policy.RealworldInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[realworld_policy.RealworldOutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        base_config = self.create_base_config(assets_dirs, model_config)
        explicit_norm_stats = self._load_explicit_norm_stats()
        if explicit_norm_stats is not None:
            base_config = dataclasses.replace(
                base_config, norm_stats=explicit_norm_stats
            )

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
