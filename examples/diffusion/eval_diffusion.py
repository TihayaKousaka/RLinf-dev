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

"""Evaluate a diffusion training recipe using its complete policy configuration."""

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict


@hydra.main(
    version_base="1.1", config_path="config", config_name="cosmos3_nft_video_ocr"
)
def main(cfg: DictConfig) -> None:
    from evaluations.eval_embodied_agent import main as evaluate

    # Training recipes keep architecture knobs under actor.model.
    with open_dict(cfg):
        cfg.rollout.model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    evaluate(cfg)


if __name__ == "__main__":
    main()
