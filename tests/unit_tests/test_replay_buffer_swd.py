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

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer


def _make_transition(index: int, expert: bool) -> Trajectory:
    value = float(index)
    obs = {
        "z_rl": torch.full((1, 1, 2), value),
        "proprio": torch.full((1, 1, 1), value),
        "ref_chunk": torch.full((1, 1, 2, 1), value),
    }
    return Trajectory(
        max_episode_length=1,
        model_weights_id=str(index),
        actions=torch.full((1, 1, 2, 1), value),
        intervene_flags=torch.full((1, 1, 2), expert, dtype=torch.bool),
        rewards=torch.full((1, 1, 1), value),
        terminations=torch.zeros((1, 1, 1), dtype=torch.bool),
        truncations=torch.zeros((1, 1, 1), dtype=torch.bool),
        dones=torch.zeros((1, 1, 1), dtype=torch.bool),
        curr_obs=obs,
        next_obs=obs,
    )


def test_swd_weights_and_expert_ratio_are_preserved() -> None:
    buffer = TrajectoryReplayBuffer(
        seed=7,
        enable_cache=True,
        cache_size=8,
        sample_window_size=0,
        swd_enable=True,
        swd_decay_step=2,
        swd_min_weight=0.1,
        swd_preserve_expert_ratio=True,
    )
    try:
        for index in range(5):
            buffer.add_trajectories([_make_transition(index, expert=index % 2 == 0)])

        weights = buffer._swd_weight(torch.arange(5, dtype=torch.long))
        assert torch.allclose(weights, torch.tensor([0.1, 0.1, 0.1, 0.5, 1.0]))

        batch = buffer.sample_chunks(1000)
        sampled_expert_ratio = (
            batch["intervene_flags"].reshape(1000, -1).any(dim=-1).float().mean()
        )
        assert abs(sampled_expert_ratio.item() - 0.6) < 1e-6

        stats = buffer.get_stats()
        assert stats["swd_enabled"] == 1.0
        assert abs(stats["swd_candidate_expert_ratio"] - 0.6) < 1e-6
        assert abs(stats["swd_sample_expert_ratio"] - 0.6) < 1e-6
        assert stats["swd_sample_age_mean"] < 2.0
    finally:
        buffer.close()


def test_uniform_sampling_remains_compatible() -> None:
    buffer = TrajectoryReplayBuffer(
        seed=11,
        enable_cache=True,
        cache_size=4,
        sample_window_size=0,
        swd_enable=False,
    )
    try:
        for index in range(3):
            buffer.add_trajectories([_make_transition(index, expert=index == 0)])
        batch = buffer.sample_chunks(32)
        assert batch["actions"].shape[0] == 32
        assert buffer.get_stats()["swd_enabled"] == 0.0
    finally:
        buffer.close()
