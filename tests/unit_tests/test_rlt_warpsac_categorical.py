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

from rlinf.algorithms.rlt.categorical import (
    categorical_cross_entropy,
    categorical_q_values,
    project_categorical_distribution,
    select_min_categorical_logits,
)
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import TwinQCritic


def test_select_min_categorical_critic_by_expected_value() -> None:
    support = torch.tensor([-1.0, 0.0, 1.0])
    logits = torch.tensor(
        [
            [[-8.0, -8.0, 8.0], [8.0, -8.0, -8.0]],
            [[-8.0, 8.0, -8.0], [-8.0, -8.0, 8.0]],
        ]
    )

    selected = select_min_categorical_logits(logits, support)
    selected_values = categorical_q_values(selected, support)

    assert selected.shape == (2, 3)
    assert torch.allclose(selected, torch.stack([logits[0, 1], logits[1, 0]]))
    assert torch.allclose(selected_values, torch.tensor([-1.0, 0.0]), atol=1.0e-5)


def test_categorical_projection_preserves_probability_and_interpolates() -> None:
    support = torch.tensor([0.0, 0.5, 1.0])
    source_logits = torch.zeros(1, 3)
    target_atoms = torch.full((1, 3), 0.25)

    projected = project_categorical_distribution(source_logits, target_atoms, support)

    assert torch.allclose(projected, torch.tensor([[0.5, 0.5, 0.0]]))
    assert torch.allclose(projected.sum(dim=-1), torch.ones(1))


def test_categorical_projection_clamps_outside_support() -> None:
    support = torch.tensor([-1.0, 0.0, 1.0])
    source_logits = torch.zeros(2, 3)
    target_atoms = torch.tensor([[-3.0, -2.0, -1.5], [1.5, 2.0, 3.0]])

    projected = project_categorical_distribution(source_logits, target_atoms, support)

    assert torch.allclose(projected[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(projected[1], torch.tensor([0.0, 0.0, 1.0]))


def test_categorical_cross_entropy_accepts_twin_logits() -> None:
    logits = torch.zeros(4, 2, 3, requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, 0.0]]).expand(4, -1)

    loss = categorical_cross_entropy(logits, targets)
    loss.backward()

    assert torch.allclose(loss.detach(), torch.log(torch.tensor(3.0)))
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_twin_q_critic_preserves_scalar_and_categorical_contracts() -> None:
    state = torch.randn(3, 4)
    action = torch.randn(3, 2)
    scalar_critic = TwinQCritic(4, 2, hidden_dim=8, num_hidden_layers=1)
    categorical_critic = TwinQCritic(
        4,
        2,
        hidden_dim=8,
        num_hidden_layers=1,
        distribution_type="categorical",
        num_bins=5,
        v_min=-2.0,
        v_max=2.0,
    )

    scalar_values = scalar_critic(state, action)
    categorical_logits = categorical_critic(state, action, return_logits=True)
    categorical_values = categorical_critic(state, action)

    assert scalar_values.shape == (3, 2)
    assert categorical_logits.shape == (3, 2, 5)
    assert categorical_values.shape == (3, 2)
    assert torch.allclose(
        categorical_values,
        categorical_q_values(categorical_logits, categorical_critic.support),
    )
