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

"""LIBERO reset contracts through the real subprocess vector environment."""

import importlib
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="module")
def libero_env(tmp_path_factory):
    # Only replace the external simulator and asset package. Physical modules
    # let spawned workers import the same fake; RLinf's vector path stays real.
    root = tmp_path_factory.mktemp("libero_vendor")
    package = root / "libero" / "libero"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def get_libero_path(name):\n"
        "    return str(Path(__file__).parent)\n"
    )
    (package / "benchmark.py").write_text(
        "from types import SimpleNamespace\n"
        "import numpy as np\n"
        "class Benchmark:\n"
        "    def get_task(self, task_id):\n"
        "        return SimpleNamespace(problem_folder='spatial',\n"
        "            bddl_file='task.bddl', language='move the object')\n"
        "    def get_task_init_states(self, task_id):\n"
        "        return np.arange(6).reshape(3, 2)\n"
    )
    (package / "envs.py").write_text(
        """
import gym
import numpy as np

class OffScreenRenderEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def __init__(self, **kwargs):
        self.fixture_pos = np.zeros(3)
        self.dynamic_state = np.zeros(2)

    def seed(self, seed):
        self.reset_seed = seed
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self.fixture_pos = self.rng.uniform(size=3)
        self.dynamic_state = np.zeros(2)
        return self.dynamic_state.copy()

    def set_init_state(self, state):
        # Like LIBERO, restoring dynamic state leaves fixture poses alone.
        self.dynamic_state = np.array(state, copy=True)
        return self.dynamic_state.copy()
"""
    )
    module_names = (
        "libero",
        "rlinf.envs.sim.libero.libero_env",
        "rlinf.envs.sim.libero.utils",
        "rlinf.envs.sim.libero.venv",
    )
    original_modules = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root))
        patch.setenv("LIBERO_TYPE", "standard")
        cls = importlib.import_module("rlinf.envs.sim.libero.libero_env").LiberoEnv
        benchmark = importlib.import_module("libero.libero.benchmark").Benchmark
        env = cls.__new__(cls)
        # Supply benchmark metadata without real assets or rendering. All reset
        # methods, seed expansion and subprocess IPC use the production classes.
        env.cfg = OmegaConf.create(
            {"task_suite_name": "libero_spatial", "init_params": {}}
        )
        env.num_envs = 4
        env.group_size = 2
        env.seed = 42
        env.task_ids = np.zeros(env.num_envs, dtype=int)
        env.trial_ids = np.zeros(env.num_envs, dtype=int)
        env.cumsum_trial_id_bins = np.array([3])
        env.task_suite = benchmark()
        env._init_env()
        try:
            yield env
        finally:
            env.env.close()
            for name in set(sys.modules) - original_modules:
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in module_names
                ):
                    del sys.modules[name]


@pytest.mark.parametrize("branch_index", [0, 8])
def test_fixed_art_branch_reset_matches_static_fixtures(libero_env, branch_index):
    env = libero_env
    env.art_branch_chunk_index = branch_index
    env.art_state_sampling = False
    env.is_eval = False
    env._reconfigure(np.zeros(4, dtype=int), np.arange(4))

    assert env.env.get_env_attr("reset_seed") == [168, 168, 170, 170]
    fixtures = env.env.get_env_attr("fixture_pos")
    np.testing.assert_array_equal(fixtures[0], fixtures[1])
    np.testing.assert_array_equal(fixtures[2], fixtures[3])
    assert not np.array_equal(fixtures[0], fixtures[2])
    for state in env.env.get_env_attr("dynamic_state"):
        np.testing.assert_array_equal(state, [0, 1])

    # Partial/reordered resets retain absolute environment group membership.
    env._reconfigure(np.array([1, 1]), np.array([3, 2]))
    assert env.env.get_env_attr("reset_seed") == [84, 84, 86, 86]
    fixtures = env.env.get_env_attr("fixture_pos")
    np.testing.assert_array_equal(fixtures[2], fixtures[3])
    for state in env.env.get_env_attr("dynamic_state", id=[2, 3]):
        np.testing.assert_array_equal(state, [2, 3])


@pytest.mark.parametrize(
    "branch_index,state_sampling,is_eval",
    [(None, False, False), (None, True, False), (8, True, False), (8, False, True)],
)
def test_other_libero_modes_keep_independent_reset_seeds(
    libero_env, branch_index, state_sampling, is_eval
):
    env = libero_env
    env.art_branch_chunk_index = branch_index
    env.art_state_sampling = state_sampling
    env.is_eval = is_eval
    env._reconfigure(np.zeros(4, dtype=int), np.arange(4))

    assert env.env.get_env_attr("reset_seed") == [168, 169, 170, 171]
    fixtures = env.env.get_env_attr("fixture_pos")
    assert not np.array_equal(fixtures[0], fixtures[1])

    env._reconfigure(np.array([1, 1]), np.array([3, 2]))
    assert env.env.get_env_attr("reset_seed") == [84, 85, 86, 87]
