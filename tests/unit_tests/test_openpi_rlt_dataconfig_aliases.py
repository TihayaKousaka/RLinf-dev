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
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
torch = pytest.importorskip("torch")
pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.dataconfig.realworld_dataconfig import (
    LeRobotRealworldDataConfig,
)
from rlinf.models.embodiment.openpi.policies.realworld_policy import (
    RealworldInputs,
    RealworldOutputs,
)
from rlinf.models.embodiment.openpi.policies.rlt_joint_policy import (
    RLTJointInputs,
    RLTJointOutputs,
)
from rlinf.models.embodiment.rlt_stage2.proprio import (
    resolve_proprio_dim,
    select_proprio,
)
from toolkits.realworld_rlt.backfill_ee_delta_actions import (
    build_realworld_19d_state,
    build_ee_delta_actions,
)

ROOT = Path(__file__).resolve().parents[2]
MANISKILL_STAGE2_CONFIG = (
    ROOT / "examples/embodiment/config/rlt_stage2_maniskill_joint.yaml"
)
REALWORLD_EE_STAGE2_CONFIG = (
    ROOT / "examples/embodiment/config/rlt_stage2_realworld_ee.yaml"
)
REALWORLD_EE_SFT_EVAL_CONFIG = (
    ROOT / "examples/embodiment/config/rlt_realworld_ee_pi05_sft_eval.yaml"
)
REALWORLD_EE_SFT_CONFIG = ROOT / "examples/sft/config/rlt_realworld_ee_pi05_sft.yaml"
REALWORLD_EE_STAGE1_CONFIG = ROOT / "examples/sft/config/rlt_stage1_realworld_ee.yaml"
REALWORLD_EE_ENV_CONFIG = (
    ROOT / "examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml"
)


def _load_yaml_config(path: Path):
    return OmegaConf.load(path)


def _normalize_config(config):
    cfg_dict = dataclasses.asdict(config)
    cfg_dict.pop("name", None)
    return cfg_dict


def _uint8_image(shape):
    return (np.arange(np.prod(shape)).reshape(shape) % 256).astype(np.uint8)


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _canonicalize_raw_sample(cfg, raw_sample):
    canonical = {
        "observation/image": raw_sample[cfg.data.image_key],
        "observation/wrist_image": raw_sample[cfg.data.wrist_image_key],
        "observation/state": raw_sample[cfg.data.state_key],
        "actions": raw_sample[cfg.data.action_key],
    }
    if "prompt" in raw_sample:
        canonical["prompt"] = raw_sample["prompt"]
    if "task" in raw_sample:
        canonical["task"] = raw_sample["task"]
    return canonical


def _assert_rlt_train_config_contract(
    cfg,
    *,
    repo_id,
    image_key,
    wrist_image_key,
):
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is True
    assert cfg.data.repo_id == repo_id
    assert cfg.data.image_key == image_key
    assert cfg.data.wrist_image_key == wrist_image_key
    assert cfg.data.state_key == "state"
    assert cfg.data.action_key == "actions"
    assert cfg.data.extra_delta_transform is False


def _assert_rlt_policy_transform_contract(
    cfg,
    raw_sample,
    *,
    expected_state_dim,
    expected_base_image_shape,
    expected_wrist_image_shape,
    expected_action_dim=8,
):
    inputs_transform = RLTJointInputs(
        model_type=cfg.model.model_type,
        use_wrist_image=True,
    )
    outputs_transform = RLTJointOutputs(output_action_dim=expected_action_dim)

    transformed = inputs_transform(_canonicalize_raw_sample(cfg, raw_sample))
    actions = outputs_transform({"actions": raw_sample[cfg.data.action_key]})

    assert transformed["state"].shape == (expected_state_dim,)
    assert transformed["actions"].shape == raw_sample[cfg.data.action_key].shape
    assert actions["actions"].shape == (cfg.model.action_horizon, expected_action_dim)
    np.testing.assert_array_equal(
        actions["actions"],
        raw_sample[cfg.data.action_key][:, :expected_action_dim],
    )

    assert transformed["image"]["base_0_rgb"].shape == expected_base_image_shape
    assert transformed["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(
        transformed["image"]["base_0_rgb"],
        raw_sample[cfg.data.image_key],
    )

    assert transformed["image"]["left_wrist_0_rgb"].shape == expected_wrist_image_shape
    assert transformed["image"]["left_wrist_0_rgb"].dtype == np.uint8
    expected_wrist = raw_sample[cfg.data.wrist_image_key]
    if np.issubdtype(expected_wrist.dtype, np.floating):
        expected_wrist = (255 * expected_wrist).astype(np.uint8)
    if expected_wrist.ndim == 3 and expected_wrist.shape[0] == 3:
        expected_wrist = np.transpose(expected_wrist, (1, 2, 0))
    np.testing.assert_array_equal(
        transformed["image"]["left_wrist_0_rgb"],
        expected_wrist,
    )

    assert transformed["image"]["right_wrist_0_rgb"].shape == expected_base_image_shape
    np.testing.assert_array_equal(
        transformed["image"]["right_wrist_0_rgb"],
        np.zeros(expected_base_image_shape, dtype=np.uint8),
    )
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert transformed["prompt"] == raw_sample["prompt"]


def _assert_realworld_policy_transform_contract(
    cfg,
    raw_sample,
    *,
    expected_base_image_shape,
    expected_extra_view_image_shape,
):
    inputs_transform = RealworldInputs(
        action_dim=cfg.model.action_dim,
        model_type=cfg.model.model_type,
    )
    outputs_transform = RealworldOutputs()

    canonical = {
        "observation/image": raw_sample["image"],
        "observation/extra_view_image": raw_sample["extra_view_image"],
        "observation/state": raw_sample["state"],
        "actions": raw_sample["actions"],
        "prompt": raw_sample["prompt"],
    }
    transformed = inputs_transform(canonical)
    actions = outputs_transform({"actions": transformed["actions"]})

    assert transformed["state"].shape == (cfg.model.action_dim,)
    state = _as_numpy(transformed["state"])
    np.testing.assert_array_equal(state[:19], raw_sample["state"])
    np.testing.assert_array_equal(state[19:], np.zeros(cfg.model.action_dim - 19))
    assert transformed["actions"].shape == (
        cfg.model.action_horizon,
        cfg.model.action_dim,
    )
    padded_actions = _as_numpy(transformed["actions"])
    np.testing.assert_array_equal(
        padded_actions[:, :7],
        raw_sample["actions"],
    )
    np.testing.assert_array_equal(
        padded_actions[:, 7:],
        np.zeros((cfg.model.action_horizon, cfg.model.action_dim - 7)),
    )
    assert actions["actions"].shape == (cfg.model.action_horizon, 7)
    np.testing.assert_array_equal(actions["actions"], raw_sample["actions"])

    assert transformed["image"]["base_0_rgb"].shape == expected_base_image_shape
    assert transformed["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(
        transformed["image"]["base_0_rgb"],
        raw_sample["image"],
    )

    assert (
        transformed["image"]["left_wrist_0_rgb"].shape
        == expected_extra_view_image_shape
    )
    assert transformed["image"]["left_wrist_0_rgb"].dtype == np.uint8
    expected_extra_view = raw_sample["extra_view_image"]
    if np.issubdtype(expected_extra_view.dtype, np.floating):
        expected_extra_view = (255 * expected_extra_view).astype(np.uint8)
    if expected_extra_view.ndim == 3 and expected_extra_view.shape[0] == 3:
        expected_extra_view = np.transpose(expected_extra_view, (1, 2, 0))
    np.testing.assert_array_equal(
        transformed["image"]["left_wrist_0_rgb"],
        expected_extra_view,
    )

    assert transformed["image"]["right_wrist_0_rgb"].shape == expected_base_image_shape
    np.testing.assert_array_equal(
        transformed["image"]["right_wrist_0_rgb"],
        np.zeros(expected_base_image_shape, dtype=np.uint8),
    )
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert transformed["prompt"] == raw_sample["prompt"]


def _assert_stage2_dimension_contract(
    cfg,
    *,
    config_name,
    action_dim,
    action_horizon,
    num_images,
    proprio_dim,
    proprio_mode=None,
):
    if isinstance(config_name, tuple):
        assert cfg.actor.model.rlt_stage2.config_name in config_name
    else:
        assert cfg.actor.model.rlt_stage2.config_name == config_name
    assert cfg.actor.model.num_action_chunks == action_horizon
    assert cfg.actor.model.action_dim == action_dim
    assert cfg.actor.model.rlt_stage2.num_images_in_input == num_images
    assert cfg.actor.model.rlt_stage2.proprio_dim == proprio_dim
    assert cfg.env.eval.action_exec_chunks == action_horizon

    if proprio_mode is not None:
        assert cfg.actor.model.rlt_stage2.proprio_mode == proprio_mode
    assert (
        resolve_proprio_dim(
            cfg.actor.model.rlt_stage2,
            default_dim=cfg.actor.model.rlt_stage2.proprio_dim,
        )
        == proprio_dim
    )


def test_rlt_maniskill_joint_dataconfig_contract():
    canonical = get_openpi_config("pi05_rlt_joint")
    legacy = get_openpi_config("pi05_rlt_maniskill_joint")

    assert _normalize_config(canonical) == _normalize_config(legacy)
    _assert_rlt_train_config_contract(
        canonical,
        repo_id="rlt_maniskill_joint",
        image_key="image",
        wrist_image_key="wrist_image",
    )

    raw_sample = {
        "image": _uint8_image((384, 384, 3)),
        "wrist_image": _uint8_image((3, 128, 128)),
        "state": np.linspace(-1.0, 1.0, 9, dtype=np.float32),
        "actions": np.arange(10 * 10, dtype=np.float32).reshape(10, 10),
        "prompt": "insert the peg in the hole",
    }
    _assert_rlt_policy_transform_contract(
        canonical,
        raw_sample,
        expected_state_dim=9,
        expected_base_image_shape=(384, 384, 3),
        expected_wrist_image_shape=(128, 128, 3),
    )


def test_rlt_maniskill_stage2_yaml_dimension_contract():
    cfg = _load_yaml_config(MANISKILL_STAGE2_CONFIG)

    assert cfg.env.train.wrap_obs_mode == "rlt_openpi_joint"
    assert cfg.env.train.init_params.sensor_configs.width == 384
    assert cfg.env.train.init_params.sensor_configs.height == 384
    _assert_stage2_dimension_contract(
        cfg,
        config_name=("pi05_rlt_joint", "pi05_rlt_maniskill_joint"),
        action_dim=8,
        action_horizon=10,
        num_images=2,
        proprio_dim=9,
        proprio_mode="maniskill_joint",
    )


def test_rlt_realworld_ee_dataconfig_contract():
    cfg = get_openpi_config("pi05_rlt_realworld_ee")

    assert isinstance(cfg.data, LeRobotRealworldDataConfig)
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is True
    assert cfg.data.repo_id == "rlt_realworld_ee"
    assert cfg.data.extra_delta_transform is False

    raw_sample = {
        "image": _uint8_image((128, 128, 3)),
        "extra_view_image": np.linspace(0.0, 1.0, 3 * 96 * 96, dtype=np.float32).reshape(
            3, 96, 96
        ),
        "state": np.linspace(-2.0, 2.0, 19, dtype=np.float32),
        "actions": np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
        "prompt": "insert the peg in the hole",
    }
    _assert_realworld_policy_transform_contract(
        cfg,
        raw_sample,
        expected_base_image_shape=(128, 128, 3),
        expected_extra_view_image_shape=(96, 96, 3),
    )


def test_rlt_realworld_ee_stage2_yaml_dimension_contract():
    cfg = _load_yaml_config(REALWORLD_EE_STAGE2_CONFIG)

    _assert_stage2_dimension_contract(
        cfg,
        config_name="pi05_rlt_realworld_ee",
        action_dim=7,
        action_horizon=10,
        num_images=2,
        proprio_dim=19,
        proprio_mode="realworld_ee",
    )
    assert cfg.env.train.gello_action_mode == "ee_delta"
    assert cfg.env.eval.gello_action_mode == "ee_delta"
    assert cfg.env.train.keyboard_reward_wrapper is None
    assert cfg.env.eval.keyboard_reward_wrapper is None


def test_rlt_stage2_explicit_proprio_mode_rejects_legacy_raw_state():
    state = torch.zeros((2, 34), dtype=torch.float32)

    with pytest.raises(ValueError, match="legacy/raw state layouts"):
        select_proprio(state, proprio_mode="realworld_ee")


def test_rlt_realworld_ee_sft_yaml_dimension_contract():
    cfg = _load_yaml_config(REALWORLD_EE_SFT_CONFIG)

    assert cfg.actor.model.action_dim == 7
    assert cfg.actor.model.openpi.config_name == "pi05_rlt_realworld_ee"
    assert cfg.actor.model.openpi.num_images_in_input == 2
    assert cfg.actor.model.openpi.action_env_dim == cfg.actor.model.action_dim
    assert cfg.actor.openpi_data.repo_id.endswith("/id_4_ee_action_state19")
    assert cfg.actor.openpi_data.norm_stats_path.endswith(
        "/id_4_ee_action_state19/norm_stats.json"
    )


def test_rlt_realworld_ee_sft_eval_yaml_dimension_contract():
    cfg = _load_yaml_config(REALWORLD_EE_SFT_EVAL_CONFIG)

    assert cfg.actor.model.action_dim == 7
    assert cfg.actor.model.num_action_chunks == 10
    assert cfg.actor.model.openpi.config_name == "pi05_rlt_realworld_ee"
    assert cfg.actor.model.openpi.num_images_in_input == 2
    assert cfg.actor.model.openpi.action_env_dim == cfg.actor.model.action_dim
    assert cfg.env.eval.action_exec_chunks == cfg.actor.model.num_action_chunks
    assert cfg.env.train.gello_action_mode == "ee_delta"
    assert cfg.env.eval.gello_action_mode == "ee_delta"


def test_rlt_realworld_ee_stage1_yaml_dimension_contract():
    cfg = _load_yaml_config(REALWORLD_EE_STAGE1_CONFIG)

    assert cfg.actor.model.action_dim == 7
    assert cfg.actor.model.rlt_stage1.config_name == "pi05_rlt_realworld_ee"
    assert cfg.actor.model.rlt_stage1.num_images_in_input == 2
    assert cfg.actor.openpi_data.repo_id.endswith("/id_4_ee_action_state19")
    assert cfg.actor.openpi_data.norm_stats_path.endswith(
        "/id_4_ee_action_state19/norm_stats.json"
    )


def test_rlt_realworld_ee_env_yaml_observation_contract():
    cfg = _load_yaml_config(REALWORLD_EE_ENV_CONFIG)
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert cfg.use_rlt_joint_obs is False
    assert cfg.use_quat_tcp_pose is False
    assert raw_cfg["main_image_key"] == "main_camera"
    assert raw_cfg["wrist_image_key"] == "wrist_camera"
    assert cfg.gello_action_mode == "ee_delta"
    assert raw_cfg["init_params"]["id"] == "PegInsertionEnv-v1"
    assert "target_ee_pose" in raw_cfg["override_cfg"]
    assert "critical_phase_reset_joint_qpos" not in raw_cfg["override_cfg"]
    assert "full_task_reset_joint_qpos" not in raw_cfg["override_cfg"]
    assert list(cfg.state_key_order) == [
        "gripper_position",
        "tcp_force",
        "tcp_pose",
        "tcp_torque",
        "tcp_vel",
    ]


def test_realworld_joint_to_ee_action_backfill_contract():
    state = np.zeros((3, 34), dtype=np.float32)
    state[:, 21:25] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state[:, 18:21] = np.array(
        [
            [0.50, 0.00, 0.10],
            [0.51, 0.02, 0.10],
            [0.51, 0.02, 0.08],
        ],
        dtype=np.float32,
    )
    actions = np.zeros((3, 8), dtype=np.float32)
    actions[:, 7] = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    ee_actions, metrics = build_ee_delta_actions(
        state,
        actions,
        pos_scale=0.02,
        rot_scale=0.1,
        clip=False,
    )

    expected = np.array(
        [
            [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(ee_actions, expected, atol=1e-6)
    assert metrics["max_abs_arm_action"] == pytest.approx(1.0)
    assert metrics["arm_clip_fraction"] == pytest.approx(0.0)


def test_realworld_34d_to_19d_state_backfill_contract():
    state = np.zeros((2, 34), dtype=np.float32)
    state[:, 0] = [0.25, 0.75]
    state[:, 1:8] = 100.0
    state[:, 8:15] = 200.0
    state[:, 15:18] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    state[:, 18:21] = [[0.5, -0.1, 0.2], [0.6, -0.2, 0.3]]
    state[:, 21:25] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state[:, 25:28] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    state[:, 28:34] = [
        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    ]

    state_19 = build_realworld_19d_state(state)

    expected = np.array(
        [
            [
                0.25,
                1.0,
                2.0,
                3.0,
                0.5,
                -0.1,
                0.2,
                0.0,
                0.0,
                0.0,
                7.0,
                8.0,
                9.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
            ],
            [
                0.75,
                4.0,
                5.0,
                6.0,
                0.6,
                -0.2,
                0.3,
                0.0,
                0.0,
                0.0,
                10.0,
                11.0,
                12.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(state_19, expected, atol=1e-6)
