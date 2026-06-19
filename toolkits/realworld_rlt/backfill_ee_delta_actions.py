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
"""Backfill realworld RLT joint-action demos into EE-delta-action demos.

The source dataset is the current single-arm realworld RLT LeRobot layout:

    state: 34D [gripper, joint_pos(7), joint_vel(7), tcp_force(3),
                tcp_pose(7 xyz+xyzw), tcp_torque(3), tcp_vel(6)]
    actions: 8D [joint_target(7), gripper]

The output keeps ``state`` unchanged at 34D, but rewrites ``actions`` to the
7D normalized action consumed by ``PegInsertionEnv-v1``:

    [dx / pos_scale, dy / pos_scale, dz / pos_scale,
     droll / rot_scale, dpitch / rot_scale, dyaw / rot_scale, gripper]

The EE target proxy is the next frame's recorded TCP pose. The last frame holds
the current TCP pose. The source is left untouched.

Run from the repo root::

    export PYTHONPATH=$(pwd)
    python toolkits/realworld_rlt/backfill_ee_delta_actions.py \\
        --src /path/to/rlt_realworld_joint \\
        --dst /path/to/rlt_realworld_ee
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

STATE_DIM = 34
ACTION_DIM_IN = 8
ACTION_DIM_OUT = 7

TCP_XYZ_SLICE = slice(18, 21)
TCP_QUAT_SLICE = slice(21, 25)
GRIPPER_ACTION_IDX = 7

DEFAULT_POS_SCALE = 0.02
DEFAULT_ROT_SCALE = 0.1
DEFAULT_GRIPPER_SCALE = 1.0


# ---------------------------------------------------------------------------
# Pure-numpy transforms
# ---------------------------------------------------------------------------


def _assert_unit_quats(state_34: np.ndarray) -> None:
    norms = np.linalg.norm(state_34[:, TCP_QUAT_SLICE], axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = int(np.argmax(np.abs(norms - 1.0)))
        raise ValueError(
            f"state[:, {TCP_QUAT_SLICE}] does not look like xyzw quaternions: "
            f"norm range [{norms.min():.4f}, {norms.max():.4f}], "
            f"worst row {bad} norm={norms[bad]:.4f}. Check the source state layout."
        )


def _assert_source_layout(state_34: np.ndarray, actions_8: np.ndarray) -> None:
    if state_34.ndim != 2 or state_34.shape[1] != STATE_DIM:
        raise ValueError(f"expected state shape (T, {STATE_DIM}), got {state_34.shape}")
    if actions_8.shape != (state_34.shape[0], ACTION_DIM_IN):
        raise ValueError(
            f"expected actions shape (T, {ACTION_DIM_IN}), got {actions_8.shape}"
        )
    _assert_unit_quats(state_34)


def _quat_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
    inv = quat.copy()
    inv[:, :3] *= -1.0
    return inv


def _quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Vectorized xyzw quaternion multiplication matching scipy Rotation order."""
    x1, y1, z1, w1 = np.moveaxis(lhs, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(rhs, -1, 0)
    out = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def _quat_xyzw_to_xyz_euler(quat: np.ndarray) -> np.ndarray:
    """Return xyz Euler angles for small EE deltas from xyzw quaternions."""
    try:
        from scipy.spatial.transform import Rotation as R
    except ModuleNotFoundError:
        pass
    else:
        return R.from_quat(quat).as_euler("xyz")

    matrix = _quat_xyzw_to_matrix(quat)
    roll = np.arctan2(matrix[:, 2, 1], matrix[:, 2, 2])
    pitch = np.arcsin(np.clip(-matrix[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def build_ee_delta_actions(
    state_34: np.ndarray,
    actions_8: np.ndarray,
    *,
    pos_scale: float = DEFAULT_POS_SCALE,
    rot_scale: float = DEFAULT_ROT_SCALE,
    gripper_scale: float = DEFAULT_GRIPPER_SCALE,
    clip: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build normalized 7D EE-delta actions from a 34D realworld RLT episode."""

    _assert_source_layout(state_34, actions_8)
    if pos_scale <= 0 or rot_scale <= 0 or gripper_scale <= 0:
        raise ValueError(
            "pos_scale, rot_scale, and gripper_scale must be positive, got "
            f"{pos_scale=}, {rot_scale=}, {gripper_scale=}."
        )

    nxt = np.empty_like(state_34)
    nxt[:-1] = state_34[1:]
    nxt[-1] = state_34[-1]

    cur_xyz = state_34[:, TCP_XYZ_SLICE]
    next_xyz = nxt[:, TCP_XYZ_SLICE]
    delta_xyz = (next_xyz - cur_xyz) / float(pos_scale)

    cur_quat = state_34[:, TCP_QUAT_SLICE]
    next_quat = nxt[:, TCP_QUAT_SLICE]
    delta_quat = _quat_multiply_xyzw(next_quat, _quat_inverse_xyzw(cur_quat))
    delta_rpy = _quat_xyzw_to_xyz_euler(delta_quat) / float(rot_scale)

    gripper = actions_8[:, GRIPPER_ACTION_IDX : GRIPPER_ACTION_IDX + 1] / float(
        gripper_scale
    )
    raw_out = np.concatenate([delta_xyz, delta_rpy, gripper], axis=-1).astype(np.float32)
    if raw_out.shape != (state_34.shape[0], ACTION_DIM_OUT):
        raise AssertionError(f"unexpected action shape {raw_out.shape}")

    clip_abs = np.abs(raw_out[:, :6])
    metrics = {
        "max_abs_arm_action": float(clip_abs.max(initial=0.0)),
        "arm_clip_fraction": float((clip_abs > 1.0 + 1e-6).mean())
        if clip_abs.size
        else 0.0,
    }
    if clip:
        raw_out[:, :6] = np.clip(raw_out[:, :6], -1.0, 1.0)
        raw_out[:, 6:] = np.clip(raw_out[:, 6:], -1.0, 1.0)
    return raw_out, metrics


# ---------------------------------------------------------------------------
# Parquet IO
# ---------------------------------------------------------------------------


def _import_pyarrow():
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def _fsl_float32(arr_2d: np.ndarray):
    """Wrap a ``(T, D)`` float array as ``FixedSizeList<float>[D]``."""
    pa, _ = _import_pyarrow()
    if arr_2d.dtype != np.float32:
        arr_2d = arr_2d.astype(np.float32)
    _, dim = arr_2d.shape
    flat = pa.array(arr_2d.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _fsl_to_numpy(col, dim: int) -> np.ndarray:
    """Pull a ``FixedSizeList<float>[D]`` column into ``(T, D)`` float32."""
    flat = np.asarray(col.combine_chunks().values.to_numpy(zero_copy_only=False))
    return flat.reshape(-1, dim).astype(np.float32, copy=False)


def _patch_hf_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    """Update Hugging Face schema metadata's actions length to 7 if present."""
    if metadata is None:
        return None
    out = dict(metadata)
    raw = out.get(b"huggingface")
    if raw is None:
        return out
    info = json.loads(raw)
    feats = info.get("info", {}).get("features", {})
    if "actions" in feats:
        feats["actions"]["length"] = ACTION_DIM_OUT
    if "state" in feats:
        feats["state"]["length"] = STATE_DIM
    out[b"huggingface"] = json.dumps(info, separators=(",", ":")).encode()
    return out


def rewrite_parquet(
    src_path: Path,
    dst_path: Path,
    *,
    pos_scale: float,
    rot_scale: float,
    gripper_scale: float,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rewrite one episode parquet. Returns unchanged state, new actions, metrics."""

    pa, pq = _import_pyarrow()
    src_table = pq.read_table(src_path)
    if "state" not in src_table.column_names or "actions" not in src_table.column_names:
        raise ValueError(f"{src_path}: missing state/actions column")

    state_type = src_table.schema.field("state").type
    actions_type = src_table.schema.field("actions").type
    state_size = getattr(state_type, "list_size", None)
    actions_size = getattr(actions_type, "list_size", None)
    if state_size == STATE_DIM and actions_size == ACTION_DIM_OUT:
        raise ValueError(
            f"{src_path}: actions are already {ACTION_DIM_OUT}D; dataset looks "
            "already backfilled. Point --src at the original 8D joint dataset, "
            "or pick a fresh --dst."
        )
    if state_size != STATE_DIM or actions_size != ACTION_DIM_IN:
        raise ValueError(
            f"{src_path}: expected source state/actions dimensions "
            f"{STATE_DIM}/{ACTION_DIM_IN}, got {state_size}/{actions_size}."
        )

    state_np = _fsl_to_numpy(src_table.column("state"), STATE_DIM)
    actions_np = _fsl_to_numpy(src_table.column("actions"), ACTION_DIM_IN)
    new_actions, metrics = build_ee_delta_actions(
        state_np,
        actions_np,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        gripper_scale=gripper_scale,
        clip=clip,
    )

    new_fields = []
    for field in src_table.schema:
        if field.name == "actions":
            new_fields.append(
                pa.field(
                    "actions",
                    pa.list_(pa.float32(), ACTION_DIM_OUT),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            new_fields.append(field)
    new_schema = pa.schema(
        new_fields, metadata=_patch_hf_metadata(src_table.schema.metadata)
    )

    new_columns = []
    for name in src_table.column_names:
        if name == "actions":
            new_columns.append(_fsl_float32(new_actions))
        else:
            new_columns.append(src_table.column(name).combine_chunks())

    new_table = pa.Table.from_arrays(new_columns, schema=new_schema)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return state_np, new_actions, metrics


# ---------------------------------------------------------------------------
# Meta / stats rewrite
# ---------------------------------------------------------------------------


def _stats_for(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def _aggregate_stats(arrays: list[np.ndarray]) -> dict:
    if not arrays:
        raise ValueError("cannot aggregate stats from an empty array list")
    return _stats_for(np.concatenate(arrays, axis=0))


def _patch_info_json(src_meta: Path, dst_meta: Path) -> None:
    with (src_meta / "info.json").open() as f:
        info = json.load(f)
    info["features"]["state"]["shape"] = [STATE_DIM]
    info["features"]["actions"]["shape"] = [ACTION_DIM_OUT]
    with (dst_meta / "info.json").open("w") as f:
        json.dump(info, f, indent=4)


def _patch_episodes_stats(
    src_meta: Path,
    dst_meta: Path,
    new_actions_per_episode: dict[int, np.ndarray],
) -> None:
    src_stats = src_meta / "episodes_stats.jsonl"
    if not src_stats.exists():
        return
    dst_stats = dst_meta / "episodes_stats.jsonl"
    with src_stats.open() as f_in, dst_stats.open("w") as f_out:
        for line in f_in:
            entry = json.loads(line)
            ep = entry["episode_index"]
            if ep in new_actions_per_episode:
                entry["stats"]["actions"] = _stats_for(new_actions_per_episode[ep])
            f_out.write(json.dumps(entry) + "\n")


def _copy_other_meta(src_meta: Path, dst_meta: Path) -> None:
    for src_file in src_meta.iterdir():
        if not src_file.is_file():
            continue
        if src_file.name in {"info.json", "episodes_stats.jsonl"}:
            continue
        shutil.copy2(src_file, dst_meta / src_file.name)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _dump_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def _patch_norm_stats(
    src: Path,
    dst: Path,
    all_state: list[np.ndarray],
    all_actions: list[np.ndarray],
) -> None:
    """Copy top-level norm_stats.json when present, patching actions stats."""

    src_stats_path = src / "norm_stats.json"
    if src_stats_path.exists():
        stats = _load_json(src_stats_path)
        container = stats.get("norm_stats", stats)
    else:
        stats = {}
        container = stats

    if all_state:
        container["state"] = _aggregate_stats(all_state)
    container["actions"] = _aggregate_stats(all_actions)
    _dump_json(dst / "norm_stats.json", stats)


def rewrite_meta(
    src: Path,
    dst: Path,
    per_episode_state: dict[int, np.ndarray],
    per_episode_actions: dict[int, np.ndarray],
) -> None:
    src_meta = src / "meta"
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    _patch_info_json(src_meta, dst_meta)
    _patch_episodes_stats(src_meta, dst_meta, per_episode_actions)
    _copy_other_meta(src_meta, dst_meta)
    _patch_norm_stats(
        src,
        dst,
        list(per_episode_state.values()),
        list(per_episode_actions.values()),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _find_parquets(src_data: Path) -> list[Path]:
    return sorted(src_data.rglob("episode_*.parquet"))


def _iter_progress(items: list[Path], *, desc: str):
    try:
        import tqdm
    except ModuleNotFoundError:
        return items
    return tqdm.tqdm(items, desc=desc)


def backfill(
    src: Path,
    dst: Path,
    *,
    pos_scale: float = DEFAULT_POS_SCALE,
    rot_scale: float = DEFAULT_ROT_SCALE,
    gripper_scale: float = DEFAULT_GRIPPER_SCALE,
    clip: bool = True,
    overwrite: bool = False,
) -> None:
    src_meta = src / "meta"
    src_data = src / "data"
    if not src_meta.is_dir() or not src_data.is_dir():
        raise FileNotFoundError(
            f"{src} does not look like a LeRobot root (expected meta/ and data/)"
        )

    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst} already exists. Use --overwrite to rewrite it."
            )
        shutil.rmtree(dst)
    dst_data = dst / "data"

    parquets = _find_parquets(src_data)
    if not parquets:
        raise FileNotFoundError(f"No episode_*.parquet under {src_data}")

    per_episode_state: dict[int, np.ndarray] = {}
    per_episode_actions: dict[int, np.ndarray] = {}
    max_abs_arm_action = 0.0
    clipped_values = 0.0
    total_arm_values = 0.0
    for src_pq in _iter_progress(parquets, desc="backfill ee delta actions"):
        rel = src_pq.relative_to(src_data)
        dst_pq = dst_data / rel
        state_np, new_actions, metrics = rewrite_parquet(
            src_pq,
            dst_pq,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            gripper_scale=gripper_scale,
            clip=clip,
        )
        ep_idx = int(src_pq.stem.split("_")[-1])
        per_episode_state[ep_idx] = state_np
        per_episode_actions[ep_idx] = new_actions
        max_abs_arm_action = max(max_abs_arm_action, metrics["max_abs_arm_action"])
        clipped_values += metrics["arm_clip_fraction"] * state_np.shape[0] * 6
        total_arm_values += state_np.shape[0] * 6

    rewrite_meta(src, dst, per_episode_state, per_episode_actions)

    total_frames = sum(arr.shape[0] for arr in per_episode_actions.values())
    clip_fraction = clipped_values / total_arm_values if total_arm_values else 0.0
    print(
        f"OK  episodes={len(per_episode_actions)}  frames={total_frames}  "
        f"state dim={STATE_DIM}  actions dim={ACTION_DIM_OUT}"
    )
    print(
        "    action scale: "
        f"pos={pos_scale} rot={rot_scale} gripper={gripper_scale}; "
        f"clip={clip} max_abs_arm_action_preclip={max_abs_arm_action:.4f} "
        f"arm_clip_fraction={clip_fraction:.6f}"
    )
    print(f"    src -> dst: {src}  ->  {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill a single-arm realworld RLT joint-action LeRobot dataset "
            "into a 34D-state/7D-EE-action dataset for PegInsertionEnv-v1."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="LeRobot root of the 34D-state/8D-joint-action dataset.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output LeRobot root for the 34D-state/7D-EE-action dataset.",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=DEFAULT_POS_SCALE,
        help="Position scale used by PegInsertionEnv action_scale[0].",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=DEFAULT_ROT_SCALE,
        help="Rotation scale used by PegInsertionEnv action_scale[1].",
    )
    parser.add_argument(
        "--gripper-scale",
        type=float,
        default=DEFAULT_GRIPPER_SCALE,
        help="Gripper scale used by PegInsertionEnv action_scale[2].",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip output actions to [-1, 1].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --dst before writing. Does NOT touch --src.",
    )
    args = parser.parse_args()

    try:
        backfill(
            args.src,
            args.dst,
            pos_scale=args.pos_scale,
            rot_scale=args.rot_scale,
            gripper_scale=args.gripper_scale,
            clip=not args.no_clip,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
