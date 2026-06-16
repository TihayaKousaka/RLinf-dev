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

"""Inspect RLinf TrajectoryReplayBuffer directories produced by RLT Stage2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_NAMES = {
    0: "BASE",
    1: "RL",
    2: "HUMAN",
    3: "MIXED",
}
PHASE_NAMES = {
    0: "UNKNOWN",
    1: "WARMUP",
    2: "ONLINE",
}
REQUIRED_FORWARD_INPUTS = (
    "rewards",
    "dones",
    "intervention",
    "source",
    "source_chunk",
    "collection_phase_id",
    "success",
    "intervention_flag",
    "episode_id",
    "step_id",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an RLT Stage2 TrajectoryReplayBuffer directory."
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a replay rank directory containing metadata.json and "
            "trajectory_index.json. Checkpoint bases with a single "
            "rlt_stage2_components/replay_buffer/rank_* directory are accepted."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text summary.",
    )
    parser.add_argument(
        "--top-episodes",
        type=int,
        default=10,
        help="Number of episode summaries to print in text mode.",
    )
    return parser.parse_args()


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "inspect_rlt_replay.py requires torch to read trajectory_*.pt files."
        ) from exc
    return torch


def _numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("inspect_rlt_replay.py requires numpy.") from exc
    return np


def _resolve_replay_dir(path: Path) -> Path:
    if path.is_file():
        raise ValueError(
            "RLT replay inspect expects a TrajectoryReplayBuffer directory, not "
            f"a file: {path}"
        )
    if _is_replay_dir(path):
        return path

    candidates = []
    for relative in (
        Path("rlt_stage2_components/replay_buffer"),
        Path("replay_buffer"),
    ):
        root = path / relative
        if root.is_dir():
            candidates.extend(
                child for child in sorted(root.glob("rank_*")) if _is_replay_dir(child)
            )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"{path} contains multiple replay rank directories; pass one explicitly."
        )
    raise FileNotFoundError(
        f"{path} is not a TrajectoryReplayBuffer directory. Expected "
        "metadata.json and trajectory_index.json."
    )


def _is_replay_dir(path: Path) -> bool:
    return (path / "metadata.json").is_file() and (
        path / "trajectory_index.json"
    ).is_file()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _load_replay_directory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = _torch()
    metadata = _load_json(path / "metadata.json")
    index_data = _load_json(path / "trajectory_index.json")
    trajectory_index = {
        int(key): value
        for key, value in index_data.get("trajectory_index", {}).items()
    }
    trajectory_ids = [int(item) for item in index_data.get("trajectory_id_list", [])]
    trajectory_format = metadata.get("trajectory_format", "pt")
    if trajectory_format != "pt":
        raise ValueError(
            f"Only pt trajectory replay is supported for RLT inspect, got {trajectory_format!r}."
        )

    trajectories = []
    for trajectory_id in trajectory_ids:
        info = trajectory_index.get(trajectory_id)
        if info is None:
            raise KeyError(f"trajectory_id {trajectory_id} is missing from index.")
        model_weights_id = info.get("model_weights_id", "")
        trajectory_path = path / f"trajectory_{trajectory_id}_{model_weights_id}.pt"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing trajectory file: {trajectory_path}")
        trajectory = torch.load(
            trajectory_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(trajectory, dict):
            raise TypeError(
                f"Expected dict in {trajectory_path}, got {type(trajectory).__name__}."
            )
        trajectories.append(trajectory)
    return metadata, trajectories


def _to_numpy(value: Any):
    np = _numpy()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_sample_field(value: Any):
    array = _to_numpy(value)
    if array.ndim >= 2:
        return array.reshape(-1, *array.shape[2:])
    return array.reshape(-1)


def _stack_forward_inputs(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    values: dict[str, list[Any]] = {key: [] for key in REQUIRED_FORWARD_INPUTS}
    missing: dict[str, int] = {}
    for trajectory in trajectories:
        forward_inputs = trajectory.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            raise KeyError("RLT replay trajectory is missing forward_inputs.")
        for key in REQUIRED_FORWARD_INPUTS:
            if key not in forward_inputs:
                missing[key] = missing.get(key, 0) + 1
                continue
            values[key].append(_flatten_sample_field(forward_inputs[key]))
    if missing:
        raise KeyError(f"RLT replay forward_inputs missing required fields: {missing}")
    return {key: np.concatenate(parts, axis=0) for key, parts in values.items()}


def _histogram(values, names: dict[int, str]) -> dict[str, int]:
    np = _numpy()
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(np.int64).reshape(-1), return_counts=True)
    return {
        names.get(int(value), str(int(value))): int(count)
        for value, count in zip(unique, counts)
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _inspect(metadata: dict[str, Any], trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    fields = _stack_forward_inputs(trajectories)
    rewards = fields["rewards"]
    dones = fields["dones"]
    intervention = fields["intervention"]
    source = fields["source"]
    source_chunk = fields["source_chunk"]
    collection_phase_id = fields["collection_phase_id"]
    success = fields["success"]
    intervention_flag = fields["intervention_flag"]
    episode_id = fields["episode_id"]
    step_id = fields["step_id"]

    size = int(rewards.shape[0])
    reward_flat = rewards.reshape(-1) if rewards.size else np.asarray([])
    transition_reward = rewards.sum(axis=1) if rewards.size else np.asarray([])
    human_chunk = np.logical_or(source_chunk == 2, source_chunk == 3)
    rl_chunk = source_chunk == 1
    base_chunk = source_chunk == 0

    return {
        "num_trajectories": int(len(trajectories)),
        "total_samples": int(metadata.get("total_samples", size)),
        "inspected_samples": size,
        "trajectory_format": metadata.get("trajectory_format", "pt"),
        "reward": {
            "mean_step": float(reward_flat.mean()) if reward_flat.size else 0.0,
            "min_step": float(reward_flat.min()) if reward_flat.size else 0.0,
            "max_step": float(reward_flat.max()) if reward_flat.size else 0.0,
            "mean_transition_sum": float(transition_reward.mean())
            if transition_reward.size
            else 0.0,
            "positive_transition_rate": _ratio((transition_reward > 0).sum(), size),
            "negative_transition_rate": _ratio((transition_reward < 0).sum(), size),
            "zero_transition_rate": _ratio((transition_reward == 0).sum(), size),
        },
        "done_rate": _ratio(np.asarray(dones).astype(bool).sum(), size),
        "success_rate": _ratio(np.asarray(success).astype(bool).sum(), size),
        "intervention_flag_rate": _ratio(
            np.asarray(intervention_flag).astype(bool).sum(),
            size,
        ),
        "intervention_action_rate": float(np.asarray(intervention).mean())
        if intervention.size
        else 0.0,
        "source": _histogram(source, SOURCE_NAMES),
        "source_chunk": {
            "base_rate": _ratio(base_chunk.sum(), source_chunk.size),
            "rl_rate": _ratio(rl_chunk.sum(), source_chunk.size),
            "human_or_mixed_rate": _ratio(human_chunk.sum(), source_chunk.size),
            "histogram": _histogram(source_chunk, SOURCE_NAMES),
        },
        "collection_phase": _histogram(collection_phase_id, PHASE_NAMES),
        "episode": _episode_summary(
            episode_id=episode_id,
            step_id=step_id,
            transition_reward=transition_reward,
            dones=dones,
            intervention_flag=intervention_flag,
            source_chunk=source_chunk,
        ),
    }


def _episode_summary(
    *,
    episode_id,
    step_id,
    transition_reward,
    dones,
    intervention_flag,
    source_chunk,
) -> list[dict[str, Any]]:
    np = _numpy()
    if episode_id.size == 0:
        return []
    episode_flat = episode_id.reshape(-1).astype(np.int64)
    step_flat = (
        step_id.reshape(-1).astype(np.int64)
        if step_id.size
        else np.zeros_like(episode_flat)
    )
    done_flat = (
        dones.reshape(-1).astype(bool)
        if dones.size
        else np.zeros_like(episode_flat, dtype=bool)
    )
    intervention_flat = (
        intervention_flag.reshape(-1).astype(bool)
        if intervention_flag.size
        else np.zeros_like(episode_flat, dtype=bool)
    )
    summaries = []
    for episode in np.unique(episode_flat):
        mask = episode_flat == episode
        if not mask.any():
            continue
        chunk = source_chunk[mask] if source_chunk.size else np.asarray([])
        human_rate = (
            float(np.logical_or(chunk == 2, chunk == 3).mean())
            if chunk.size
            else 0.0
        )
        rewards = transition_reward[mask] if transition_reward.size else np.asarray([])
        summaries.append(
            {
                "episode_id": int(episode),
                "transitions": int(mask.sum()),
                "min_step_id": int(step_flat[mask].min()) if step_flat.size else 0,
                "max_step_id": int(step_flat[mask].max()) if step_flat.size else 0,
                "reward_sum": float(rewards.sum()) if rewards.size else 0.0,
                "done_count": int(done_flat[mask].sum()) if done_flat.size else 0,
                "intervention_transition_rate": float(intervention_flat[mask].mean())
                if intervention_flat.size
                else 0.0,
                "human_or_mixed_chunk_rate": human_rate,
            }
        )
    return summaries


def _print_text(
    summary: dict[str, Any],
    *,
    source_path: Path,
    top_episodes: int,
) -> None:
    print(f"Replay: {source_path}")
    print(
        "trajectories={num_trajectories} total_samples={total_samples} "
        "inspected_samples={inspected_samples} format={trajectory_format}".format(
            **summary
        )
    )
    reward = summary["reward"]
    print(
        "reward: mean_step={mean:.4f} transition_sum_mean={sum_mean:.4f} "
        "positive={pos:.3f} zero={zero:.3f} negative={neg:.3f}".format(
            mean=reward["mean_step"],
            sum_mean=reward["mean_transition_sum"],
            pos=reward["positive_transition_rate"],
            zero=reward["zero_transition_rate"],
            neg=reward["negative_transition_rate"],
        )
    )
    print(
        "flags: done_rate={done:.3f} success_rate={success:.3f} "
        "intervention_flag_rate={intervention:.3f} "
        "intervention_action_rate={action:.3f}".format(
            done=summary["done_rate"],
            success=summary["success_rate"],
            intervention=summary["intervention_flag_rate"],
            action=summary["intervention_action_rate"],
        )
    )
    print(f"source: {summary['source']}")
    print(f"source_chunk: {summary['source_chunk']}")
    print(f"collection_phase: {summary['collection_phase']}")
    episodes = summary["episode"]
    print(f"episodes: {len(episodes)} unique ids")
    for item in episodes[:top_episodes]:
        print(
            "  episode={episode_id} transitions={transitions} "
            "steps={min_step_id}-{max_step_id} reward_sum={reward_sum:.3f} "
            "done_count={done_count} intervention_rate="
            "{intervention_transition_rate:.3f} "
            "human_chunk_rate={human_or_mixed_chunk_rate:.3f}".format(**item)
        )


def main() -> int:
    args = _parse_args()
    replay_dir = _resolve_replay_dir(args.path)
    metadata, trajectories = _load_replay_directory(replay_dir)
    summary = _inspect(metadata, trajectories)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text(
            summary,
            source_path=replay_dir,
            top_episodes=args.top_episodes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
