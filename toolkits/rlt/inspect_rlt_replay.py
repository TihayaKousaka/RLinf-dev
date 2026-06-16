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

"""Inspect saved RLT Stage2 replay buffers for real-world debugging."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an RLT Stage2 replay buffer snapshot or checkpoint."
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to buffer.pt, a replay autosave directory, or "
            "rlt_stage2_components/checkpoint_rank_*.pt."
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


def _load_torch_file(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "inspect_rlt_replay.py requires torch to read replay snapshots."
        ) from exc

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(state).__name__}.")
    return state


def _numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "inspect_rlt_replay.py requires numpy to inspect replay snapshots."
        ) from exc
    return np


def _resolve_replay_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "buffer.pt"
        if candidate.exists():
            return candidate
        candidates = sorted(path.glob("checkpoint_rank_*.pt"))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ValueError(
                f"{path} contains multiple checkpoint_rank_*.pt files; pass one explicitly."
            )
        raise FileNotFoundError(f"{path} does not contain buffer.pt or checkpoint_rank_*.pt.")
    return path


def _extract_replay_state(state: dict[str, Any]) -> dict[str, Any]:
    if "replay_buffer" in state:
        replay_state = state["replay_buffer"]
        if replay_state is None:
            raise ValueError("checkpoint has replay_buffer=None.")
        return replay_state
    if "size" in state and "rewards" in state:
        return state
    raise KeyError("Could not find replay buffer state in file.")


def _array(state: dict[str, Any], key: str, *, default=None):
    np = _numpy()
    if key not in state:
        if default is None:
            return None
        return np.asarray(default)
    value = state[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _histogram(values, names: dict[int, str]) -> dict[str, int]:
    np = _numpy()
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(np.int64).reshape(-1), return_counts=True)
    return {names.get(int(value), str(int(value))): int(count) for value, count in zip(unique, counts)}


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _inspect(replay_state: dict[str, Any]) -> dict[str, Any]:
    np = _numpy()
    size = int(replay_state.get("size", 0))
    capacity = int(replay_state.get("capacity", 0))
    chunk_length = int(replay_state.get("chunk_length", 0))
    action_chunk_dim = int(replay_state.get("action_chunk_dim", 0))

    rewards = _array(replay_state, "rewards", default=np.zeros((0, 0)))[:size]
    dones = _array(replay_state, "dones", default=np.zeros((0, 1)))[:size]
    intervention = _array(replay_state, "intervention", default=np.zeros((0, 1)))[:size]
    source = _array(replay_state, "source", default=np.zeros((0, 1)))[:size]
    source_chunk = _array(replay_state, "source_chunk", default=np.zeros((0, 0)))[:size]
    collection_phase_id = _array(
        replay_state,
        "collection_phase_id",
        default=np.zeros((0, 1)),
    )[:size]
    success = _array(replay_state, "success", default=np.zeros((0, 1)))[:size]
    intervention_flag = _array(
        replay_state,
        "intervention_flag",
        default=np.zeros((0, 1), dtype=np.bool_),
    )[:size]
    episode_id = _array(replay_state, "episode_id", default=np.zeros((0, 1)))[:size]
    step_id = _array(replay_state, "step_id", default=np.zeros((0, 1)))[:size]

    reward_flat = rewards.reshape(-1) if rewards is not None else np.asarray([])
    transition_reward = rewards.sum(axis=1) if rewards.size else np.asarray([])
    human_chunk = np.logical_or(source_chunk == 2, source_chunk == 3)
    rl_chunk = source_chunk == 1
    base_chunk = source_chunk == 0

    summary: dict[str, Any] = {
        "size": size,
        "capacity": capacity,
        "fill_ratio": _ratio(size, capacity),
        "chunk_length": chunk_length,
        "action_chunk_dim": action_chunk_dim,
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
    return summary


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
    step_flat = step_id.reshape(-1).astype(np.int64) if step_id.size else np.zeros_like(episode_flat)
    done_flat = dones.reshape(-1).astype(bool) if dones.size else np.zeros_like(episode_flat, dtype=bool)
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


def _print_text(summary: dict[str, Any], *, source_path: Path, top_episodes: int) -> None:
    print(f"Replay: {source_path}")
    print(
        "size={size} capacity={capacity} fill={fill:.3f} chunk_length={chunk}".format(
            size=summary["size"],
            capacity=summary["capacity"],
            fill=summary["fill_ratio"],
            chunk=summary["chunk_length"],
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
        "intervention_flag_rate={intervention:.3f} intervention_action_rate={action:.3f}".format(
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
            "  episode={episode_id} transitions={transitions} steps={min_step_id}-{max_step_id} "
            "reward_sum={reward_sum:.3f} done_count={done_count} "
            "intervention_rate={intervention_transition_rate:.3f} "
            "human_chunk_rate={human_or_mixed_chunk_rate:.3f}".format(**item)
        )


def main() -> int:
    args = _parse_args()
    replay_path = _resolve_replay_path(args.path)
    replay_state = _extract_replay_state(_load_torch_file(replay_path))
    summary = _inspect(replay_state)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text(summary, source_path=replay_path, top_episodes=args.top_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
