#!/usr/bin/env python3
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

"""Extract a single-task LeRobot dataset from a multi-task LeRobot dataset.

The script keeps parquet image columns untouched. It only filters whole episodes
by ``task_index`` and rewrites LeRobot metadata/index columns, so datasets
created with LeRobot's native image writer stay loadable by RLinf/OpenPI.

Example:

    python toolkits/lerobot/filter_lerobot_by_task.py \\
        --src /mnt/public2/xiekaizhi/rlt-openpi-sim/data/libero_10 \\
        --dst /mnt/public2/xiekaizhi/rlt-openpi-sim/data/libero_10_task_7_white_mug_pudding \\
        --task-index 7 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow.") from exc
    return pa, pc, pq


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _episode_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    rel = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    ).format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset_root / rel


def _task_text(meta_dir: Path, task_index: int) -> str:
    for row in _read_jsonl(meta_dir / "tasks.jsonl"):
        if int(row.get("task_index", -1)) == int(task_index):
            return str(row.get("task", ""))
    return f"task_{task_index}"


def _selected_episodes(
    dataset_root: Path,
    info: dict[str, Any],
    *,
    task_index: int,
    pq: Any,
    pc: Any,
) -> list[tuple[dict[str, Any], Path]]:
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    out: list[tuple[dict[str, Any], Path]] = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        parquet_path = _episode_path(dataset_root, info, ep_idx)
        if not parquet_path.is_file():
            print(f"[filter] WARNING missing parquet: {parquet_path}", file=sys.stderr)
            continue
        table = pq.read_table(parquet_path, columns=["task_index"])
        task_col = table["task_index"]
        mask = pc.equal(task_col, task_index)
        if bool(pc.any(mask).as_py()):
            unique = set(task_col.to_pylist())
            if unique != {task_index}:
                raise ValueError(
                    f"Episode {ep_idx} mixes task_index values {sorted(unique)}. "
                    "This script filters whole episodes only."
                )
            out.append((ep, parquet_path))
    return out


def _shift_stat_values(values: Any, offset: int) -> Any:
    if isinstance(values, list):
        return [v + offset for v in values]
    if values is None:
        return values
    return values + offset


def _patch_episode_stats(
    stats: dict[str, Any],
    *,
    new_episode_index: int,
    old_index_min: int,
    new_index_min: int,
) -> dict[str, Any]:
    out = json.loads(json.dumps(stats))
    if "episode_index" in out:
        count = out["episode_index"].get("count", [1])
        out["episode_index"] = {
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
            "std": [0.0],
            "count": count,
        }
    if "task_index" in out:
        count = out["task_index"].get("count", [1])
        out["task_index"] = {
            "min": [0],
            "max": [0],
            "mean": [0.0],
            "std": [0.0],
            "count": count,
        }
    if "index" in out:
        offset = new_index_min - old_index_min
        out["index"]["min"] = _shift_stat_values(out["index"].get("min"), offset)
        out["index"]["max"] = _shift_stat_values(out["index"].get("max"), offset)
        out["index"]["mean"] = _shift_stat_values(out["index"].get("mean"), offset)
    return out


def filter_lerobot_by_task(
    src: Path,
    dst: Path,
    *,
    task_index: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not _is_lerobot_dataset(src):
        raise ValueError(f"Source is not a LeRobot dataset root: {src}")
    if src == dst:
        raise ValueError("--src and --dst must be different directories.")

    pa, pc, pq = _require_pyarrow()
    info = _read_json(src / "meta" / "info.json")
    task = _task_text(src / "meta", task_index)
    selected = _selected_episodes(src, info, task_index=task_index, pq=pq, pc=pc)
    if not selected:
        raise ValueError(f"No episodes found with task_index={task_index}.")

    print(f"[filter] Source: {src}")
    print(f"[filter] Dest:   {dst}")
    print(f"[filter] Task:   {task_index} -> {task!r}")
    print(f"[filter] Episodes: {len(selected)}")
    if dry_run:
        return len(selected)

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}. Use --overwrite.")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    chunks_size = int(info.get("chunks_size", 1000))
    global_frame_index = 0
    out_episodes: list[dict[str, Any]] = []
    out_episode_stats: list[dict[str, Any]] = []
    src_stats = {
        int(row["episode_index"]): row.get("stats", {})
        for row in _read_jsonl(src / "meta" / "episodes_stats.jsonl")
    }

    for new_ep_idx, (ep_meta, parquet_path) in enumerate(selected):
        old_ep_idx = int(ep_meta["episode_index"])
        table = pq.read_table(parquet_path)
        n_frames = table.num_rows

        old_index_values = (
            table["index"].to_pylist() if "index" in table.column_names else []
        )
        old_index_min = int(old_index_values[0]) if old_index_values else 0
        new_indices = list(range(global_frame_index, global_frame_index + n_frames))

        for col_name, values, pa_type in (
            ("episode_index", [new_ep_idx] * n_frames, pa.int64()),
            ("index", new_indices, pa.int64()),
            ("task_index", [0] * n_frames, pa.int64()),
        ):
            col_idx = table.column_names.index(col_name)
            table = table.set_column(col_idx, col_name, pa.array(values, type=pa_type))

        new_chunk = new_ep_idx // chunks_size
        out_dir = dst / "data" / f"chunk-{new_chunk:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_{new_ep_idx:06d}.parquet"
        pq.write_table(table, out_path)

        out_episodes.append(
            {
                **{k: v for k, v in ep_meta.items() if k != "episode_index"},
                "episode_index": new_ep_idx,
                "tasks": [task],
                "length": n_frames,
            }
        )
        if old_ep_idx in src_stats:
            out_episode_stats.append(
                {
                    "episode_index": new_ep_idx,
                    "stats": _patch_episode_stats(
                        src_stats[old_ep_idx],
                        new_episode_index=new_ep_idx,
                        old_index_min=old_index_min,
                        new_index_min=global_frame_index,
                    ),
                }
            )

        global_frame_index += n_frames

    out_info = dict(info)
    out_info.update(
        {
            "total_episodes": len(selected),
            "total_frames": global_frame_index,
            "total_tasks": 1,
            "total_chunks": max(1, (len(selected) + chunks_size - 1) // chunks_size),
            "chunks_size": chunks_size,
            "splits": {"train": f"0:{len(selected)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        }
    )
    if int(out_info.get("total_videos", 0)) == 0:
        out_info.pop("video_path", None)
    _write_json(dst / "meta" / "info.json", out_info)
    _write_jsonl(dst / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(dst / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    if out_episode_stats:
        _write_jsonl(dst / "meta" / "episodes_stats.jsonl", out_episode_stats)

    print(f"[filter] Done: {len(selected)} episodes, {global_frame_index} frames.")
    return len(selected)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Source dataset root.")
    parser.add_argument("--dst", required=True, type=Path, help="Output dataset root.")
    parser.add_argument("--task-index", required=True, type=int, help="Task index to keep.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --dst.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selection info.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        filter_lerobot_by_task(
            args.src,
            args.dst,
            task_index=args.task_index,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[filter] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
