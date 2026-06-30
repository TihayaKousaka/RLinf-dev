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

"""Convert LeRobot v2.1 LIBERO columns to RLinf/OpenPI LIBERO columns.

The public LeRobot LIBERO dataset can use keys such as::

    observation.images.image
    observation.images.wrist_image
    observation.state
    action

RLinf's current ``pi05_libero`` OpenPI data config expects the raw dataset
columns to be::

    image
    wrist_image
    state
    actions

This script copies a LeRobot dataset, renames those parquet columns, updates
metadata feature/stat keys, decodes LeRobot image-struct cells into uint8 image
arrays, and optionally injects a fixed prompt column.

Run from the repo root::

    python toolkits/lerobot/convert_libero_v21_to_rlinf.py \
        --src /path/to/libero_10_task6_raw \
        --dst /path/to/libero_10_task6_rlinf
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PROMPT = (
    "put the white mug on the plate and put the chocolate pudding to the right of the plate"
)

COLUMN_RENAMES = {
    "observation.images.image": "image",
    "observation.images.wrist_image": "wrist_image",
    "observation.state": "state",
    "action": "actions",
}

PROMPT_FEATURE = {
    "dtype": "string",
    "shape": [1],
    "names": None,
}

IMAGE_COLUMNS = ("image", "wrist_image")
KEEP_COLUMNS = {
    "image",
    "wrist_image",
    "state",
    "actions",
    "prompt",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a LeRobot v2.1 LIBERO dataset and rename columns for "
            "RLinf's pi05_libero OpenPI dataconfig."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source LeRobot dataset root containing meta/info.json and data/.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination dataset root to create.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to write into meta/tasks.jsonl and, optionally, prompt column.",
    )
    parser.add_argument(
        "--add-prompt-column",
        action="store_true",
        help="Also add a parquet prompt column. By default only tasks.jsonl is updated.",
    )
    parser.add_argument(
        "--overwrite-prompt",
        action="store_true",
        help="Replace an existing prompt parquet column instead of leaving it as-is.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --dst first if it already exists.",
    )
    parser.add_argument(
        "--keep-image-structs",
        action="store_true",
        help=(
            "Do not decode LeRobot image struct cells. Use only if your loader "
            "already decodes image columns before OpenPI transforms."
        ),
    )
    parser.add_argument(
        "--keep-unused-columns",
        action="store_true",
        help="Keep source columns not needed by RLinf pi05_libero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing --dst.",
    )
    return parser


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Run this in the same environment used "
            "for LeRobot/RLinf data loading, or install pyarrow."
        ) from exc
    return pa, pq


def _load_json(path: Path) -> dict[str, Any]:
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


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _resolve_parquet_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if files:
        return files
    return sorted((dataset_root / "data").glob("**/*.parquet"))


def _rename_mapping_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a feature/stat mapping with known keys renamed."""

    out = dict(mapping)
    for src_key, dst_key in COLUMN_RENAMES.items():
        if src_key not in out:
            continue
        if dst_key in out and dst_key != src_key:
            raise ValueError(
                f"Cannot rename metadata key {src_key!r} to {dst_key!r}: "
                "destination key already exists."
            )
        out[dst_key] = out.pop(src_key)
    return out


def _update_image_feature(feature: Any, *, decode_image_structs: bool) -> Any:
    if not decode_image_structs or not isinstance(feature, dict):
        return feature
    if feature.get("dtype") != "image":
        return feature
    out = dict(feature)
    # The converted parquet column stores raw HWC uint8 arrays, not HF Image
    # structs. OpenPI's LIBERO input transform accepts numpy-like image arrays.
    out["dtype"] = "uint8"
    return out


def _update_info_json(
    meta_dir: Path,
    *,
    add_prompt_column: bool,
    decode_image_structs: bool,
    keep_unused_columns: bool,
) -> None:
    info_path = meta_dir / "info.json"
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{info_path} does not contain a features mapping.")

    features = _rename_mapping_keys(features)
    if not keep_unused_columns:
        features = {key: value for key, value in features.items() if key in KEEP_COLUMNS}
    for key in IMAGE_COLUMNS:
        if key in features:
            features[key] = _update_image_feature(
                features[key], decode_image_structs=decode_image_structs
            )
    info["features"] = features
    if add_prompt_column:
        info["features"].setdefault("prompt", PROMPT_FEATURE)
    _write_json(info_path, info)


def _update_stats_json(meta_dir: Path) -> None:
    stats_path = meta_dir / "stats.json"
    if stats_path.is_file():
        stats = _load_json(stats_path)
        _write_json(stats_path, _rename_mapping_keys(stats))


def _update_episodes_stats_jsonl(meta_dir: Path) -> None:
    stats_path = meta_dir / "episodes_stats.jsonl"
    rows = _read_jsonl(stats_path)
    if not rows:
        return

    new_rows = []
    for row in rows:
        row = dict(row)
        if isinstance(row.get("stats"), dict):
            row["stats"] = _rename_mapping_keys(row["stats"])
        new_rows.append(row)
    _write_jsonl(stats_path, new_rows)


def _extract_task_indices(table: Any) -> set[int]:
    if "task_index" not in table.column_names:
        return set()
    values = table["task_index"].to_pylist()
    task_indices: set[int] = set()
    for value in values:
        if isinstance(value, list):
            if value:
                task_indices.add(int(value[0]))
        elif value is not None:
            task_indices.add(int(value))
    return task_indices


def _update_tasks_jsonl(meta_dir: Path, task_indices: set[int], prompt: str) -> None:
    tasks_path = meta_dir / "tasks.jsonl"
    rows = _read_jsonl(tasks_path)
    if not task_indices:
        task_indices = {
            int(row["task_index"])
            for row in rows
            if isinstance(row, dict) and "task_index" in row
        } or {0}

    by_index = {
        int(row["task_index"]): dict(row)
        for row in rows
        if isinstance(row, dict) and "task_index" in row
    }
    for task_index in sorted(task_indices):
        row = by_index.get(task_index, {"task_index": task_index})
        row["task"] = prompt
        by_index[task_index] = row

    _write_jsonl(tasks_path, [by_index[idx] for idx in sorted(by_index)])


def _rename_table_columns(table: Any) -> tuple[Any, list[tuple[str, str]]]:
    names = list(table.column_names)
    changed: list[tuple[str, str]] = []
    for idx, name in enumerate(names):
        if name not in COLUMN_RENAMES:
            continue
        new_name = COLUMN_RENAMES[name]
        if new_name in names and new_name != name:
            raise ValueError(
                f"Cannot rename parquet column {name!r} to {new_name!r}: "
                "destination column already exists."
            )
        names[idx] = new_name
        changed.append((name, new_name))
    if changed:
        table = table.rename_columns(names)
    return table, changed


def _first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_image_struct(value: Any) -> bool:
    return isinstance(value, dict) and ("bytes" in value or "path" in value)


def _resolve_image_path(path_value: str, *, dataset_root: Path, parquet_path: Path) -> Path:
    image_path = Path(path_value)
    if image_path.is_absolute():
        return image_path
    candidates = [
        dataset_root / image_path,
        parquet_path.parent / image_path,
        parquet_path.parent / path_value,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _decode_image_struct(value: Any, *, dataset_root: Path, parquet_path: Path) -> Any:
    if not _is_image_struct(value):
        return value

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to decode LeRobot image struct columns. "
            "Install Pillow or rerun with --keep-image-structs."
        ) from exc

    raw_bytes = value.get("bytes")
    if raw_bytes is not None:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            return image.convert("RGB")

    path_value = value.get("path")
    if path_value:
        image_path = _resolve_image_path(
            str(path_value), dataset_root=dataset_root, parquet_path=parquet_path
        )
        with Image.open(image_path) as image:
            return image.convert("RGB")

    raise ValueError(f"Invalid image struct without bytes/path in {parquet_path}")


def _uint8_nested_type(ndim: int, pa: Any) -> Any:
    data_type = pa.uint8()
    for _ in range(ndim):
        data_type = pa.list_(data_type)
    return data_type


def _decode_image_columns(table: Any, *, dataset_root: Path, parquet_path: Path, pa: Any) -> Any:
    for column_name in IMAGE_COLUMNS:
        if column_name not in table.column_names:
            continue

        values = table[column_name].to_pylist()
        first = _first_non_null(values)
        if not _is_image_struct(first):
            continue

        decoded_arrays = []
        for value in values:
            decoded = _decode_image_struct(
                value, dataset_root=dataset_root, parquet_path=parquet_path
            )
            array = np.asarray(decoded, dtype=np.uint8)
            if array.ndim != 3:
                raise ValueError(
                    f"Decoded image column {column_name!r} in {parquet_path} "
                    f"has shape {array.shape}; expected HWC."
                )
            decoded_arrays.append(array)

        stacked = np.stack(decoded_arrays, axis=0)
        column = pa.array(
            [frame.tolist() for frame in stacked],
            type=_uint8_nested_type(stacked.ndim - 1, pa),
        )
        col_idx = table.column_names.index(column_name)
        table = table.set_column(col_idx, column_name, column)
    return table


def _set_prompt_column(
    table: Any,
    *,
    prompt: str,
    overwrite_prompt: bool,
    pa: Any,
) -> tuple[Any, bool]:
    prompt_array = pa.array([prompt] * table.num_rows, type=pa.string())
    if "prompt" in table.column_names:
        if not overwrite_prompt:
            return table, False
        col_idx = table.column_names.index("prompt")
        return table.set_column(col_idx, "prompt", prompt_array), True
    return table.append_column("prompt", prompt_array), True


def _convert_parquets(
    dataset_root: Path,
    *,
    prompt: str,
    add_prompt_column: bool,
    overwrite_prompt: bool,
    decode_image_structs: bool,
    keep_unused_columns: bool,
) -> tuple[int, set[int]]:
    pa, pq = _require_pyarrow()
    parquet_files = _resolve_parquet_files(dataset_root)
    if not parquet_files:
        raise ValueError(f"No parquet files found under {dataset_root / 'data'}")

    all_task_indices: set[int] = set()
    renamed_count = 0
    prompt_count = 0
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        all_task_indices.update(_extract_task_indices(table))
        table, changed = _rename_table_columns(table)
        if not keep_unused_columns:
            keep_names = [name for name in table.column_names if name in KEEP_COLUMNS]
            table = table.select(keep_names)
        if decode_image_structs:
            table = _decode_image_columns(
                table, dataset_root=dataset_root, parquet_path=parquet_path, pa=pa
            )
        if changed:
            renamed_count += 1
        if add_prompt_column:
            table, prompt_changed = _set_prompt_column(
                table,
                prompt=prompt,
                overwrite_prompt=overwrite_prompt,
                pa=pa,
            )
            prompt_count += int(prompt_changed)
        pq.write_table(table, parquet_path)

    print(
        f"[convert] Rewrote {len(parquet_files)} parquet file(s); "
        f"renamed columns in {renamed_count}; prompt column changed in {prompt_count}."
    )
    return len(parquet_files), all_task_indices


def convert_dataset(
    src: Path,
    dst: Path,
    *,
    prompt: str,
    add_prompt_column: bool = False,
    overwrite_prompt: bool = False,
    decode_image_structs: bool = True,
    keep_unused_columns: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not _is_lerobot_dataset(src):
        raise ValueError(f"Source does not look like a LeRobot dataset root: {src}")
    if src == dst:
        raise ValueError("--src and --dst must be different directories.")

    parquet_files = _resolve_parquet_files(src)
    if not parquet_files:
        raise ValueError(f"No parquet files found under {src / 'data'}")

    print(f"[convert] Source: {src}")
    print(f"[convert] Dest:   {dst}")
    print(f"[convert] Parquet files: {len(parquet_files)}")
    print(f"[convert] Prompt: {prompt!r}")
    print(f"[convert] Add prompt column: {add_prompt_column}")
    print(f"[convert] Decode image structs: {decode_image_structs}")
    print(f"[convert] Keep unused columns: {keep_unused_columns}")

    if dry_run:
        return

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite.")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    _update_info_json(
        dst / "meta",
        add_prompt_column=add_prompt_column,
        decode_image_structs=decode_image_structs,
        keep_unused_columns=keep_unused_columns,
    )
    _update_stats_json(dst / "meta")
    _update_episodes_stats_jsonl(dst / "meta")
    _, task_indices = _convert_parquets(
        dst,
        prompt=prompt,
        add_prompt_column=add_prompt_column,
        overwrite_prompt=overwrite_prompt,
        decode_image_structs=decode_image_structs,
        keep_unused_columns=keep_unused_columns,
    )
    _update_tasks_jsonl(dst / "meta", task_indices, prompt)

    print("[convert] Done.")
    print("[convert] Required RLinf raw keys are now: image, wrist_image, state, actions.")
    print(f"[convert] Use this in YAML data.train_data_paths.dataset_path: {dst}")


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        convert_dataset(
            args.src,
            args.dst,
            prompt=args.prompt,
            add_prompt_column=args.add_prompt_column,
            overwrite_prompt=args.overwrite_prompt,
            decode_image_structs=not args.keep_image_structs,
            keep_unused_columns=args.keep_unused_columns,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[convert] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
