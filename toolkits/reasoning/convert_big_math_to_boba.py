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

"""Convert Big-Math-RL-Verified-Processed data to RLinf math/Boba JSONL.

The RLinf reasoning dataset loader reads JSON/JSONL records. This converter
turns Hugging Face ``datasets`` rows or local parquet files into the same
format used by the existing GSM8K/Boba configs:

    {
      "prompt": "<｜User｜>\\n...\\n<｜Assistant｜><think>\\n",
      "task": "math",
      "query_id": "big-math-all-train-0",
      "solutions": ["\\boxed{10\\%}"],
      "source": "orca_math",
      "domain": ["Mathematics -> Applied Mathematics -> Math Word Problems"],
      "llama8b_solve_rate": 0.890625
    }

Examples:
    Download from Hugging Face and create a small validation split:

    python toolkits/reasoning/convert_big_math_to_boba.py \\
        --dataset open-r1/Big-Math-RL-Verified-Processed \\
        --subset all \\
        --split train \\
        --output /path/to/big_math_train_boba.jsonl \\
        --val-output /path/to/big_math_val_boba.jsonl \\
        --val-size 2048 \\
        --shuffle \\
        --seed 42

    Convert already downloaded parquet files:

    python toolkits/reasoning/convert_big_math_to_boba.py \\
        --input /path/to/all/train-00000-of-00001.parquet \\
        --output /path/to/big_math_train_boba.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "open-r1/Big-Math-RL-Verified-Processed"
DEFAULT_PROMPT_TEMPLATE = (
    "<｜User｜>\n"
    "{question} Please reason step by step, and put your final answer within "
    "\\boxed{{}}.\n"
    "<｜Assistant｜><think>\n"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            records.append(obj)
    return records


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None
        for key in ("data", "train", "examples", "records"):
            value = data.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise ValueError(
                f"{path} is a JSON object; expected a list or one of "
                "'data', 'train', 'examples', 'records' to contain a list"
            )
    else:
        raise ValueError(f"{path} must contain a JSON list or JSON object")

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item {index} is not a JSON object")
    return records


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading parquet files requires pyarrow. Install pyarrow or use "
            "--dataset to load through Hugging Face datasets."
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def _expand_input_paths(paths: list[str], recursive: bool) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        matches = glob.glob(raw_path, recursive=recursive)
        if not matches:
            matches = [raw_path]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                pattern = "**/*" if recursive else "*"
                expanded.extend(
                    sorted(
                        child
                        for child in path.glob(pattern)
                        if child.suffix.lower() in {".json", ".jsonl", ".parquet"}
                    )
                )
            else:
                expanded.append(path)
    return sorted(dict.fromkeys(expanded))


def load_local_records(paths: list[str], recursive: bool) -> list[dict[str, Any]]:
    """Load local JSON, JSONL, or parquet records."""
    records: list[dict[str, Any]] = []
    input_paths = _expand_input_paths(paths, recursive=recursive)
    if not input_paths:
        raise ValueError("No input files matched --input")

    for path in input_paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records.extend(_read_jsonl(path))
        elif suffix == ".json":
            records.extend(_read_json(path))
        elif suffix == ".parquet":
            records.extend(_read_parquet(path))
        else:
            raise ValueError(
                f"Unsupported input suffix {suffix!r} for {path}; use "
                ".json, .jsonl, or .parquet"
            )
    return records


def load_hf_records(
    dataset: str,
    subset: str,
    split: str,
    streaming: bool,
) -> list[dict[str, Any]]:
    """Load Big-Math records through Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading from Hugging Face requires the datasets package. "
            "Install it or use --input with local parquet files."
        ) from exc

    ds = load_dataset(dataset, subset, split=split, streaming=streaming)
    return [dict(item) for item in ds]


def _as_text(value: Any, *, field_name: str, index: int) -> str:
    if value is None:
        raise ValueError(f"item {index} has null {field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"item {index} has empty {field_name}")
    return text


def boxed_answer(answer: str) -> str:
    """Return a compact ``\\boxed{...}`` answer unless it already looks boxed."""
    answer = answer.strip()
    if "\\boxed" in answer:
        return answer
    return f"\\boxed{{{answer}}}"


def normalize_solutions(value: Any) -> list[str]:
    """Normalize a Big-Math solution field into RLinf reference answers."""
    if isinstance(value, list):
        return [boxed_answer(str(item)) for item in value if str(item).strip()]
    return [boxed_answer(str(value))]


def build_prompt(question: Any, prompt_template: str) -> str:
    """Render one RLinf rollout prompt."""
    return prompt_template.format(question=str(question).strip())


def _split_csv(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    items: set[str] = set()
    for value in values:
        items.update(part.strip() for part in value.split(",") if part.strip())
    return items or None


def _domain_matches(domain: Any, needles: set[str] | None) -> bool:
    if needles is None:
        return True
    if isinstance(domain, list):
        haystack = "\n".join(str(item) for item in domain)
    else:
        haystack = str(domain)
    return any(needle in haystack for needle in needles)


def _solve_rate_matches(
    solve_rate: Any,
    min_solve_rate: float | None,
    max_solve_rate: float | None,
) -> bool:
    if min_solve_rate is None and max_solve_rate is None:
        return True
    if solve_rate is None:
        return False
    rate = float(solve_rate)
    if min_solve_rate is not None and rate < min_solve_rate:
        return False
    if max_solve_rate is not None and rate > max_solve_rate:
        return False
    return True


def _make_query_id(
    record: dict[str, Any],
    query_id_key: str | None,
    query_prefix: str,
    index: int,
) -> str:
    if query_id_key is not None and query_id_key in record:
        value = record[query_id_key]
        if value is not None and str(value).strip():
            return str(value)
    return f"{query_prefix}-{index}"


def _load_tokenizer(tokenizer_path: str | None, trust_remote_code: bool):
    if tokenizer_path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "--max-prompt-tokens requires transformers to load --tokenizer."
        ) from exc
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def convert_records(
    records: Iterable[dict[str, Any]],
    *,
    prompt_key: str,
    solution_key: str,
    query_id_key: str | None,
    query_prefix: str,
    prompt_template: str,
    task: str,
    keep_metadata: bool,
    sources: set[str] | None,
    domains: set[str] | None,
    min_solve_rate: float | None,
    max_solve_rate: float | None,
    tokenizer: Any | None,
    max_prompt_tokens: int | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert Big-Math rows to RLinf math/Boba records."""
    converted: list[dict[str, Any]] = []
    stats = {
        "input": 0,
        "converted": 0,
        "skipped_missing": 0,
        "skipped_filter": 0,
        "skipped_prompt_length": 0,
    }

    for index, record in enumerate(records):
        stats["input"] += 1
        try:
            question = _as_text(
                record.get(prompt_key), field_name=prompt_key, index=index
            )
            solution = record.get(solution_key)
            solutions = normalize_solutions(solution)
        except ValueError:
            stats["skipped_missing"] += 1
            continue

        source = record.get("source")
        if sources is not None and str(source) not in sources:
            stats["skipped_filter"] += 1
            continue
        if not _domain_matches(record.get("domain"), domains):
            stats["skipped_filter"] += 1
            continue
        if not _solve_rate_matches(
            record.get("llama8b_solve_rate"),
            min_solve_rate,
            max_solve_rate,
        ):
            stats["skipped_filter"] += 1
            continue

        prompt = build_prompt(question, prompt_template)
        if tokenizer is not None and max_prompt_tokens is not None:
            prompt_tokens = tokenizer.encode(prompt)
            if len(prompt_tokens) > max_prompt_tokens:
                stats["skipped_prompt_length"] += 1
                continue

        output = {
            "prompt": prompt,
            "task": task,
            "query_id": _make_query_id(record, query_id_key, query_prefix, index),
            "solutions": solutions,
        }
        if keep_metadata:
            for key in ("source", "domain", "llama8b_solve_rate"):
                if key in record:
                    output[key] = record[key]

        converted.append(output)
        stats["converted"] += 1
        if limit is not None and len(converted) >= limit:
            break

    return converted, stats


def split_records(
    records: list[dict[str, Any]],
    *,
    val_size: int | None,
    val_ratio: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split converted records into train and validation lists."""
    if val_size is not None and val_ratio is not None:
        raise ValueError("Use only one of --val-size and --val-ratio")
    if val_ratio is not None:
        if not 0 < val_ratio < 1:
            raise ValueError("--val-ratio must be between 0 and 1")
        val_size = int(round(len(records) * val_ratio))
    if val_size is None or val_size <= 0:
        return records, []
    if val_size >= len(records):
        raise ValueError("--val-size must be smaller than the converted record count")
    return records[val_size:], records[:val_size]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert open-r1/Big-Math-RL-Verified-Processed parquet/HF data "
            "to RLinf math/Boba JSONL."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        default=None,
        help=f"Hugging Face dataset name. Suggested: {DEFAULT_DATASET}",
    )
    source_group.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Local .parquet/.json/.jsonl file, directory, or glob.",
    )
    parser.add_argument("--subset", default="all", help="HF dataset config/subset.")
    parser.add_argument("--split", default="train", help="HF split to load.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode before materializing rows.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively expand local input directories/globs.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Train output JSONL."
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Optional validation output JSONL.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Number of converted records to place in validation.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Fraction of converted records to place in validation.",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Input field containing the question.",
    )
    parser.add_argument(
        "--solution-key",
        default="solution",
        help="Input field containing the final answer.",
    )
    parser.add_argument(
        "--query-id-key",
        default=None,
        help="Optional input field to reuse as query_id.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="Prefix used when query_id is generated from the row index.",
    )
    parser.add_argument("--task", default="math", help="Task field in output records.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format string with a {question} placeholder.",
    )
    parser.add_argument(
        "--drop-metadata",
        action="store_true",
        help="Do not copy source/domain/llama8b_solve_rate into output records.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Keep only source values. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--domain-contains",
        action="append",
        default=None,
        help="Keep rows whose domain contains this text. May be repeated.",
    )
    parser.add_argument(
        "--min-llama8b-solve-rate",
        type=float,
        default=None,
        help="Keep rows with llama8b_solve_rate >= this value.",
    )
    parser.add_argument(
        "--max-llama8b-solve-rate",
        type=float,
        default=None,
        help="Keep rows with llama8b_solve_rate <= this value.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/name used for prompt token length filtering.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Drop records whose rendered prompt exceeds this token length.",
    )
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading --tokenizer.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max records to write.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before split.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "{question}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain a {question} placeholder")
    if args.max_prompt_tokens is not None and args.tokenizer is None:
        raise ValueError("--max-prompt-tokens requires --tokenizer")
    if args.val_output is None and (
        args.val_size is not None or args.val_ratio is not None
    ):
        raise ValueError("--val-size/--val-ratio require --val-output")

    if args.dataset is not None:
        raw_records = load_hf_records(
            dataset=args.dataset,
            subset=args.subset,
            split=args.split,
            streaming=args.streaming,
        )
        dataset_name = args.dataset.rsplit("/", maxsplit=1)[-1]
    else:
        raw_records = load_local_records(args.input, recursive=args.recursive)
        dataset_name = "big-math"

    query_prefix = args.query_prefix
    if query_prefix is None:
        query_prefix = f"{dataset_name}-{args.subset}-{args.split}".replace("/", "-")

    tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_trust_remote_code)
    converted, stats = convert_records(
        raw_records,
        prompt_key=args.prompt_key,
        solution_key=args.solution_key,
        query_id_key=args.query_id_key,
        query_prefix=query_prefix,
        prompt_template=args.prompt_template,
        task=args.task,
        keep_metadata=not args.drop_metadata,
        sources=_split_csv(args.source),
        domains=_split_csv(args.domain_contains),
        min_solve_rate=args.min_llama8b_solve_rate,
        max_solve_rate=args.max_llama8b_solve_rate,
        tokenizer=tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        limit=args.limit,
    )
    if not converted:
        raise ValueError(f"No records were converted. Stats: {stats}")

    if args.shuffle:
        random.Random(args.seed).shuffle(converted)

    train_records, val_records = split_records(
        converted,
        val_size=args.val_size,
        val_ratio=args.val_ratio,
    )
    write_jsonl(args.output, train_records)
    if args.val_output is not None:
        write_jsonl(args.val_output, val_records)

    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"Wrote train records: {len(train_records)} -> {args.output}")
    if args.val_output is not None:
        print(f"Wrote validation records: {len(val_records)} -> {args.val_output}")


if __name__ == "__main__":
    main()
