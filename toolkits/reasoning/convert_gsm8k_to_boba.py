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

"""Convert GSM8K-style data to the RLinf math/Boba JSONL format.

The output format matches the reasoning math dataset format used by the
existing Boba configs:

    {
      "prompt": "<｜User｜>\\n...\\n<｜Assistant｜><think>\\n",
      "task": "math",
      "query_id": "gsm8k-train-0",
      "solutions": ["\\boxed{72}"]
    }

Example:
    python toolkits/reasoning/convert_gsm8k_to_boba.py \\
        --input /path/to/train.json \\
        --output /path/to/gsm8k_train_boba.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL records."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported input suffix {suffix!r}; use .json or .jsonl")


def get_field(
    record: dict[str, Any],
    primary_key: str,
    fallback_keys: tuple[str, ...],
    *,
    index: int,
) -> Any:
    """Return a field value using a primary key and ordered fallbacks."""
    for key in (primary_key, *fallback_keys):
        if key in record:
            return record[key]
    keys = ", ".join((primary_key, *fallback_keys))
    raise KeyError(f"item {index} does not contain any of: {keys}")


def normalize_answer(answer: Any) -> list[str]:
    """Normalize raw GSM8K final answers into boxed math references."""
    if isinstance(answer, list):
        return [boxed_answer(str(item)) for item in answer]
    return [boxed_answer(str(answer))]


def boxed_answer(answer: str) -> str:
    """Return a compact ``\\boxed{...}`` answer unless it is already boxed."""
    answer = answer.strip()
    if "####" in answer:
        answer = answer.rsplit("####", maxsplit=1)[-1].strip()
    if "\\boxed" in answer:
        return answer
    return f"\\boxed{{{answer}}}"


def build_prompt(question: Any, prompt_template: str) -> str:
    """Render one RLinf rollout prompt."""
    if isinstance(question, list):
        raise ValueError(
            "Conversation-style prompts are not supported by this converter. "
            "Use RLinf data.apply_chat_template instead."
        )
    return prompt_template.format(question=str(question).strip())


def convert_records(
    records: list[dict[str, Any]],
    *,
    prompt_key: str,
    answer_key: str,
    query_id_key: str | None,
    query_prefix: str,
    prompt_template: str,
    task: str,
) -> list[dict[str, Any]]:
    """Convert input records to RLinf math/Boba records."""
    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question = get_field(
            record,
            prompt_key,
            ("question", "prompt", "problem"),
            index=index,
        )
        answer = get_field(
            record,
            answer_key,
            ("answer", "solutions", "label"),
            index=index,
        )
        if query_id_key is not None and query_id_key in record:
            query_id = str(record[query_id_key])
        else:
            query_id = f"{query_prefix}-{index}"

        converted.append(
            {
                "prompt": build_prompt(question, prompt_template),
                "task": task,
                "query_id": query_id,
                "solutions": normalize_answer(answer),
            }
        )
    return converted


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GSM8K-style JSON/JSONL data to RLinf math/Boba JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .json/.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output .jsonl")
    parser.add_argument(
        "--prompt-key",
        default="text",
        help="Input field containing the question. Defaults to 'text'.",
    )
    parser.add_argument(
        "--answer-key",
        default="label",
        help="Input field containing the final answer. Defaults to 'label'.",
    )
    parser.add_argument(
        "--query-id-key",
        default=None,
        help="Optional input field to reuse as query_id.",
    )
    parser.add_argument(
        "--query-prefix",
        default="gsm8k-train",
        help="Prefix used when query_id is generated from the row index.",
    )
    parser.add_argument(
        "--task",
        default="math",
        help="Task field written to each output record.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format string with a {question} placeholder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    converted = convert_records(
        records,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        query_id_key=args.query_id_key,
        query_prefix=args.query_prefix,
        prompt_template=args.prompt_template,
        task=args.task,
    )
    write_jsonl(args.output, converted)
    print(f"Converted {len(converted)} records: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
