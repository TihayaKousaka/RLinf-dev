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

import json
import importlib.util
from pathlib import Path

STATUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "rlt_stage2"
    / "status.py"
)
SPEC = importlib.util.spec_from_file_location("rlt_stage2_status", STATUS_PATH)
assert SPEC is not None and SPEC.loader is not None
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)

PHASE_ONLINE = status.PHASE_ONLINE
PHASE_WARMUP = status.PHASE_WARMUP
PHASE_WARMUP_WAIT_ONLINE = status.PHASE_WARMUP_WAIT_ONLINE
phase_id = status.phase_id
resolve_rollout_phase = status.resolve_rollout_phase
resolve_training_phase = status.resolve_training_phase
write_status_json = status.write_status_json


def test_resolve_training_phase_tracks_buffer_and_online_gate():
    assert (
        resolve_training_phase(buffer_ready=False, ready_for_online=False)
        == PHASE_WARMUP
    )
    assert (
        resolve_training_phase(buffer_ready=True, ready_for_online=False)
        == PHASE_WARMUP_WAIT_ONLINE
    )
    assert (
        resolve_training_phase(buffer_ready=True, ready_for_online=True)
        == PHASE_ONLINE
    )


def test_resolve_rollout_phase_tracks_student_control():
    assert (
        resolve_rollout_phase(ready_for_online=False, student_control_rate=0.0)
        == PHASE_WARMUP
    )
    assert (
        resolve_rollout_phase(ready_for_online=True, student_control_rate=0.0)
        == PHASE_WARMUP_WAIT_ONLINE
    )
    assert (
        resolve_rollout_phase(ready_for_online=True, student_control_rate=0.5)
        == PHASE_ONLINE
    )


def test_write_status_json_atomically(tmp_path):
    path = tmp_path / "status" / "rlt_status.json"
    write_status_json(
        str(path),
        {
            "phase": PHASE_ONLINE,
            "phase_id": phase_id(PHASE_ONLINE),
        },
    )

    payload = json.loads(path.read_text())
    assert payload["phase"] == PHASE_ONLINE
    assert payload["phase_id"] == 2
    assert not path.with_suffix(".json.tmp").exists()
