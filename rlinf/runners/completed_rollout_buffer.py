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

from dataclasses import dataclass
from typing import Optional

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle


@dataclass
class CompletedRollout:
    """Metadata for a completed rollout batch waiting to be trained."""

    handle: Handle
    step_id: int
    channel_id: int
    output_channel: Channel
    agent_metrics: dict
    head_version: Optional[int] = None
    tail_version: Optional[int] = None

    def is_stale(self, current_version: int, max_staleness: int) -> bool:
        if self.head_version is None:
            return False
        return int(current_version) - self.head_version > max_staleness


class CompletedRolloutBuffer:
    """FIFO buffer for completed rollout batches.

    The rollout payloads remain in their output channels. This buffer only keeps
    metadata and channel references so the runner can choose which completed
    batch the actor should consume next.
    """

    def __init__(self):
        self._items: list[CompletedRollout] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, rollout: CompletedRollout) -> None:
        self._items.append(rollout)

    def pop_trainable(
        self, current_version: int, max_staleness: int
    ) -> tuple[Optional[CompletedRollout], list[CompletedRollout]]:
        """Pop the first trainable rollout and return stale rollouts separately."""
        stale_rollouts = []
        while self._items:
            rollout = self._items.pop(0)
            if rollout.is_stale(current_version, max_staleness):
                stale_rollouts.append(rollout)
                continue
            return rollout, stale_rollouts
        return None, stale_rollouts
