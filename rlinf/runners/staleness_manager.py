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
from threading import RLock


@dataclass
class RolloutStat:
    """Counters used by staleness-aware async rollout scheduling."""

    accepted: int = 0
    rejected: int = 0
    discarded: int = 0
    running: int = 0


class StalenessManager:
    """Capacity controller for AReaL-style async rollout submission.

    The current Search-R1 async runner consumes one completed rollout batch per
    actor update, so the unit here is a rollout batch rather than an individual
    sample. The same formula can later be reused with sample counts once a
    completed experience buffer is introduced.
    """

    def __init__(
        self,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
    ):
        if max_concurrent_rollouts < 1:
            raise ValueError("max_concurrent_rollouts must be >= 1")
        if consumer_batch_size < 1:
            raise ValueError("consumer_batch_size must be >= 1")
        if max_staleness < 0:
            raise ValueError("max_staleness must be >= 0")

        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness
        self._stat = RolloutStat()
        self._lock = RLock()

    def on_version_recovered(self, version: int) -> None:
        """Bound capacity correctly when resuming from a non-zero version."""
        with self._lock:
            self._stat.accepted = int(version) * self.consumer_batch_size
            self._stat.rejected = 0
            self._stat.discarded = 0
            self._stat.running = 0

    def get_capacity(self, current_version: int) -> int:
        """Return how many new rollout batches may be submitted now."""
        with self._lock:
            concurrency_capacity = self.max_concurrent_rollouts - self._stat.running
            produced = self._stat.accepted + self._stat.running
            staleness_capacity = (
                (self.max_staleness + int(current_version) + 1)
                * self.consumer_batch_size
                - produced
            )
            return min(concurrency_capacity, staleness_capacity)

    def on_rollout_submitted(self) -> None:
        with self._lock:
            self._stat.running += 1

    def on_rollout_accepted(self) -> None:
        with self._lock:
            self._stat.running -= 1
            self._stat.accepted += 1

    def on_rollout_rejected(self) -> None:
        with self._lock:
            self._stat.running -= 1
            self._stat.rejected += 1

    def on_rollout_discarded(self) -> None:
        with self._lock:
            self._stat.discarded += 1

    def get_stats(self, current_version: int) -> dict[str, int]:
        with self._lock:
            capacity = self.get_capacity(current_version)
            return {
                "accepted": self._stat.accepted,
                "rejected": self._stat.rejected,
                "discarded": self._stat.discarded,
                "running": self._stat.running,
                "capacity": capacity,
                "max_staleness": self.max_staleness,
            }
