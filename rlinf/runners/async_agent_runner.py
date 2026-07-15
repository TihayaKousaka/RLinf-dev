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

import logging
import time
from dataclasses import dataclass

from tqdm import tqdm

from rlinf.runners.completed_rollout_buffer import (
    CompletedRollout,
    CompletedRolloutBuffer,
)
from rlinf.runners.agent_runner import AgentRunner
from rlinf.runners.staleness_manager import StalenessManager
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.runner_utils import check_progress


@dataclass
class _SubmittedRollout:
    """Bookkeeping for a submitted agent rollout batch."""

    handle: Handle
    step_id: int
    channel_id: int
    output_channel: Channel


class AsyncAgentRunner(AgentRunner):
    """Agent runner that overlaps agent rollout generation with actor training.

    This is the first async-RL integration step for agent workloads. It keeps the
    existing ``AgentLoopWorker`` and actor training APIs intact, and only changes
    runner-side scheduling: several rollout batches can be in flight while the
    actor consumes completed batches from the existing channels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        async_cfg = self.cfg.runner.get("async_rl", {})
        self.max_concurrent_rollout_batches = int(
            async_cfg.get("max_concurrent_rollout_batches", 2)
        )
        self.prefill_rollout_batches = int(
            async_cfg.get(
                "prefill_rollout_batches", self.max_concurrent_rollout_batches
            )
        )
        if self.max_concurrent_rollout_batches < 1:
            raise ValueError(
                "runner.async_rl.max_concurrent_rollout_batches must be >= 1"
            )
        if self.prefill_rollout_batches < 1:
            raise ValueError("runner.async_rl.prefill_rollout_batches must be >= 1")
        self.prefill_rollout_batches = min(
            self.prefill_rollout_batches, self.max_concurrent_rollout_batches
        )
        self.max_head_offpolicyness = int(
            async_cfg.get(
                "max_head_offpolicyness",
                async_cfg.get("max_staleness", self.max_concurrent_rollout_batches),
            )
        )
        self.staleness_manager = StalenessManager(
            max_concurrent_rollouts=self.max_concurrent_rollout_batches,
            consumer_batch_size=int(async_cfg.get("consumer_batch_size", 1)),
            max_staleness=self.max_head_offpolicyness,
        )
        self.completed_rollout_buffer = CompletedRolloutBuffer()
        self._submitted_rollout_steps = 0
        self._running_rollouts: list[_SubmittedRollout] = []
        self._rollout_channels = [
            Channel.create(f"AsyncAgentRollout-{idx}")
            for idx in range(self.max_concurrent_rollout_batches)
        ]
        self._available_rollout_channel_ids = list(
            range(self.max_concurrent_rollout_batches)
        )
        self._train_iter = None

    def _sync_weights(self, version: int | None = None):
        if version is None:
            version = self.global_steps
        super()._sync_weights(version=version)

    def _start_rollout_services(self) -> None:
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        for solid_rollout_name, solid_rollout in self.solid_rollouts.items():
            solid_rollout.rollout_serverless(
                self.solid_generate_input_channels[solid_rollout_name],
                self.generate_output_channel,
            )
        for tool_worker in self.tool_workers:
            tool_worker.start_server()

    def _next_train_batch(self):
        if self._submitted_rollout_steps >= self.max_steps:
            return None
        if self._train_iter is None:
            epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
            self._train_iter = (
                batch for _ in epoch_iter for batch in self.train_dataloader
            )
        try:
            return next(self._train_iter)
        except StopIteration:
            return None

    def _submit_rollout_batch(self, batch) -> bool:
        if not self._available_rollout_channel_ids:
            return False
        channel_id = self._available_rollout_channel_ids.pop(0)
        output_channel = self._rollout_channels[channel_id]
        self._put_batch(batch, self.batch_split_num)
        handle = self.agent_loop.run_agentloop_rollout(
            input_channel=self.dataloader_channel,
            output_channel=output_channel,
        )
        self._running_rollouts.append(
            _SubmittedRollout(
                handle=handle,
                step_id=self._submitted_rollout_steps,
                channel_id=channel_id,
                output_channel=output_channel,
            )
        )
        self._submitted_rollout_steps += 1
        self.staleness_manager.on_rollout_submitted()
        return True

    def _release_rollout_channel_id(self, channel_id: int) -> None:
        self._available_rollout_channel_ids.append(channel_id)
        self._available_rollout_channel_ids.sort()

    def _fill_rollout_backlog(self, target_size: int | None = None) -> None:
        if target_size is None:
            target_size = self.max_concurrent_rollout_batches
        target_size = min(target_size, self.max_concurrent_rollout_batches)

        while (
            len(self._running_rollouts) < target_size
            and self._available_rollout_channel_ids
            and self.staleness_manager.get_capacity(self.global_steps) > 0
        ):
            batch = self._next_train_batch()
            if batch is None:
                return
            with self.timer("submit_rollout"):
                self._submit_rollout_batch(batch)

    def _get_optional_int_metric(self, metrics: dict, key: str) -> int | None:
        value = metrics.get(key)
        if value is None:
            return None
        return int(value)

    def _to_completed_rollout(
        self, submitted: _SubmittedRollout, agent_metrics: dict
    ) -> CompletedRollout:
        head_version = self._get_optional_int_metric(
            agent_metrics, "async/head_version"
        )
        tail_version = self._get_optional_int_metric(
            agent_metrics, "async/tail_version"
        )
        return CompletedRollout(
            handle=submitted.handle,
            step_id=submitted.step_id,
            channel_id=submitted.channel_id,
            output_channel=submitted.output_channel,
            agent_metrics=agent_metrics,
            head_version=head_version,
            tail_version=tail_version,
        )

    def _poll_completed_rollouts(self) -> None:
        still_running = []
        for submitted in self._running_rollouts:
            if not submitted.handle.done():
                still_running.append(submitted)
                continue
            agent_metrics = submitted.handle.wait()[0] or {}
            completed = self._to_completed_rollout(submitted, agent_metrics)
            self.completed_rollout_buffer.add(completed)
            self.staleness_manager.on_rollout_accepted()
        self._running_rollouts = still_running

    def _drain_completed_rollout(self, completed: CompletedRollout) -> None:
        for _ in range(self.cfg.data.rollout_batch_size):
            completed.output_channel.get()

    def _discard_completed_rollouts(
        self, completed_rollouts: list[CompletedRollout]
    ) -> None:
        for completed in completed_rollouts:
            self._drain_completed_rollout(completed)
            self._release_rollout_channel_id(completed.channel_id)
            self.staleness_manager.on_rollout_discarded()

    def _pop_trainable_completed_rollout(self) -> CompletedRollout | None:
        completed, stale_rollouts = self.completed_rollout_buffer.pop_trainable(
            current_version=self.global_steps,
            max_staleness=self.max_head_offpolicyness,
        )
        self._discard_completed_rollouts(stale_rollouts)
        return completed

    def _wait_trainable_rollout(self) -> CompletedRollout:
        while True:
            self._poll_completed_rollouts()
            completed = self._pop_trainable_completed_rollout()
            if completed is not None:
                return completed
            self._fill_rollout_backlog()
            if not self._running_rollouts and len(self.completed_rollout_buffer) == 0:
                raise RuntimeError(
                    "No running rollouts are available. "
                    "The train dataloader may be exhausted before max_steps."
                )
            time.sleep(0.05)

    def _run_actor_training(self, rollout_channel: Channel):
        if self.recompute_logprobs:
            infer_handle = self.inference.run_inference(
                input_channel=rollout_channel,
                output_channel=self.inference_channel,
                compute_ref_logprobs=self.compute_ref_logprobs,
            )
            inference_channel = self.inference_channel
        else:
            infer_handle = None
            inference_channel = rollout_channel

        actor_handle = self.actor.run_training(input_channel=inference_channel)
        return actor_handle, infer_handle

    def run(self):
        if self.reward is not None:
            raise NotImplementedError(
                "AsyncAgentRunner currently supports agent-loop rewards only. "
                "RewardWorker support should be added with an async experience buffer."
            )
        if self.is_pipeline:
            raise NotImplementedError(
                "AsyncAgentRunner does not support pipeline mode yet."
            )

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.run_timer.start_time()
        self._submitted_rollout_steps = self.global_steps
        self.staleness_manager.on_version_recovered(self.global_steps)
        self._running_rollouts.clear()
        self.completed_rollout_buffer = CompletedRolloutBuffer()
        self._available_rollout_channel_ids = list(
            range(self.max_concurrent_rollout_batches)
        )
        self._train_iter = None
        self._start_rollout_services()
        try:
            self.actor.set_current_version(self.global_steps).wait()
            with self.timer("sync_weights"):
                self._sync_weights()
            self._fill_rollout_backlog(self.prefill_rollout_batches)

            while self.global_steps < self.max_steps:
                with self.timer("step"):
                    with self.timer("wait_rollout"):
                        completed = self._wait_trainable_rollout()
                        agent_metrics = completed.agent_metrics

                    actor_handle, infer_handle = self._run_actor_training(
                        completed.output_channel
                    )

                    # Keep rollout GPUs busy while the actor consumes the ready batch.
                    self._fill_rollout_backlog()

                    metrics = actor_handle.wait()
                    self._release_rollout_channel_id(completed.channel_id)
                    actor_rollout_metrics = metrics[0][0]
                    actor_training_metrics = metrics[0][1]
                    self.global_steps += 1
                    self.actor.set_current_version(self.global_steps).wait()

                    with self.timer("sync_weights"):
                        self._sync_weights()
                    self._fill_rollout_backlog()

                    run_time_exceeded = self.run_timer.is_finished()
                    _, save_model, is_train_end = check_progress(
                        self.global_steps,
                        self.max_steps,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=run_time_exceeded,
                    )

                    if save_model:
                        self._save_checkpoint()

                    if is_train_end:
                        logging.info(
                            "Step limit given by max_steps=%s reached. Stopping run",
                            self.max_steps,
                        )
                        return

                    if run_time_exceeded:
                        logging.info(
                            "Time limit given by run_timer=%s reached. Stopping run",
                            self.run_timer,
                        )
                        return

                time_metrics = self.timer.consume_durations()
                time_metrics["rollout"] = completed.handle.consume_duration()
                time_metrics["training"] = actor_handle.consume_duration()
                if infer_handle is not None:
                    time_metrics["inference"] = infer_handle.consume_duration(
                        reduction_type="min"
                    )

                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                staleness_stats = {
                    f"async_manager/{k}": v
                    for k, v in self.staleness_manager.get_stats(
                        self.global_steps
                    ).items()
                }
                staleness_stats["async_manager/completed_buffer_size"] = len(
                    self.completed_rollout_buffer
                )
                agent_metrics.update(staleness_stats)

                self.metric_logger.log(agent_metrics, logging_steps)
                self.metric_logger.log(log_time_metrics, logging_steps)
                if actor_rollout_metrics is not None:
                    rollout_metrics = {
                        f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
                    }
                    self.metric_logger.log(rollout_metrics, logging_steps)
                for i in range(self.cfg.algorithm.n_minibatches):
                    training_metrics = {
                        f"train/{k}": v for k, v in actor_training_metrics[i].items()
                    }
                    self.metric_logger.log(training_metrics, logging_steps + i)

                logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}
                logging_metrics.update(agent_metrics)
                if actor_rollout_metrics is not None:
                    logging_metrics.update(actor_rollout_metrics)
                logging_metrics.update(actor_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics, refresh=False)
                global_pbar.update(1)
        finally:
            for tool_worker in self.tool_workers:
                tool_worker.stop_server()
            self.metric_logger.finish()
