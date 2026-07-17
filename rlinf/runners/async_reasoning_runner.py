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
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.runners.staleness_manager import StalenessManager
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.runner_utils import check_progress


@dataclass
class _SubmittedRollout:
    """Bookkeeping for a submitted reasoning rollout batch."""

    handle: Handle
    step_id: int
    channel_id: int
    output_channel: Channel
    submitted_version: int
    scheduler_handle: Handle | None = None


@dataclass
class _TrainingChainResult:
    """Handles and metrics produced while training one completed rollout."""

    actor_handle: Handle | None
    actor_infer_handle: Handle | None
    critic_infer_handle: Handle | None
    critic_train_handle: Handle | None
    reward_handle: Handle
    actor_rollout_metrics: dict | None
    actor_training_metrics: list[dict] | None
    critic_training_metrics: list[dict] | None


class AsyncReasoningRunner(ReasoningRunner):
    """Reasoning runner that overlaps rollout generation with PPO training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        async_cfg = self.cfg.runner.get("async_rl", {})
        self.max_concurrent_rollout_batches = int(
            async_cfg.get("max_concurrent_rollout_batches", 1)
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
        if self.recompute_logprobs:
            raise ValueError(
                "AsyncReasoningRunner requires algorithm.recompute_logprobs=false "
                "so PPO old_logprobs come from the behavior rollout policy."
            )
        if not self.cfg.rollout.return_logprobs:
            raise ValueError(
                "AsyncReasoningRunner requires rollout.return_logprobs=true."
            )
        if self.component_placement.is_collocated:
            raise ValueError(
                "AsyncReasoningRunner requires disaggregated placement so rollout "
                "generation can overlap with actor/critic training on separate GPUs."
            )

        self.staleness_manager = StalenessManager(
            max_concurrent_rollouts=self.max_concurrent_rollout_batches,
            consumer_batch_size=int(async_cfg.get("consumer_batch_size", 1)),
            max_staleness=self.max_head_offpolicyness,
        )
        self.completed_rollout_buffer = CompletedRolloutBuffer()
        self._submitted_rollout_steps = 0
        self._running_rollouts: list[_SubmittedRollout] = []
        # One extra channel lets the trainer consume batch A while rollout batch B
        # is being generated even when max_concurrent_rollout_batches == 1.
        self._rollout_channel_count = self.max_concurrent_rollout_batches + 1
        self._rollout_channels = [
            Channel.create(f"AsyncReasoningRollout-{idx}")
            for idx in range(self._rollout_channel_count)
        ]
        self._available_rollout_channel_ids = list(range(self._rollout_channel_count))
        self._train_iter = None
        self._rollout_weight_version: int | None = None
        self._submit_rollout_duration = 0.0
        self._sync_rollout_duration = 0.0
        self._sync_inference_duration = 0.0

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

    def _sync_inference_weights(self) -> None:
        start_time = time.perf_counter()
        if self.has_dedicated_actor_inference:
            self.actor.sync_model_to_inference()
            self.actor_inference.sync_model_from_actor().wait()

        if self.has_dedicated_critic_inference:
            self.critic.sync_model_to_inference()
            self.critic_inference.sync_model_from_actor().wait()
        self._sync_inference_duration += time.perf_counter() - start_time

    def _sync_rollout_weights_if_idle(self, version: int) -> bool:
        if self._rollout_weight_version == version:
            return True
        if self._running_rollouts:
            return False

        start_time = time.perf_counter()
        self.actor.sync_model_to_rollout()
        self.rollout.sync_model_from_actor(version=version).wait()
        self.actor.del_reshard_state_dict().wait()
        self._rollout_weight_version = int(version)
        self._sync_rollout_duration += time.perf_counter() - start_time
        return True

    def _submit_rollout_batch(self, batch, submitted_version: int) -> bool:
        if not self._available_rollout_channel_ids:
            return False
        channel_id = self._available_rollout_channel_ids.pop(0)
        output_channel = self._rollout_channels[channel_id]
        self._put_batch(batch)

        scheduler_handle = None
        if self.scheduler is not None:
            scheduler_handle = self.scheduler.schedule()

        handle = self.rollout.rollout(
            input_channel=self.dataloader_channel,
            output_channel=output_channel,
        )
        self._running_rollouts.append(
            _SubmittedRollout(
                handle=handle,
                step_id=self._submitted_rollout_steps,
                channel_id=channel_id,
                output_channel=output_channel,
                submitted_version=int(submitted_version),
                scheduler_handle=scheduler_handle,
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
            if not self._sync_rollout_weights_if_idle(self.global_steps):
                return
            batch = self._next_train_batch()
            if batch is None:
                return
            start_time = time.perf_counter()
            self._submit_rollout_batch(batch, self._rollout_weight_version)
            self._submit_rollout_duration += time.perf_counter() - start_time

    def _to_completed_rollout(self, submitted: _SubmittedRollout) -> CompletedRollout:
        metrics = {
            "async/submitted_version": submitted.submitted_version,
            "async/rollout_step_id": submitted.step_id,
        }
        return CompletedRollout(
            handle=submitted.handle,
            step_id=submitted.step_id,
            channel_id=submitted.channel_id,
            output_channel=submitted.output_channel,
            agent_metrics=metrics,
            head_version=submitted.submitted_version,
            tail_version=submitted.submitted_version,
        )

    def _poll_completed_rollouts(self) -> None:
        still_running = []
        for submitted in self._running_rollouts:
            if not submitted.handle.done():
                still_running.append(submitted)
                continue
            submitted.handle.wait()
            if submitted.scheduler_handle is not None:
                submitted.scheduler_handle.wait()
            completed = self._to_completed_rollout(submitted)
            self.completed_rollout_buffer.add(completed)
            self.staleness_manager.on_rollout_accepted()
        self._running_rollouts = still_running

    def _drain_completed_rollout(self, completed: CompletedRollout) -> None:
        target_num_sequences = (
            self.cfg.data.rollout_batch_size * self.cfg.algorithm.group_size
        )
        drained_num_sequences = 0
        while drained_num_sequences < target_num_sequences:
            rollout_result = completed.output_channel.get()
            drained_num_sequences += rollout_result.num_sequence
        if drained_num_sequences != target_num_sequences:
            raise RuntimeError(
                "Drained stale rollout has unexpected number of sequences: "
                f"{drained_num_sequences} vs {target_num_sequences}"
            )

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

    def _run_training_chain(self, completed: CompletedRollout) -> _TrainingChainResult:
        if self.reward is None:
            raise RuntimeError("AsyncReasoningRunner requires a reward worker.")

        with self.timer("sync_inference_weights"):
            self._sync_inference_weights()

        reward_handle: Handle = self.reward.compute_rewards(
            input_channel=completed.output_channel,
            output_channel=self.reward_channel,
        )

        actor_infer_handle = None
        actor_inference_channel = self.reward_channel

        if self.critic:
            critic_infer_handle: Handle = self.critic_inference.run_inference(
                input_channel=actor_inference_channel,
                output_channel=self.value_channel,
                do_offload=False,
            )
            training_input_channel = self.value_channel
        else:
            critic_infer_handle = None
            training_input_channel = actor_inference_channel

        critic_warmup_steps = self.cfg.algorithm.get("critic_warmup_steps", 0)
        critic_warmup = (
            critic_warmup_steps > 0 and self.global_steps < critic_warmup_steps
        )
        if critic_warmup:
            raise NotImplementedError("critic warm up is not implemented yet")

        if self.is_pipeline:
            if self.critic:
                actor_training_input_channel = training_input_channel[0]
                critic_training_input_channel = training_input_channel[1]
            else:
                actor_training_input_channel = training_input_channel
        else:
            if self.critic:
                critic_training_input_channel = training_input_channel
                critic_training_output_channel = self.critic_output_channel
                actor_training_input_channel = critic_training_output_channel
            else:
                actor_training_input_channel = training_input_channel

        if self.critic:
            critic_train_handle: Handle = self.critic.run_training(
                input_channel=critic_training_input_channel,
                output_channel=(
                    None if self.is_pipeline else critic_training_output_channel
                ),
                compute_rollout_metrics=False,
            )
        else:
            critic_train_handle = None

        actor_handle: Handle = self.actor.run_training(
            input_channel=actor_training_input_channel,
            do_offload=False,
        )

        actor_metrics = actor_handle.wait()
        critic_training_metrics = None
        if critic_train_handle is not None:
            critic_metrics = critic_train_handle.wait()
            _, critic_training_metrics = critic_metrics[0]

        actor_rollout_metrics = actor_metrics[0][0]
        actor_training_metrics = actor_metrics[0][1]

        return _TrainingChainResult(
            actor_handle=actor_handle,
            actor_infer_handle=actor_infer_handle,
            critic_infer_handle=critic_infer_handle,
            critic_train_handle=critic_train_handle,
            reward_handle=reward_handle,
            actor_rollout_metrics=actor_rollout_metrics,
            actor_training_metrics=actor_training_metrics,
            critic_training_metrics=critic_training_metrics,
        )

    def _log_step_metrics(
        self,
        completed: CompletedRollout,
        chain_result: _TrainingChainResult,
        train_version: int,
        logging_steps: int,
        global_pbar: tqdm,
    ) -> None:
        time_metrics = self.timer.consume_durations()
        if self._submit_rollout_duration > 0:
            time_metrics["submit_rollout"] = self._submit_rollout_duration
            self._submit_rollout_duration = 0.0
        if self._sync_rollout_duration > 0:
            time_metrics["sync_rollout_weights"] = self._sync_rollout_duration
            self._sync_rollout_duration = 0.0
        if self._sync_inference_duration > 0:
            time_metrics["sync_inference_weights"] = self._sync_inference_duration
            self._sync_inference_duration = 0.0

        time_metrics["rollout"] = completed.handle.consume_duration()
        time_metrics["actor/training"] = chain_result.actor_handle.consume_duration()
        time_metrics["reward"] = chain_result.reward_handle.consume_duration()
        if chain_result.actor_infer_handle is not None:
            chain_result.actor_infer_handle.wait()
            time_metrics["actor/inference"] = (
                chain_result.actor_infer_handle.consume_duration(reduction_type="min")
            )
        if chain_result.critic_infer_handle is not None:
            chain_result.critic_infer_handle.wait()
            time_metrics["critic/inference"] = (
                chain_result.critic_infer_handle.consume_duration()
            )
        if chain_result.critic_train_handle is not None:
            time_metrics["critic/training"] = (
                chain_result.critic_train_handle.consume_duration()
            )

        log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in (chain_result.actor_rollout_metrics or {}).items()
        }
        staleness = train_version - int(completed.head_version)
        async_metrics = {
            "async/current_version": train_version,
            "async/post_train_version": self.global_steps,
            "async/head_version": int(completed.head_version),
            "async/tail_version": int(completed.tail_version),
            "async/staleness": staleness,
            "async/completed_buffer_size": len(self.completed_rollout_buffer),
        }
        async_metrics.update(
            {
                f"async_manager/{k}": v
                for k, v in self.staleness_manager.get_stats(
                    self.global_steps
                ).items()
            }
        )
        async_metrics.update(completed.agent_metrics)

        self.metric_logger.log(log_time_metrics, logging_steps)
        self.metric_logger.log(rollout_metrics, logging_steps)
        self.metric_logger.log(async_metrics, logging_steps)

        for i in range(self.cfg.algorithm.n_minibatches):
            training_metrics = {}
            if chain_result.actor_training_metrics is not None:
                for k, v in chain_result.actor_training_metrics[i].items():
                    training_metrics[f"actor/training/{k}"] = v
            if chain_result.critic_training_metrics is not None:
                for k, v in chain_result.critic_training_metrics[i].items():
                    training_metrics[f"critic/training/{k}"] = v
            self.metric_logger.log(training_metrics, logging_steps + i)

        logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}
        logging_metrics.update(chain_result.actor_rollout_metrics or {})
        logging_metrics.update(async_metrics)
        if chain_result.actor_training_metrics is not None:
            logging_metrics.update(chain_result.actor_training_metrics[-1])
        if chain_result.critic_training_metrics is not None:
            logging_metrics.update(chain_result.critic_training_metrics[-1])

        global_pbar.set_postfix(logging_metrics, refresh=False)
        global_pbar.update(1)

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=1650,
        )

        self.run_timer.start_time()
        self._submitted_rollout_steps = self.global_steps
        self.staleness_manager.on_version_recovered(self.global_steps)
        self._running_rollouts.clear()
        self.completed_rollout_buffer = CompletedRolloutBuffer()
        self._available_rollout_channel_ids = list(range(self._rollout_channel_count))
        self._train_iter = None
        self._rollout_weight_version = None
        try:
            self._fill_rollout_backlog(self.prefill_rollout_batches)
            self.timer.consume_durations()
            self._submit_rollout_duration = 0.0
            self._sync_rollout_duration = 0.0
            self._sync_inference_duration = 0.0

            while self.global_steps < self.max_steps:
                with self.timer("step"):
                    with self.timer("wait_rollout"):
                        completed = self._wait_trainable_rollout()

                    self._fill_rollout_backlog()
                    train_version = self.global_steps
                    chain_result = self._run_training_chain(completed)
                    self._release_rollout_channel_id(completed.channel_id)
                    self.global_steps += 1

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
                    if run_time_exceeded:
                        logging.info(
                            "Time limit given by run_timer=%s reached. Stopping run",
                            self.run_timer,
                        )

                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                self._log_step_metrics(
                    completed=completed,
                    chain_result=chain_result,
                    train_version=train_version,
                    logging_steps=logging_steps,
                    global_pbar=global_pbar,
                )

                if is_train_end or run_time_exceeded:
                    return
        finally:
            self.metric_logger.finish()
