Two-Stage RLT on ManiSkill Joint Control
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This page explains how to run the **joint-control RLT pipeline** for
**ManiSkill PegInsertionSideWideClearance-v1** in RLinf. Unlike the standard
PPO/GRPO VLA fine-tuning recipe in ``maniskill.rst``, this example follows a
multi-stage post-training workflow:

**data collection -> OpenPI SFT base -> Stage1 RL token -> Stage2 online TD3**

The focus here is:

- collecting replayable joint-control LeRobot data
- preparing the OpenPI SFT base checkpoint
- running Stage1 RL token training
- running Stage2 online RL
- keeping slots for screenshots, curves, and videos in the final doc page

Task and Method Overview
------------------------

The current RLT ManiSkill setup uses:

- **Environment**: ManiSkill3 ``PegInsertionSideWideClearance-v1``
- **Control mode**: ``pd_joint_delta_pos``
- **Observation**: third-view RGB + wrist RGB + first 9 Panda qpos values
- **Action**: 10-step action chunk, 8-dim joint action per step
- **Prompt**: ``insert the peg in the hole``
- **Reward**: sparse ``only_success``

The key idea of RLT is to **freeze the large VLA backbone and only train a small RL head**.

1. **OpenPI SFT base**: first obtain an OpenPI policy that already outputs joint chunk actions.
2. **Stage1**: train an RL token encoder/decoder to learn a compact state representation from VLA embeddings.
3. **Stage2**: freeze both the VLA and RL token, and only train a direct Gaussian actor plus twin-Q critic.

Current implementation notes
----------------------------

The current RLinf integration keeps the learner path lightweight:

- rollout workers run the frozen OpenPI VLA and frozen RL-token encoder
- actor workers only train the Stage2 actor and critic
- rollout workers sync only ``actor.*`` weights from the learner

This matters for both performance and interpretation. If Stage2 behavior looks
wrong, first check whether the rollout-side cached features, learner-side
replay, and actor-only weight sync are aligned.

Prerequisites
-------------

1. Clone RLinf
~~~~~~~~~~~~~~

.. code-block:: bash

   # For mainland China users, use a mirror if needed
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker**

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Then switch to the correct environment inside the container:

.. code-block:: bash

   source switch_env openpi

**Option 2: Local environment**

.. code-block:: bash

   # Add --use-mirror if needed
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

3. Download ManiSkill assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd rlinf/envs/maniskill
   # Optional: export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

4. Prepare OpenPI / openpi-RLT dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example depends on OpenPI configs and RLT-specific dataconfig wiring. If your
environment does not already provide ``openpi-RLT``, make sure its source path is
available in ``PYTHONPATH``. The scripts in this repo will first try to import it from
``../openpi-RLT/src`` relative to RLinf.

Dataset Layout
--------------

Both Stage1 and Stage2 use the ``pi05_rlt_joint`` dataconfig.
The LeRobot dataset schema is:

- ``image``: main view image
- ``wrist_image``: wrist camera image
- ``state``: first 9 Panda qpos values
- ``actions``: 8-dim ``pd_joint_delta_pos`` action
- ``task``: language instruction

A recommended layout is:

.. code-block:: text

   /data/rlt_maniskill_joint/
   /data/rlt_maniskill_joint_videos/

If you use an absolute path, the collector writes directly there. If you use a relative
``repo_id``, set ``HF_LEROBOT_HOME`` first.

Step 0: Prepare the OpenPI SFT Base
-----------------------------------

RLT Stage1 and Stage2 do not start from scratch. You first need a joint-control OpenPI
SFT checkpoint that serves as the base policy for the later stages.

The repo already contains a matching config:

- ``examples/sft/config/rlt_maniskill_joint_pi05_sft.yaml``

Launch it with:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh rlt_maniskill_joint_pi05_sft

The key fields are:

- ``actor.model.model_path``: OpenPI base weights
- ``actor.openpi_data.repo_id``: logical repo id for the ManiSkill joint LeRobot dataset
- ``actor.model.openpi.config_name``: ``pi05_rlt_joint``

After training, you should get a checkpoint like:

.. code-block:: text

   logs/<time>/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor

Both Stage1 and Stage2 should point their ``model_path`` to this kind of SFT checkpoint.

Step 1: Prepare the Joint-Control Dataset
-----------------------------------------

The public RLT workflow assumes that you already have a LeRobot-format dataset
matching ``pi05_rlt_joint``.

At minimum, the dataset must provide:

- ``image``: main-view RGB image
- ``wrist_image``: wrist RGB image
- ``state``: first 9 Panda qpos values
- ``actions``: 8D ``pd_joint_delta_pos`` action
- ``task``: instruction text

Normalization statistics
~~~~~~~~~~~~~~~~~~~~~~~~

If this is a newly collected dataset, generate ``norm_stats.json`` before SFT or RLT:

.. code-block:: bash

   export HF_LEROBOT_HOME=/data
   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_rlt_joint \
     --repo-id rlt_maniskill_joint

If you collected into an absolute path, you can move the dataset under ``HF_LEROBOT_HOME``
and then run the tool with its repo name.

Step 2: Run Stage1 RL Token Training
------------------------------------

Stage1 trains the RL token encoder/decoder while keeping the large VLA frozen.
The main files are:

- ``examples/sft/config/rlt_stage1_maniskill_joint.yaml``
- launcher: ``examples/sft/train_rlt_stage1.sh``

1. Update the key paths first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At minimum, check:

- ``data.train_data_paths[0].dataset_path``: your joint LeRobot dataset
- ``actor.openpi_data.repo_id``: the dataset repo id used to place ``norm_stats.json`` under ``actor.model.model_path/<repo_id>/``
- ``actor.model.model_path``: the OpenPI SFT base checkpoint from Step 0

Launch command:

.. code-block:: bash

   bash examples/sft/train_rlt_stage1.sh

If you prefer Hydra overrides instead of editing yaml:

.. code-block:: bash

   bash examples/sft/train_rlt_stage1.sh rlt_stage1_maniskill_joint \
     data.train_data_paths[0].dataset_path=/data/rlt_maniskill_joint \
     actor.openpi_data.repo_id=rlt_maniskill_joint \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_2000/actor \
     runner.logger.logger_backends='[tensorboard]'

2. Core Stage1 settings
~~~~~~~~~~~~~~~~~~~~~~~

The default config currently uses:

- ``num_action_chunks: 10``
- ``action_dim: 8``
- ``embedding_dim: 2048``
- ``vla_finetune_alpha: 0.0``, meaning the VLA stays frozen

The main output artifact is:

.. code-block:: text

   logs/<time>/rlt_stage1_maniskill_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt

That ``rl_token_model.pt`` is the checkpoint required by Stage2.

3. Optional: evaluate the SFT base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before Stage2, it is useful to verify that the joint SFT path is sensible.
The repo contains a matching eval config:

- ``examples/embodiment/config/rlt_maniskill_joint_pi05_sft_eval.yaml``

Evaluate a joint SFT checkpoint with the standard wrapper:

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh rlt_maniskill_joint_pi05_sft_eval LIBERO \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor \
     rollout.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_xxx/actor

Step 3: Run Stage2 Online TD3 Training
--------------------------------------

Stage2 runs online RL with a frozen VLA, a frozen RL token encoder, and a small actor-critic.
The main files are:

- ``examples/embodiment/config/rlt_stage2_maniskill_joint.yaml``
- launcher: ``examples/embodiment/run_embodiment.sh``

1. Update the key Stage2 paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At minimum, check:

- ``actor.model.model_path``: the SFT base checkpoint
- ``rollout.expert_model.model_path``: expert/reference checkpoint
- ``actor.model.rlt_stage2.rl_token_path``: the ``rl_token_model.pt`` from Stage1
- ``actor.openpi_data.repo_id`` matches the subdirectory that stores ``norm_stats.json`` under each checkpoint root

It is also recommended to switch the logger backend to ``tensorboard`` and enable eval video:

.. code-block:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard"]

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

2. Launch command
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint LIBERO

To override paths directly from the command line:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint LIBERO \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     rollout.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     rollout.expert_model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_8000/actor \
     actor.model.rlt_stage2.rl_token_path=/path/to/rlt_stage1_maniskill_joint/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt \
     actor.openpi_data.repo_id=rlt_maniskill_joint \
     runner.logger.logger_backends='[tensorboard]' \
     env.eval.video_cfg.save_video=True

.. note::

   The second positional argument of ``run_embodiment.sh`` is written into
   ``ROBOT_PLATFORM``. This config uses ``policy_setup: panda-qpos`` internally,
   so keeping the wrapper interface unchanged is fine.

3. Important Stage2 hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some important defaults in the current config:

- ``algorithm.warmup_min_size: 1000``: algorithm-level warmup setting; used as the fallback when replay has no explicit ready threshold
- ``algorithm.replay_buffer.min_buffer_size: 600``: the current scheduler's replay-ready threshold; check both values when changing warmup
- ``algorithm.warmup_post_collect_updates: 30000``: critic warmup before the actor goes online
- ``algorithm.train_every_transitions: 5``: add training budget every 5 new replay transitions
- ``algorithm.max_updates_per_train_step: 1600``: cap actual learner updates in one runner step
- ``actor.model.rlt_stage2.buffer_capacity: 50000``: active sample cap for RLT replay; ``fill_ratio`` saturates here
- ``actor.model.rlt_stage2.replay_subsample_stride: 0``: use boundary-only replay by default; set a positive value to build step-stride windows from rollout step traces
- ``actor.model.rlt_stage2.actor_noise_sigma: 0.003``: fixed Gaussian std; eval uses the mean
- ``actor.model.rlt_stage2.ref_action_dropout: 0.4``: prevent the actor from merely copying the VLA reference
- ``actor.model.rlt_stage2.delta_weight: 0.1``: smoothness regularizer on within-chunk action deltas

4. Current rollout and replay semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important implementation details are:

- rollout workers open the student-control gate only after the synced learner
  update version reaches ``algorithm.warmup_post_collect_updates``
- before that gate opens, rollout executes the base VLA reference chunk
- the default public config uses boundary-only replay because
  ``replay_subsample_stride`` is ``0``
- when ``replay_subsample_stride > 0``, rollout emits per-step RLT traces and
  replay builds denser sliding windows without crossing episode boundaries
- the learner uses RLinf ``TrajectoryReplayBuffer``; each RLT transition is wrapped
  as a 1-sample trajectory and active samples are capped by ``buffer_capacity``
- when expert intervention executes a replacement action, replay stores the base
  ``ref_chunk``, the executed ``action_chunk``, and ``source_chunk``; actor loss
  uses ``action_chunk`` as the BC target on ``HUMAN/MIXED`` steps

5. Stage2 outputs
~~~~~~~~~~~~~~~~~

Logs, checkpoints, and eval videos usually end up in:

.. code-block:: text

   logs/<time>-rlt_stage2_maniskill_joint/
   logs/<time>-rlt_stage2_maniskill_joint/checkpoints/
   logs/<time>-rlt_stage2_maniskill_joint/video/eval/

Evaluation, Visualization, and Result Slots
--------------------------------------------

1. TensorBoard
~~~~~~~~~~~~~~

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

2. Metrics to prioritize
~~~~~~~~~~~~~~~~~~~~~~~~

For Stage2, prioritize:

- ``eval/success_once``: the main final metric
- ``env/success_once``: useful for online data collection diagnostics, but may be affected by intervention
- ``train/replay_buffer/size``: whether replay has passed the warmup threshold
- ``train/replay_buffer/fill_ratio``: whether active samples have reached ``buffer_capacity``
- ``train/replay_buffer/intervention_rate``: how much expert data is being injected
- ``train/rlt_stage2/pending_update_budget``: whether the learner is falling behind the incoming data
- ``env/rlt_ready_for_online`` / ``env/student_control_rate``: online gate and actual actor-control ratio
- ``env/expert_intervention_requested_rate`` / ``env/expert_intervention_actual_rate``: expert trigger and intervention rates
- ``env/deviation_rate``: how often rollout deviation/intervention checks fire

For this task, **do not treat train success as the final answer**. Fixed-reset-id
``eval/success_once`` is the more reliable checkpoint comparison metric.

3. Evaluation command
~~~~~~~~~~~~~~~~~~~~~

To evaluate a Stage2 checkpoint separately, reuse the same config:

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh rlt_stage2_maniskill_joint LIBERO \
     actor.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     rollout.model.model_path=/path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_1000/actor \
     actor.model.rlt_stage2.rl_token_path=/path/to/rlt_stage1_maniskill_joint/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt \
     runner.ckpt_path=/path/to/stage2_checkpoint.pt \
     env.eval.video_cfg.save_video=True

4. Screenshot / video placeholders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final page should ideally include:

- Stage2 ``eval/success_once`` curve screenshots
- success/failure episode videos from ``logs/.../video/eval``

Below is a placeholder media section with the same overall style as the current ManiSkill page.
You can later replace these with RLT-joint-specific figures and videos.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvla.png" style="width: 100%;"/>
       <p><em>Stage1 / Stage2 curve placeholder</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvlaoft.png" style="width: 100%;"/>
       <p><em>Evaluation screenshot placeholder</em></p>
     </div>
   </div>

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

Checklist
---------

If the pipeline does not behave as expected, first check that these are aligned:

- ``actor.model.model_path`` points to the intended OpenPI joint SFT base
- ``actor.model.rlt_stage2.rl_token_path`` comes from the matching Stage1 run
- ``actor.openpi_data.repo_id`` matches the checkpoint subdirectory that stores ``norm_stats.json``
- ``num_action_chunks`` / ``action_dim`` stay at ``10`` / ``8``
- dataset prompts are normalized to ``insert the peg in the hole``
- ``warmup_post_collect_updates`` is interpreted as completed learner updates,
  not runner steps

If Stage2 shows "train goes up, eval does not", inspect:

- whether the learner lags behind replay via ``pending_update_budget``
- whether intervention is too frequent or almost never triggered
- whether the base VLA, RL token, and norm stats are mismatched
- whether eval is using a fixed but unusually hard set of reset ids
- whether the policy was still in the base-VLA warmup regime when you expected
  student control
