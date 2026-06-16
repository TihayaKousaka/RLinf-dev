Real-World Franka Joint-Control RLT
===================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This page describes the RLinf real-world Franka joint-control RLT workflow. It
uses RLinf's own real-world environment, RLT Stage1/Stage2 model code, workers,
and configs.

The complete workflow is:

.. code-block:: text

   realworld joint LeRobot data
     -> OpenPI pi0.5 joint SFT
     -> SFT realworld evaluation
     -> Stage1 RL-token training
     -> Stage2 realworld online RLT

Environment
-----------

This example targets a Franka arm controlled through absolute joint targets.

- **Environment type**: ``realworld``
- **Task config**: ``examples/embodiment/config/env/realworld_rlt_joint_peg_insertion.yaml``
- **Stage2 config**: ``examples/embodiment/config/rlt_stage2_realworld_joint.yaml``
- **Robot**: Franka with two cameras and optional GELLO intervention
- **Default task**: ``insert the peg in the hole``
- **Action**: 8D absolute joint action

The action vector follows this layout:

.. code-block:: text

   action[0:7] = Franka 7 joint target positions, in radians
   action[7]   = gripper command

The observation adapter exposes the real-world joint dataset layout used by the
OpenPI and RLT configs:

.. code-block:: text

   state = gripper
         + joint_pos(7)
         + joint_vel(7)
         + tcp_force(3 xyz)
         + tcp_pose(7 xyz + quat xyzw)
         + tcp_torque(3)
         + tcp_vel(6 lin3 + ang3)

``main_camera`` is used as the third-person image view and ``wrist_camera`` is
used as the wrist image view. Keep these names aligned with the LeRobot dataset
and ``norm_stats.json`` used for SFT and Stage1.

Method Overview
---------------

RLT freezes the large VLA backbone during Stage2 and trains only a small online
RL head:

1. **OpenPI SFT base**: train or load an OpenPI pi0.5 checkpoint that already
   produces real-world joint action chunks.
2. **Stage1 RL token**: train the RL-token module offline from the same
   real-world joint dataset.
3. **Stage2 online RLT**: freeze the SFT base and RL token, then train the
   Stage2 Gaussian actor and twin-Q critic online on the real robot.

In the RLinf implementation, rollout workers run the frozen OpenPI policy and
the frozen RL-token encoder. Actor workers train the Stage2 actor/critic and
sync actor weights back to rollout.

Machine Roles
-------------

This page keeps the lab shorthand ``master`` and ``slave``. They are machine
roles only, not algorithmic concepts and not robot master/slave control.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Name
     - Typical rank
     - Responsibility
   * - ``master`` / GPU head
     - ``RLINF_NODE_RANK=0``
     - Ray head, actor/rollout/training, and the only node where the training entry script is submitted
   * - ``slave`` / robot control
     - ``RLINF_NODE_RANK=1``
     - Ray worker connected to Franka, cameras, GELLO, and keyboard input; runs env workers and the Franka controller

All nodes must be on the same reachable network. The slave must be able to open
``http://<robot_ip>/desk`` and connect to ``<head_ip>:6379``.

Dependency Installation
-----------------------

Master / GPU head
~~~~~~~~~~~~~~~~~

The master needs the RLinf OpenPI training environment used by SFT and Stage1.
It is recommended to follow the same OpenPI installation style as :doc:`pi0`,
because the validated real-world pipeline currently reuses that environment.
If you already completed SFT/Stage1, reuse the same environment directly;
otherwise prepare it with one of the following options.

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Switch to the OpenPI virtual environment via the built-in ``switch_env`` tool:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   cd /path/to/RLinf
   # For mainland China users, you can add `--use-mirror` to install.sh for better download speed.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

If SFT/Stage1 is already done, you can also simply reactivate the original environment, for example:

.. code:: bash

   cd /path/to/RLinf
   source <your_rlinf_openpi_venv>/bin/activate

Check that the master can import the model and RLinf stack:

.. code-block:: bash

   python - <<'PY'
   import rlinf
   print("rlinf ok:", rlinf.__file__)
   try:
       import openpi
       print("openpi ok:", openpi.__file__)
   except Exception as exc:
       print("openpi import failed:", exc)
   PY

Stage2 real-world RLT currently reuses the validated RLinf OpenPI environment
above; there is no separate ``--model rlt_stage2 --env franka`` installer
target. Note that this OpenPI environment is only for the master side; the
slave still needs its own Franka/ROS real-world dependencies.

Slave / robot control
~~~~~~~~~~~~~~~~~~~~~

The slave needs Franka, camera, GELLO, keyboard, and ROS dependencies. Follow
the same Franka setup style as :doc:`franka`, then install the RLinf Franka env:

.. code-block:: bash

   cd /path/to/RLinf

   # Ubuntu 20.04 + ROS Noetic is recommended for Franka control.
   bash requirements/install.sh embodied --env franka --venv franka-venv
   source franka-venv/bin/activate

The Franka installer defaults to ``LIBFRANKA_VERSION=0.15.0`` and
``FRANKA_ROS_VERSION=0.10.0``. If your robot firmware requires a different
compatible pair, set them before installation:

.. code-block:: bash

   export LIBFRANKA_VERSION=<compatible-libfranka-version>
   export FRANKA_ROS_VERSION=<compatible-franka-ros-version>
   bash requirements/install.sh embodied --env franka --venv franka-venv

If ROS, libfranka, franka_ros, and serl_franka_controllers are already installed
manually, skip ROS installation and source your workspace explicitly:

.. code-block:: bash

   export SKIP_ROS=1
   bash requirements/install.sh embodied --env franka --venv franka-venv
   source franka-venv/bin/activate
   source /opt/ros/noetic/setup.bash
   source <your_catkin_ws>/devel/setup.bash

Before starting Ray, verify that the slave can import the real-world runtime
dependencies:

.. code-block:: bash

   python - <<'PY'
   import evdev
   import pyrealsense2
   import serial
   import rlinf
   print("realworld deps ok")
   PY

Start Ray
---------

Ray captures the active Python interpreter and environment variables at
``ray start``. Activate the correct venv, source ROS on the slave, and export all
camera/GELLO/keyboard variables before starting Ray.

``ray_utils/realworld/setup_before_ray.sh`` is a template. Before sourcing it,
edit the network interface and venv activation line on each machine:

.. code-block:: bash

   # Optional: set this only when you need to pin communication to a specific NIC.
   # Example NIC names: eth0, eno1, enp134s0f0.
   export RLINF_COMM_NET_DEVICES="<nic_name>"
   source <your_venv_path>/bin/activate

If the machine has only one usable NIC, you can usually leave
``RLINF_COMM_NET_DEVICES`` unset.

Master:

.. code-block:: bash

   cd /path/to/RLinf
   source <your_rlinf_openpi_venv>/bin/activate
   source ray_utils/realworld/setup_before_ray.sh
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

Slave:

.. code-block:: bash

   cd /path/to/RLinf
   source franka-venv/bin/activate
   source ray_utils/realworld/setup_before_ray.sh

   # If franka-venv/bin/activate does not source ROS/catkin for you:
   # source /opt/ros/noetic/setup.bash
   # source <your_catkin_ws>/devel/setup.bash

   export RLINF_NODE_RANK=1
   export RLT_REALWORLD_ROBOT_IP=<Franka IP>
   export RLT_REALWORLD_GELLO_PORT=/dev/serial/by-id/<your-gello-port>
   export RLT_REALWORLD_MAIN_CAMERA_SERIAL=<main-camera-serial>
   export RLT_REALWORLD_WRIST_CAMERA_SERIAL=<wrist-camera-serial>
   export RLT_REALWORLD_MAIN_CAMERA_TYPE=realsense
   export RLT_REALWORLD_WRIST_CAMERA_TYPE=lumos
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventX

   ray start --address=<head_ip>:6379

Except for ``RLINF_NODE_RANK``, the hardware-facing fields above can also be
written directly in YAML or passed through
``cluster.node_groups[].env_configs[].env_vars``. Keeping them as shell
environment variables is mainly for faster on-site switching. ``RLINF_NODE_RANK``
itself still has to be set as a machine environment variable before
``ray start``.

Find and grant access to the keyboard event device on the slave:

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   sudo chmod 666 /dev/input/eventX

If the master and slave do not share the same RLinf checkout path, enable code
sync before launching the training script on the master:

.. code-block:: bash

   export RLINF_CODE_WORKING_DIR=auto

Run ``ray status`` and confirm that the number of nodes matches
``cluster.num_nodes``.

Hardware Preflight
------------------

Before any real robot run, check the following on site:

1. Franka Desk has no error and is in programmable-control mode.
2. The arm workspace is clear; the peg, hole, cameras, and cables will not be swept by the arm.
3. ``reset_joint_qpos`` is a safe reset joint posture.
4. ``critical_phase_reset_joint_qpos`` is a safe start posture for ``critical_phase``.
5. ``full_task_reset_joint_qpos`` is a safe task start posture for ``full_task``.
6. ``target_pos`` is the 3D xyz target used by the base env for automatic
   success checking. In the current Stage2 setup, both train and eval enable
   ``keyboard_reward_wrapper: single_stage``, so final success/failure is still
   usually decided by keyboard reward; however, ``target_pos`` should still be
   filled with calibrated values for automatic success checks, debugging, and
   pure automatic evaluation.
7. For the first smoke test, reduce ``max_joint_delta`` to a small value such as ``0.03``.
8. A human operator must watch the robot, logs, keyboard reward, and emergency stop throughout online training.

After Ray is started on the slave and hardware environment variables are set,
run the unified preflight check from the slave:

.. code-block:: bash

   python -m toolkits.realworld_check.check_realworld_rlt_stack

The script checks:

- whether key placeholders remain in ``rlt_stage2_realworld_joint.yaml`` and
  ``realworld_rlt_joint_peg_insertion.yaml``.
- Franka IP, controller readiness, 7D joint state, and TCP pose readability.
- ``main_camera`` and ``wrist_camera`` serial/type capture, including measured FPS.
- GELLO serial existence and 7D joint/gripper readability.
- whether ``RLINF_KEYBOARD_DEVICE`` is readable and supports ``a/b/c/v``.

For partial checks on one machine, skip unavailable hardware:

.. code-block:: bash

   python -m toolkits.realworld_check.check_realworld_rlt_stack \
     --skip-franka --skip-cameras

The original single-purpose scripts are still useful for interactive debugging:

.. code-block:: bash

   export FRANKA_ROBOT_IP=<Franka IP>
   python -m toolkits.realworld_check.test_franka_controller
   python -m toolkits.realworld_check.test_franka_camera
   python rlinf/envs/realworld/common/gello/gello_expert.py --port /dev/serial/by-id/<your-gello-port>

Do not start Stage2 if any hardware item reports ``FAIL``. Do not enable
``use_gello`` until GELLO data can be read reliably.

Configuration
-------------

Cluster placement
~~~~~~~~~~~~~~~~~

``examples/embodiment/config/rlt_stage2_realworld_joint.yaml`` defaults to a
two-node cluster:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       rollout:
         node_group: "4090"
         placement: 0
       env:
         node_group: realworld
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: realworld
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: "${oc.env:RLT_REALWORLD_ROBOT_IP,ROBOT_IP}"
               node_rank: 1

Model paths
~~~~~~~~~~~

Set these paths on the master before Stage2, or edit the YAML directly:

.. code-block:: bash

   export RLT_REALWORLD_STAGE2_BASE_PATH=/path/to/sft/checkpoints/global_step_xxx/actor
   export RLT_REALWORLD_STAGE1_RL_TOKEN_PATH=/path/to/rl_token_model.pt
   export RLT_REALWORLD_NORM_STATS_PATH=/path/to/norm_stats.json

The Stage2 base path must point to the SFT ``actor/`` directory. The RL-token
path must point to Stage1's ``actor/rl_token/rl_token_model.pt``.

Warmup and intervention
~~~~~~~~~~~~~~~~~~~~~~~

Stage2 warmup and online gating are configured in
``examples/embodiment/config/rlt_stage2_realworld_joint.yaml``:

.. code-block:: yaml

   algorithm:
     warmup_min_size: 100
     warmup_post_collect_updates: 1000
     intervention:
       enable: True
       mode: human_override

``warmup_min_size`` is the minimum replay transition/window count. It is not the
number of episodes. ``warmup_post_collect_updates`` is the number of learner
updates required after replay reaches the warmup size. ``human_override`` uses
the real-world human intervention action as the intervention source.

Teleoperation and keyboard reward are configured under both ``env.train`` and
``env.eval``:

.. code-block:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage
       use_spacemouse: False
       use_gello: True
       gello_port: "${oc.env:RLT_REALWORLD_GELLO_PORT,/dev/ttyUSB0}"
       gello_action_mode: joint_target
     eval:
       keyboard_reward_wrapper: single_stage
       use_spacemouse: False
       use_gello: True
       gello_port: "${oc.env:RLT_REALWORLD_GELLO_PORT,/dev/ttyUSB0}"
       gello_action_mode: joint_target

For pure-policy evaluation, set ``env.eval.use_gello`` to ``False``. During
early Stage2 training, keep keyboard reward enabled.

Task and intervention
~~~~~~~~~~~~~~~~~~~~~

Important task fields live in
``examples/embodiment/config/env/realworld_rlt_joint_peg_insertion.yaml``:

.. code-block:: yaml

   task_mode: critical_phase
   critical_phase_key: v
   record_prefix_before_critical_phase: false
   gello_action_mode: joint_target

   override_cfg:
     target_pos: TARGET_POS
     reset_joint_qpos: RESET_JOINT_QPOS
     critical_phase_reset_joint_qpos: CRITICAL_PHASE_RESET_JOINT_QPOS
     full_task_reset_joint_qpos: FULL_TASK_RESET_JOINT_QPOS
     max_joint_delta: 0.08

For the first hardware smoke test, use a smaller ``max_joint_delta`` such as
``0.03``. Keep ``keyboard_reward_wrapper: single_stage`` enabled while
calibrating ``target_pos`` and success thresholds.

These four placeholders are all real-world calibration values:

- ``target_pos``: the spatial target used by the base env for automatic success
  checking. Current Stage2 runs still primarily use keyboard reward for final
  success/failure.
- ``reset_joint_qpos``: the final 7D reset joint target that is actually used,
  in radians.
- ``critical_phase_reset_joint_qpos``: the reset joint target used in
  ``task_mode: critical_phase``.
- ``full_task_reset_joint_qpos``: the reset joint target used in
  ``task_mode: full_task``.

Camera and state semantics must match the training dataset:

.. code-block:: text

   extra_view_image -> main_camera, third-person view
   image            -> wrist_camera, wrist view

The task config expects:

.. code-block:: yaml

   main_image_key: "${oc.env:RLT_REALWORLD_MAIN_IMAGE_KEY,main_camera}"
   wrist_image_key: "${oc.env:RLT_REALWORLD_WRIST_IMAGE_KEY,wrist_camera}"

Training Stages
---------------

Stage 0: SFT
~~~~~~~~~~~~

Update ``examples/sft/config/rlt_realworld_joint_pi05_sft.yaml`` with the real
dataset path, OpenPI data config, model path, and normalization stats.
At minimum, check:

- ``data.train_data_paths[].dataset_path``
- ``actor.openpi_data.repo_id``
- ``actor.openpi_data.norm_stats_path``
- ``actor.model.model_path``
- ``runner.max_steps``
- ``runner.save_interval``

Then run:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh rlt_realworld_joint_pi05_sft

The key output is:

.. code-block:: text

   logs/<time>-rlt_realworld_joint_pi05_sft/rlt_realworld_joint_pi05_sft/checkpoints/global_step_xxx/actor

Stage 0.5: real-world SFT evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before online RL, evaluate the SFT policy on hardware:

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh rlt_realworld_joint_pi05_sft_eval

Before launching, update at least:

- ``cluster.node_groups[].hardware.configs[].robot_ip``
- ``cluster.node_groups[].hardware.configs[].camera_infos``
- ``actor.model.model_path``
- ``actor.model.openpi.config_name``
- ``actor.model.openpi_data.norm_stats_path``
- ``env.eval.override_cfg.target_pos``
- ``env.eval.override_cfg.reset_joint_qpos``
- ``env.eval.override_cfg.critical_phase_reset_joint_qpos``
- ``env.eval.override_cfg.full_task_reset_joint_qpos``
- ``env.eval.override_cfg.max_joint_delta``
- ``env.eval.max_episode_steps``
- ``env.eval.max_steps_per_rollout_epoch``

Do not continue to Stage2 if the SFT policy moves in the wrong direction,
jumps unexpectedly, or uses mismatched camera/action semantics.

The SFT evaluation should be pure policy evaluation. Keep ``use_gello`` and
``use_spacemouse`` disabled unless you are intentionally debugging intervention.

Stage 1: RL-token training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage1 is offline and does not require the robot. Update
``examples/sft/config/rlt_stage1_realworld_joint.yaml`` first:

- ``data.train_data_paths[].dataset_path``
- ``actor.openpi_data.repo_id``
- ``actor.openpi_data.norm_stats_path``
- ``actor.model.model_path``
- ``actor.model.rlt_stage1.config_name``
- ``runner.max_steps``
- ``runner.save_interval``

Then run:

.. code-block:: bash

   bash examples/sft/train_rlt_stage1.sh rlt_stage1_realworld_joint

The key output is:

.. code-block:: text

   logs/<time>/rlt_stage1_realworld_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt

Stage 2: online real-world RLT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the master/slave Ray cluster is ready and Stage2 paths are set, launch
from the master:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint

Before launching, check the Stage2 YAML fields from the sections above,
especially:

- ``cluster.num_nodes`` and ``cluster.node_groups``
- ``actor.model.model_path``
- ``actor.model.rlt_stage2.norm_stats_path``
- ``actor.model.rlt_stage2.rl_token_path``
- ``algorithm.warmup_min_size``
- ``algorithm.warmup_post_collect_updates``
- ``algorithm.intervention``
- ``env.train.task_mode``
- ``env.train.keyboard_reward_wrapper``
- ``env.train.use_gello``
- ``env.train.gello_action_mode``
- ``env.train.override_cfg.max_joint_delta``
- corresponding ``env.eval`` fields

For different hardware modes, copy the YAML instead of using many one-off Hydra
overrides at the robot:

.. code-block:: text

   examples/embodiment/config/rlt_stage2_realworld_joint_smoke.yaml
   examples/embodiment/config/rlt_stage2_realworld_joint_full_task.yaml

Launch copied configs with the same entry point:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_smoke
   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_full_task

Critical Phase and Full Task
----------------------------

The default mode is ``critical_phase``:

.. code-block:: yaml

   env:
     train:
       task_mode: critical_phase
       critical_phase_key: v
       record_prefix_before_critical_phase: false

In ``critical_phase`` mode:

- reset directly starts the critical phase
- warmup still collects frozen reference-policy data
- after online readiness, the Stage2 actor can control from the beginning of the episode
- this is suitable when the robot resets near the hole and only alignment/insertion should be optimized

In ``full_task`` mode:

- reset starts from the non-critical prefix
- the prefix is controlled by the SFT/base/reference policy; Stage2 does not control it
- the operator presses ``v`` near the hole or at the chosen critical-phase boundary
- replay recording starts from the critical phase by default
- after online readiness, Stage2 actor controls only in the critical phase

To use ``full_task``, update both train and eval:

.. code-block:: yaml

   env:
     train:
       task_mode: full_task
       critical_phase_key: v
       record_prefix_before_critical_phase: false
     eval:
       task_mode: full_task
       critical_phase_key: v
       record_prefix_before_critical_phase: false

Usually keep ``record_prefix_before_critical_phase`` as ``false``. Recording the
prefix can dilute the learning signal for the fine critical phase.

Warmup and Online Operation
---------------------------

Stage2 uses replay-buffer size and learner updates to gate online control:

- ``warmup_min_size`` counts replay transitions/windows, not episodes.
- ``warmup_post_collect_updates`` counts actor/critic learner update steps.
- With ``replay_subsample_stride: 0``, replay is built at chunk boundaries.

At runtime, Stage2 emits fixed-format online switching status lines:

.. code-block:: text

   [RLT_STATUS][actor] phase=warmup ready=0 buffer_ready=0 replay=42/100 update=0/1000 pending=0
   [RLT_STATUS][env] phase=warmup ready=0 critical=1.00 record=1.00 student=0.00

It also writes read-only status files:

.. code-block:: text

   ${runner.logger.log_path}/status/rlt_actor_status_rank0.json
   ${runner.logger.log_path}/status/rlt_env_status_rank0.json

On site, you can inspect them directly:

.. code-block:: bash

   watch -n 1 'cat ../results/status/rlt_actor_status_rank0.json; cat ../results/status/rlt_env_status_rank0.json'

``phase`` has three values:

1. ``warmup``: replay has not reached ``warmup_min_size`` yet, so Stage2 is
   still collecting warmup data.
2. ``warmup_wait_online``: replay is ready, but the learner is still finishing
   ``warmup_post_collect_updates``.
3. ``online``: ``update_step >= warmup_post_collect_updates`` and
   ``ready_for_online=true``; later attempts can let the Stage2 actor take over.

The operator flow is:

1. Before ``replay_size >= warmup_min_size``, let the frozen SFT/Stage1 reference
   policy run. GELLO is only a safety override.
2. After replay is full but before enough learner updates, keep using the same
   warmup behavior or wait for the learner.
3. Once both conditions are met, the next attempts may use the Stage2 actor in
   the critical phase. GELLO becomes a correction tool.
4. Use keyboard reward during early hardware runs: ``c`` for success,
   ``a`` for failure/abort, ``b`` for zero reward without ending, and ``v`` to
   enter the critical phase in ``full_task`` mode.

Keyboard semantics:

.. code-block:: text

   c = success, reward=1, end the current episode
   a = failure / danger / abort, reward=-1, end the current episode
   b = reward=0, do not end the episode
   v = enter critical phase in full_task mode

On-site rules:

1. Reset the peg/hole before each episode.
2. ``critical_phase`` mode does not require pressing ``v``.
3. In ``full_task`` mode, press ``v`` when the peg is near the hole or at the critical boundary.
4. During warmup, do not move GELLO when the reference policy is safe.
5. If the reference policy is slightly wrong but still safe, let it expose the issue.
6. If the robot may collide, get stuck, or enter a dangerous state, use GELLO or emergency stop, then press ``a``.
7. After online readiness, GELLO becomes a correction tool. Correct only the necessary segment, then release control.
8. Press ``c`` after successful insertion and ``a`` after failure or an unhelpful continuation.

The current joint-target wrapper does not have a "hold button to intervene"
switch. If GELLO readings differ from the policy action beyond the wrapper
threshold, the action can be overwritten. Keep the GELLO posture under control.

Reward and Success
------------------

The automatic success check is primarily configured by:

.. code-block:: yaml

   override_cfg:
     target_pos: [...]
     reward_threshold: [0.015, 0.015, 0.03, 0.2, 0.2, 0.2]
     check_orientation_success: false
     success_hold_steps: 1

The default joint peg-insertion setup mainly checks xyz position because
``check_orientation_success`` is ``false``.

Early hardware runs should keep keyboard reward enabled:

.. code-block:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage
     eval:
       keyboard_reward_wrapper: single_stage

If ``target_pos`` is not calibrated yet, manual keyboard reward is usually more
reliable than automatic success. This is also the current default Stage2 path:
both train and eval keep ``keyboard_reward_wrapper: single_stage`` enabled, and
the operator provides the final reward via ``a/b/c``. Once the automatic
success threshold is calibrated, consider disabling keyboard reward for
evaluation.

Run SOP
-------

Before each run:

1. Franka Desk has no error.
2. The slave has sourced the ROS workspace.
3. ``RLINF_NODE_RANK`` was set before ``ray start``. Hardware-facing fields
   such as ``RLT_REALWORLD_ROBOT_IP``, camera serials,
   ``RLT_REALWORLD_GELLO_PORT``, and ``RLINF_KEYBOARD_DEVICE`` must also be
   visible before env workers start; they can be provided either via shell
   ``export`` before ``ray start`` or via YAML / ``env_vars``.
4. ``ray status`` shows both nodes.
5. ``RLT_REALWORLD_STAGE2_BASE_PATH`` points to the SFT ``actor/`` directory.
6. ``RLT_REALWORLD_STAGE1_RL_TOKEN_PATH`` points to Stage1 ``rl_token_model.pt``.
7. ``RLT_REALWORLD_NORM_STATS_PATH`` points to the matching dataset's ``norm_stats.json``.
8. GELLO and keyboard event input are readable.
9. ``max_joint_delta`` is small for the first smoke test.
10. In ``full_task`` mode, ``v`` does not conflict with reward keys ``a/b/c``.

During the run:

1. Watch episode by episode; do not leave the robot unattended.
2. Watch the robot motion, not only logs.
3. Press ``c`` promptly on success and ``a`` promptly on clear failure.
4. Intervene only for necessary segments, especially in the critical phase.
5. Stop the run and debug hardware first if velocity, direction, controller
   state, or camera stream becomes abnormal.

After the run:

1. Check ``logs/.../video/train`` or ``logs/.../video/eval``.
2. Check that reward matches keyboard input and real success.
3. Check replay growth.
4. Check intervention logging.
5. If smoke testing fails, do not increase training steps.

Troubleshooting
---------------

- ``RLT_REALWORLD_STAGE2_BASE_PATH`` should point to the SFT ``actor/`` directory, not ``rl_token_model.pt``.
- ``RLT_REALWORLD_STAGE1_RL_TOKEN_PATH`` is the Stage1 ``rl_token_model.pt`` path.
- ``RLINF_NODE_RANK`` and the ROS workspace must be set before ``ray start``.
  Hardware-facing fields such as ``RLINF_KEYBOARD_DEVICE``, GELLO port, and
  camera serials must also be visible before env workers start; they can come
  from shell ``export`` or from YAML / ``env_vars``.
- ``use_gello: True`` requires a valid ``gello_port``.
- For joint RLT, prefer GELLO. Do not treat SpaceMouse as a joint-target intervention device.
- If reward stays at 0, check ``target_pos``, ``reward_threshold``, ``check_orientation_success``, or temporarily keep ``keyboard_reward_wrapper: single_stage``.
- If keyboard input is ignored, check that the keyboard is connected to the slave, ``RLINF_KEYBOARD_DEVICE`` was set before slave ``ray start``, and the event device is readable.
- If ``full_task`` never lets the actor control, check ``env/rlt_in_critical_phase`` and ``env/rlt_record_transition``. If they stay 0, the ``v`` key may not be read by the env worker.
- If workers cannot find Franka/ROS packages, the slave likely did not source the ROS workspace before ``ray start``.
- If robot reset is unsafe, fix the joint qpos before training.

Resume Stage2
-------------

Stage2 checkpoints are stored under:

.. code-block:: text

   logs/<time>-rlt_stage2_realworld_joint/rlt_stage2_realworld_joint/checkpoints/global_step_xxx/actor

To resume, set:

.. code-block:: yaml

   runner:
     resume_dir: /path/to/checkpoints/global_step_xxx

Then launch the same entry point:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint

If you want to keep the original config untouched, copy it first:

.. code-block:: text

   examples/embodiment/config/rlt_stage2_realworld_joint_resume.yaml

Then launch:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_resume
