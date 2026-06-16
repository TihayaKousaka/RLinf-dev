Franka 真机 Joint-Control RLT
==============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍 RLinf 中 Franka 真机 joint-control RLT 流程。该流程使用 RLinf
自己的真机环境、RLT Stage1/Stage2 模型代码、workers 和配置文件。

完整链路为：

.. code-block:: text

   realworld joint LeRobot data
     -> OpenPI pi0.5 joint SFT
     -> SFT realworld evaluation
     -> Stage1 RL-token training
     -> Stage2 realworld online RLT

环境
----

本示例面向 Franka 机械臂的 absolute joint target 控制。

- **Environment type**: ``realworld``
- **Task config**: ``examples/embodiment/config/env/realworld_rlt_joint_peg_insertion.yaml``
- **Stage2 config**: ``examples/embodiment/config/rlt_stage2_realworld_joint.yaml``
- **Robot**: Franka，双相机，可选 GELLO 干预
- **Default task**: ``insert the peg in the hole``
- **Action**: 8D absolute joint action

动作向量语义如下：

.. code-block:: text

   action[0:7] = Franka 7 个关节目标角，单位 radian
   action[7]   = gripper command

观测适配器会导出 OpenPI 和 RLT 配置使用的真机 joint 数据集布局：

.. code-block:: text

   state = gripper
         + joint_pos(7)
         + joint_vel(7)
         + tcp_force(3 xyz)
         + tcp_pose(7 xyz + quat xyzw)
         + tcp_torque(3)
         + tcp_vel(6 lin3 + ang3)

``main_camera`` 对应第三人称视角图像，``wrist_camera`` 对应腕部视角图像。
这些名称必须和 SFT、Stage1 使用的 LeRobot 数据集以及 ``norm_stats.json`` 保持一致。

方法概览
--------

RLT 在 Stage2 中冻结大的 VLA 主干，只训练小型在线 RL 头部：

1. **OpenPI SFT 基座**：先训练或加载一个能输出真机 joint action chunk 的
   OpenPI pi0.5 checkpoint。
2. **Stage1 RL token**：使用同一份真机 joint 数据集离线训练 RL-token 模块。
3. **Stage2 online RLT**：冻结 SFT 基座和 RL token，在真机上在线训练 Stage2
   Gaussian actor 和 twin-Q critic。

在 RLinf 实现中，rollout workers 负责运行冻结的 OpenPI policy 和冻结的
RL-token encoder。actor workers 负责训练 Stage2 actor/critic，并把 actor 权重同步回
rollout。

机器角色
--------

本文档保留了实验现场常用的 ``master`` 和 ``slave`` 叫法。注意：不同于 DAgger，它们只是机器角色，
不是算法概念或机器人主从控制。

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - 名称
     - 典型 rank
     - 职责
   * - ``master`` / GPU head
     - ``RLINF_NODE_RANK=0``
     - Ray head，运行 actor/rollout/training，也是唯一提交训练入口脚本的节点
   * - ``slave`` / robot control
     - ``RLINF_NODE_RANK=1``
     - Ray worker，连接 Franka、相机、GELLO 和键盘输入，运行 env workers 与 Franka controller

所有节点必须处于同一可达网络。slave 必须能打开 ``http://<robot_ip>/desk``，
并能连接 ``<head_ip>:6379``。

依赖安装
--------

Master / GPU head
~~~~~~~~~~~~~~~~~

master 需要 SFT 和 Stage1 使用的 RLinf OpenPI 训练环境。这里建议直接沿用
:doc:`pi0` 中的 OpenPI 安装方式，因为当前验证通过的真机链路就是基于这套环境。
如果你已经完成 SFT/Stage1，直接复用同一个虚拟环境即可；如果需要重新搭建，可按下面两种方式之一准备。

**选项 1：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

请通过镜像内置的 ``switch_env`` 工具切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   cd /path/to/RLinf
   # 为提高国内依赖安装速度，可以添加 --use-mirror 到 install.sh 命令
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

如果你已经完成 SFT/Stage1，也可以直接激活原来的环境，例如：

.. code:: bash

   cd /path/to/RLinf
   source <your_rlinf_openpi_venv>/bin/activate

确认 master 能找到模型和 RLinf 依赖：

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

Stage2 真机 RLT 当前复用上面这套经过 SFT/Stage1 验证的 RLinf OpenPI 环境；
没有单独的 ``--model rlt_stage2 --env franka`` 安装入口。注意这套 OpenPI
环境只用于 master；slave 仍然需要单独安装 Franka/ROS 真机依赖。

Slave / robot control
~~~~~~~~~~~~~~~~~~~~~

slave 需要 Franka、相机、GELLO、键盘和 ROS 依赖。请参考 :doc:`franka` 的 Franka
依赖安装方式，然后安装 RLinf Franka env：

.. code-block:: bash

   cd /path/to/RLinf

   # 推荐 Ubuntu 20.04 + ROS Noetic 用于 Franka 控制。
   bash requirements/install.sh embodied --env franka --venv franka-venv
   source franka-venv/bin/activate

Franka 安装脚本默认使用 ``LIBFRANKA_VERSION=0.15.0`` 和
``FRANKA_ROS_VERSION=0.10.0``。如果你的机器人固件需要其他兼容版本，请在安装前设置：

.. code-block:: bash

   export LIBFRANKA_VERSION=<compatible-libfranka-version>
   export FRANKA_ROS_VERSION=<compatible-franka-ros-version>
   bash requirements/install.sh embodied --env franka --venv franka-venv

如果 ROS、libfranka、franka_ros 和 serl_franka_controllers 已经手动安装好，
可以跳过 ROS 安装并显式 source 工作空间：

.. code-block:: bash

   export SKIP_ROS=1
   bash requirements/install.sh embodied --env franka --venv franka-venv
   source franka-venv/bin/activate
   source /opt/ros/noetic/setup.bash
   source <your_catkin_ws>/devel/setup.bash

启动 Ray 前，确认 slave 能 import 真机运行依赖：

.. code-block:: bash

   python - <<'PY'
   import evdev
   import pyrealsense2
   import serial
   import rlinf
   print("realworld deps ok")
   PY

启动 Ray
--------

Ray 会在 ``ray start`` 时捕获当前 Python 解释器和环境变量。因此要在启动 Ray 前激活正确
venv、在 slave 上 source ROS，并设置所有相机/GELLO/键盘变量。

``ray_utils/realworld/setup_before_ray.sh`` 是模板脚本。source 之前，请在每台机器上修改网卡
和 venv 激活路径：

.. code-block:: bash

   # 可选：只有需要固定通信网卡时才设置。
   # 网卡名示例：eth0、eno1、enp134s0f0。
   export RLINF_COMM_NET_DEVICES="<nic_name>"
   source <your_venv_path>/bin/activate

如果机器只有一块可用网卡，通常可以不设置 ``RLINF_COMM_NET_DEVICES``。

master：

.. code-block:: bash

   cd /path/to/RLinf
   source <your_rlinf_openpi_venv>/bin/activate
   source ray_utils/realworld/setup_before_ray.sh
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

slave：

.. code-block:: bash

   cd /path/to/RLinf
   source franka-venv/bin/activate
   source ray_utils/realworld/setup_before_ray.sh

   # 如果 franka-venv/bin/activate 没有自动 source ROS/catkin：
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

除 ``RLINF_NODE_RANK`` 以外，上述硬件相关字段都可以直接写进 YAML，或者通过
``cluster.node_groups[].env_configs[].env_vars`` 这类配置下发。保留成 shell
环境变量只是为了现场切换更快。``RLINF_NODE_RANK`` 本身仍然必须在 ``ray start``
之前作为机器环境变量设置。

在 slave 上查找键盘 event 设备并授予读权限：

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   sudo chmod 666 /dev/input/eventX

如果 master 和 slave 不共享同一个 RLinf 代码路径，在 master 启动训练脚本前打开代码同步：

.. code-block:: bash

   export RLINF_CODE_WORKING_DIR=auto

执行 ``ray status``，确认节点数量与 ``cluster.num_nodes`` 一致。

真机前检查
----------

任何真机运行前，都需要现场确认：

1. Franka Desk 无 error，并已进入可编程控制模式。
2. 机械臂工作空间清空，peg、hole、相机和线缆不会被手臂扫到。
3. ``reset_joint_qpos`` 是安全复位关节位姿。
4. ``critical_phase_reset_joint_qpos`` 是 ``critical_phase`` 模式下的安全起始关节位姿。
5. ``full_task_reset_joint_qpos`` 是 ``full_task`` 模式下的安全任务起始关节位姿。
6. ``target_pos`` 是底层 env 自动成功判定参考的 3D xyz 目标位置。在当前
   Stage2 配置里，train/eval 默认都开启了 ``keyboard_reward_wrapper: single_stage``，
   所以最终成功/失败通常仍由键盘奖励主导；但 ``target_pos`` 仍应填写真实标定值，
   便于自动成功判定、排障和纯自动 eval。
7. 第一次 smoke 把 ``max_joint_delta`` 调小，例如 ``0.03``。
8. 真机在线训练必须有人盯机器人、日志、键盘奖励和急停。

在 slave 上启动 Ray、设置好硬件环境变量后，先跑统一自检：

.. code-block:: bash

   python -m toolkits.realworld_check.check_realworld_rlt_stack

这个脚本会一次性检查：

- ``rlt_stage2_realworld_joint.yaml`` 和
  ``realworld_rlt_joint_peg_insertion.yaml`` 里的关键占位符是否还没替换。
- Franka IP、controller ready、7D joint state、TCP pose 是否可读。
- ``main_camera`` 和 ``wrist_camera`` 的 serial/type 是否能逐个取流，并报告实际 FPS。
- GELLO 串口是否存在、是否能读到 7D joint 和 gripper。
- ``RLINF_KEYBOARD_DEVICE`` 是否可读，并且是否支持 ``a/b/c/v``。

如果某台机器只想先做局部检查，可以跳过部分硬件项：

.. code-block:: bash

   python -m toolkits.realworld_check.check_realworld_rlt_stack \
     --skip-franka --skip-cameras

单项原始脚本仍可用于交互式排查：

.. code-block:: bash

   export FRANKA_ROBOT_IP=<Franka IP>
   python -m toolkits.realworld_check.test_franka_controller
   python -m toolkits.realworld_check.test_franka_camera
   python rlinf/envs/realworld/common/gello/gello_expert.py --port /dev/serial/by-id/<your-gello-port>

统一自检里任何硬件项 ``FAIL`` 时都不要开始 Stage2。GELLO 读不到数据时不要打开
``use_gello``。

配置
----

集群放置
~~~~~~~~

``examples/embodiment/config/rlt_stage2_realworld_joint.yaml`` 默认使用双节点集群：

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

模型路径
~~~~~~~~

Stage2 前可以在 master 设置这些环境变量，也可以直接修改 YAML：

.. code-block:: bash

   export RLT_REALWORLD_STAGE2_BASE_PATH=/path/to/sft/checkpoints/global_step_xxx/actor
   export RLT_REALWORLD_STAGE1_RL_TOKEN_PATH=/path/to/rl_token_model.pt
   export RLT_REALWORLD_NORM_STATS_PATH=/path/to/norm_stats.json

Stage2 base path 必须指向 SFT 的 ``actor/`` 目录。RL-token path 必须指向
Stage1 的 ``actor/rl_token/rl_token_model.pt``。

Warmup 与干预
~~~~~~~~~~~~~

Stage2 warmup 和 online gate 在
``examples/embodiment/config/rlt_stage2_realworld_joint.yaml`` 中配置：

.. code-block:: yaml

   algorithm:
     warmup_min_size: 100
     warmup_post_collect_updates: 1000
     intervention:
       enable: True
       mode: human_override

``warmup_min_size`` 是 replay transition/window 数，不是 episode 数。
``warmup_post_collect_updates`` 是 replay 满后所需的 learner update 步数。
``human_override`` 表示用真机人在环动作作为干预来源。

遥操作和键盘奖励需要在 ``env.train`` 和 ``env.eval`` 中设置：

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

如果 eval 要纯 policy 跑，可以把 ``env.eval.use_gello`` 改成 ``False``。
Stage2 早期训练建议保留键盘奖励。

任务与干预
~~~~~~~~~~

关键任务字段位于
``examples/embodiment/config/env/realworld_rlt_joint_peg_insertion.yaml``：

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

第一次真机 smoke 建议把 ``max_joint_delta`` 改小，例如 ``0.03``。在校准
``target_pos`` 和 success threshold 时，建议保留 ``keyboard_reward_wrapper: single_stage``。

这四个占位字段本质上都是真机标定值：

- ``target_pos``：底层 env 自动成功判定参考的空间目标点。当前 Stage2 默认主要由
  键盘奖励判成功，但这个字段仍建议认真标定。
- ``reset_joint_qpos``：最终真正执行的 7D reset 关节角，单位 rad。
- ``critical_phase_reset_joint_qpos``：``task_mode: critical_phase`` 时使用的 reset 关节角。
- ``full_task_reset_joint_qpos``：``task_mode: full_task`` 时使用的 reset 关节角。

相机和状态语义必须与训练数据集一致：

.. code-block:: text

   extra_view_image -> main_camera，第三人称视角
   image            -> wrist_camera，腕部视角

任务配置中对应字段为：

.. code-block:: yaml

   main_image_key: "${oc.env:RLT_REALWORLD_MAIN_IMAGE_KEY,main_camera}"
   wrist_image_key: "${oc.env:RLT_REALWORLD_WRIST_IMAGE_KEY,wrist_camera}"

训练阶段
--------

Stage 0：SFT
~~~~~~~~~~~~

修改 ``examples/sft/config/rlt_realworld_joint_pi05_sft.yaml`` 中的真实数据路径、
OpenPI data config、模型路径和归一化统计。至少检查：

- ``data.train_data_paths[].dataset_path``
- ``actor.openpi_data.repo_id``
- ``actor.openpi_data.norm_stats_path``
- ``actor.model.model_path``
- ``runner.max_steps``
- ``runner.save_interval``

然后启动：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh rlt_realworld_joint_pi05_sft

关键产物为：

.. code-block:: text

   logs/<time>-rlt_realworld_joint_pi05_sft/rlt_realworld_joint_pi05_sft/checkpoints/global_step_xxx/actor

Stage 0.5：SFT 真机评估
~~~~~~~~~~~~~~~~~~~~~~~

在线 RL 前，先在真机上评估 SFT policy：

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh rlt_realworld_joint_pi05_sft_eval

启动前至少检查：

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

如果 SFT policy 方向错误、动作突跳，或相机/action 语义不匹配，不要继续进入 Stage2。
SFT eval 应该是纯 policy eval；除非专门调试 intervention，否则保持 ``use_gello`` 和
``use_spacemouse`` 关闭。

Stage 1：RL-token 训练
~~~~~~~~~~~~~~~~~~~~~~

Stage1 是离线训练，不需要连接真机。先修改
``examples/sft/config/rlt_stage1_realworld_joint.yaml``：

- ``data.train_data_paths[].dataset_path``
- ``actor.openpi_data.repo_id``
- ``actor.openpi_data.norm_stats_path``
- ``actor.model.model_path``
- ``actor.model.rlt_stage1.config_name``
- ``runner.max_steps``
- ``runner.save_interval``

然后启动：

.. code-block:: bash

   bash examples/sft/train_rlt_stage1.sh rlt_stage1_realworld_joint

关键产物为：

.. code-block:: text

   logs/<time>/rlt_stage1_realworld_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt

Stage 2：在线真机 RLT
~~~~~~~~~~~~~~~~~~~~~

master/slave Ray 集群准备好，并设置好 Stage2 路径后，在 master 启动：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint

启动前检查前文提到的 Stage2 YAML 字段，尤其是：

- ``cluster.num_nodes`` 和 ``cluster.node_groups``
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
- ``env.eval`` 下对应字段

如果要区分 smoke、critical-phase 正式跑、full-task 正式跑，建议复制 YAML，
不要在真机现场拼大量 Hydra overrides：

.. code-block:: text

   examples/embodiment/config/rlt_stage2_realworld_joint_smoke.yaml
   examples/embodiment/config/rlt_stage2_realworld_joint_full_task.yaml

复制后的配置仍然用同一个入口启动：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_smoke
   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_full_task

Critical Phase 和 Full Task
---------------------------

默认模式是 ``critical_phase``：

.. code-block:: yaml

   env:
     train:
       task_mode: critical_phase
       critical_phase_key: v
       record_prefix_before_critical_phase: false

``critical_phase`` 模式下：

- reset 后立刻进入 critical phase
- warmup 期间仍然收冻结 reference policy 数据
- online ready 后，Stage2 actor 可以从 episode 开始就控制
- 适合把机器人 reset 到孔口附近，只训练对孔/插入这段

``full_task`` 模式下：

- reset 后先是非关键 prefix
- prefix 由 SFT/base/reference policy 控制，Stage2 不控制
- 操作者看到 peg 到孔口附近或到达自己定义的 critical phase 边界时按 ``v``
- replay 默认从 critical phase 开始记录
- online ready 后，Stage2 actor 只在 critical phase 接管

使用 ``full_task`` 时，需要同时修改 train 和 eval：

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

通常不要把 ``record_prefix_before_critical_phase`` 改成 ``true``，否则 prefix 会稀释
critical phase 的学习信号。

Warmup 与在线操作
-----------------

Stage2 使用 replay buffer 大小和 learner 更新步数来控制 online 接管：

- ``warmup_min_size`` 统计 replay transitions/windows，不是 episode 数。
- ``warmup_post_collect_updates`` 统计 actor/critic learner update 步数。
- ``replay_subsample_stride: 0`` 时，replay 按 chunk boundary 构造。

Stage2 运行时会在日志里输出固定格式的在线切换状态：

.. code-block:: text

   [RLT_STATUS][actor] phase=warmup ready=0 buffer_ready=0 replay=42/100 update=0/1000 pending=0
   [RLT_STATUS][env] phase=warmup ready=0 critical=1.00 record=1.00 student=0.00

同时会写只读状态文件：

.. code-block:: text

   ${runner.logger.log_path}/status/rlt_actor_status_rank0.json
   ${runner.logger.log_path}/status/rlt_env_status_rank0.json

现场可以直接查看：

.. code-block:: bash

   watch -n 1 'cat ../results/status/rlt_actor_status_rank0.json; cat ../results/status/rlt_env_status_rank0.json'

``phase`` 有三个值：

1. ``warmup``：replay 还没达到 ``warmup_min_size``，还在收 warmup 数据。
2. ``warmup_wait_online``：replay 已经够了，后台 learner 正在补
   ``warmup_post_collect_updates``。
3. ``online``：``update_step >= warmup_post_collect_updates``，
   ``ready_for_online=true``，后续尝试可以让 Stage2 actor 上场。

现场操作流程：

1. 在 ``replay_size >= warmup_min_size`` 之前，让冻结的 SFT/Stage1 reference
   policy 运行。GELLO 只作为安全接管。
2. replay 已满但 learner update 还没满足时，继续按 warmup 规则操作，或等待 learner。
3. 两个条件都满足后，后续尝试可在 critical phase 使用 Stage2 actor。此时 GELLO 变成纠偏工具。
4. 真机早期建议使用键盘奖励：``c`` 表示成功，``a`` 表示失败/中止，``b`` 表示
   reward 为 0 且不结束，``v`` 用于 ``full_task`` 模式进入 critical phase。

键盘语义：

.. code-block:: text

   c = 成功，reward=1，结束当前 episode
   a = 失败/危险/放弃，reward=-1，结束当前 episode
   b = reward=0，不结束
   v = full_task 中进入 critical phase

现场规则：

1. 每个 episode 前把 peg/hole 放回初始状态。
2. ``critical_phase`` 模式不用按 ``v``。
3. ``full_task`` 模式等 peg 到孔口附近或 critical phase 边界时按 ``v``。
4. warmup 阶段 reference policy 正常且安全时，不要动 GELLO。
5. reference policy 有点偏但还安全时，也先让它暴露问题。
6. 要撞、卡死、危险时用 GELLO 或急停救一下，然后按 ``a``。
7. online ready 后，GELLO 从保险变成纠偏；只纠必要片段，随后尽快松手。
8. 成功插入按 ``c``，失败或无意义继续按 ``a``。

当前 joint-target wrapper 没有“按住按钮才接管”的显式开关。只要 GELLO 读数和
policy action 差异超过 wrapper 阈值，就可能覆盖 policy action，因此操作者必须盯着
GELLO 当前姿态。

奖励和成功判定
--------------

自动成功判定主要看：

.. code-block:: yaml

   override_cfg:
     target_pos: [...]
     reward_threshold: [0.015, 0.015, 0.03, 0.2, 0.2, 0.2]
     check_orientation_success: false
     success_hold_steps: 1

当前 joint peg insertion 默认主要看 xyz 是否进入阈值，因为
``check_orientation_success`` 是 ``false``。

真机早期建议打开键盘奖励：

.. code-block:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage
     eval:
       keyboard_reward_wrapper: single_stage

如果 ``target_pos`` 还没完全校准，人工键盘奖励通常比自动 reward 更稳。当前
Stage2 默认就是这一路径：train/eval 都保留 ``keyboard_reward_wrapper: single_stage``，
由操作员按 ``a/b/c`` 给出最终奖励。等自动 success threshold 校准后，再考虑关闭
eval 的键盘奖励。

运行 SOP
--------

运行前：

1. Franka Desk 无 error。
2. slave 已 source ROS workspace。
3. ``RLINF_NODE_RANK`` 必须在 ``ray start`` 前设置。``RLT_REALWORLD_ROBOT_IP``、
   相机 serial、``RLT_REALWORLD_GELLO_PORT``、``RLINF_KEYBOARD_DEVICE`` 等硬件项
   也必须在 env worker 进程启动前可见；它们可以在 ``ray start`` 前通过 shell
   ``export``，也可以直接写进 YAML / ``env_vars``。
4. ``ray status`` 看到两个节点。
5. ``RLT_REALWORLD_STAGE2_BASE_PATH`` 指向 SFT ``actor/`` 目录。
6. ``RLT_REALWORLD_STAGE1_RL_TOKEN_PATH`` 指向 Stage1 ``rl_token_model.pt``。
7. ``RLT_REALWORLD_NORM_STATS_PATH`` 指向对应数据集的 ``norm_stats.json``。
8. GELLO 能读数，键盘 event 可读。
9. 第一次 smoke 使用较小的 ``max_joint_delta``。
10. 如果跑 ``full_task``，确认 ``v`` 没有和奖励键 ``a/b/c`` 冲突。

运行中：

1. 一个 episode 一个 episode 地看，不要离开。
2. 看机器人实际动作，不要只看日志。
3. 成功及时按 ``c``，明显失败及时按 ``a``。
4. 只在必要片段接管，尤其是 critical phase。
5. 出现异常速度、异常方向、控制器报错、相机掉帧时，停止当前 run，先排硬件。

运行后：

1. 查看 ``logs/.../video/train`` 或 ``logs/.../video/eval``。
2. 检查 reward 是否符合现场按键和实际成功。
3. 检查 replay 是否增长。
4. 检查 intervention 是否被记录。
5. smoke 有问题时不要扩大训练步数。

可选 replay debug 快照
~~~~~~~~~~~~~~~~~~~~~~

正常主线不建议默认 dump 原始 replay；这属于现场排障工具。怀疑采集错了时再打开：

.. code-block:: yaml

   algorithm:
     replay_buffer:
       auto_save: true
       auto_save_path: ${runner.logger.log_path}/debug/rlt_replay
       trajectory_format: pt

打开后 actor 会按 RLinf ``TrajectoryReplayBuffer`` 标准格式增量保存：

.. code-block:: text

   ${runner.logger.log_path}/debug/rlt_replay/rank_0/metadata.json
   ${runner.logger.log_path}/debug/rlt_replay/rank_0/trajectory_index.json
   ${runner.logger.log_path}/debug/rlt_replay/rank_0/trajectory_*.pt

检查 replay 内容：

.. code-block:: bash

   python -m toolkits.rlt.inspect_rlt_replay ../results/debug/rlt_replay/rank_0

重点看 ``total_samples`` / ``inspected_samples`` 是否增长、reward 是否长期全 0、``source_chunk`` 里是否真的有
``HUMAN/MIXED``、``collection_phase`` 是否符合 warmup/online 预期，以及
``intervention_flag_rate`` 是否和现场 GELLO 接管大致一致。

常见坑
------

- ``RLT_REALWORLD_STAGE2_BASE_PATH`` 应该指向 SFT 的 ``actor/`` 目录，不是 ``rl_token_model.pt``。
- ``RLT_REALWORLD_STAGE1_RL_TOKEN_PATH`` 才是 Stage1 的 ``rl_token_model.pt``。
- ``RLINF_NODE_RANK``、ROS workspace 必须在 ``ray start`` 前设置。``RLINF_KEYBOARD_DEVICE``、
  GELLO 串口、相机 serial 等硬件项也必须在 env worker 启动前可见；它们既可以在
  ``ray start`` 前 ``export``，也可以写进 YAML / ``env_vars``。
- ``use_gello: True`` 时必须设置有效 ``gello_port``。
- joint RLT 优先用 GELLO，不要直接把 SpaceMouse 当 joint-target 接管设备。
- 如果 reward 一直是 0，先查 ``target_pos``、``reward_threshold``、``check_orientation_success``，或临时保留 ``keyboard_reward_wrapper: single_stage``。
- 如果按键没反应，先查键盘是否接在 slave、``RLINF_KEYBOARD_DEVICE`` 是否在 slave ``ray start`` 前设置、event 设备是否有读权限。
- 如果 ``full_task`` 一直没有 actor 接管，先看日志/指标里的 ``env/rlt_in_critical_phase`` 和 ``env/rlt_record_transition``。它们一直是 0，通常说明没按到 ``v`` 或键盘 event 没被 env worker 读到。
- 如果 worker 找不到 Franka/ROS 包，多半是 slave 在 ``ray start`` 前没有 source ROS workspace。
- 如果机器人 reset 都不安全，先修 joint qpos，不要继续训练。

Stage2 Resume
-------------

Stage2 checkpoint 位于：

.. code-block:: text

   logs/<time>-rlt_stage2_realworld_joint/rlt_stage2_realworld_joint/checkpoints/global_step_xxx/actor

继续训练时设置：

.. code-block:: yaml

   runner:
     resume_dir: /path/to/checkpoints/global_step_xxx

然后仍然使用同一入口：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint

如果想保留原配置，建议复制一个 resume 配置：

.. code-block:: text

   examples/embodiment/config/rlt_stage2_realworld_joint_resume.yaml

然后启动：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_resume
