# Realworld RLT Joint Runbook

这份文档只说明 Franka 真机 joint-control RLT 流程，不涉及 ManiSkill/仿真路径。

目标链路：

```text
realworld joint LeRobot data
  -> OpenPI pi0.5 joint SFT
  -> SFT realworld eval
  -> Stage1 RL-token training
  -> Stage2 realworld online RLT
```

启动原则：超参和路径写进 YAML，命令行只负责启动。不要在真机现场用一大串 Hydra override 临时拼配置。

## 1. 机器分工

典型是两台机器。这里保留 `master` / `slave` 的现场叫法，但它们只是机器角色，不是算法概念，也不是机器人主从控制：

- **master / GPU head 节点**：Ray head，通常 `RLINF_NODE_RANK=0`，负责训练和提交入口脚本。
- **slave / robot control 节点**：Ray worker，通常 `RLINF_NODE_RANK=1`，负责连接真机硬件并运行 env/controller。

```text
master / GPU head 节点，RLINF_NODE_RANK=0
  - 跑 actor
  - 跑 rollout
  - 提交训练入口

slave / robot control 节点，RLINF_NODE_RANK=1
  - 和 Franka、相机、GELLO、键盘相连
  - 跑 env worker
  - 跑 Franka controller
```

硬件连接检查：

1. Franka 控制柜和 slave 在同一局域网，slave 能打开 `http://<robot_ip>/desk`。
2. master 和 slave 网络互通，slave 能连接 `<head_ip>:6379`。
3. `main_camera` 和 `wrist_camera` 都接在 slave 上。
4. GELLO 接 slave，建议使用 `/dev/serial/by-id/...` 这种稳定串口路径。
5. 键盘奖励读取的是 slave 上的 Linux input event，不是 master 终端 stdin。
6. 急停、Desk 页面和 slave 终端都要在操作者能立刻处理的位置。

当前 RLT joint 环境使用 8D absolute joint action：

```text
action[0:7] = Franka 7 个关节目标角，单位 radian
action[7]   = gripper command
```

所以人在环优先用 GELLO 的 `joint_target` 模式。SpaceMouse 是末端位姿 delta 语义，不建议直接用于这条 joint RLT 链路。

## 2. Master/Slave 环境配置

这一节包括两部分：先在 master 和 slave 分别装好依赖，再启动 Ray。Ray 会在 `ray start` 时捕获 Python 解释器和环境变量，所以依赖、ROS workspace、相机/GELLO/键盘变量都必须在 `ray start` 前准备好。

Stage2 真机环境全部来自 RLinf：`rlinf/envs/realworld`、`rlinf/models/embodiment/rlt_stage2`、`rlinf/workers/*` 和 `examples/embodiment/config/rlt_stage2_realworld_joint.yaml`。文档里提到 OpenPI 时，只表示 RLinf 内置支持的 OpenPI 模型依赖，不表示依赖其他在线训练框架。

### 2.1 依赖安装

master / GPU head 节点负责 SFT、Stage1、Stage2 actor/rollout 训练，需要 RLinf + OpenPI 模型训练环境：

```bash
cd /path/to/RLinf

# 如果已有可跑 SFT/Stage1 的 RLinf OpenPI 环境，直接激活同一个环境即可。
source <your_rlinf_openpi_venv>/bin/activate
```

如果你此前按 RLinf `pi0` 文档里的 Docker 方式跑过 SFT/Stage1，master 一般就是复用同一套
OpenPI 环境，例如 `rlinf/rlinf:agentic-rlinf0.2-maniskill_libero` 镜像里
`source switch_env openpi` 后的 Python 环境。

注意：当前安装脚本没有单独的 `--model rlt_stage2 --env franka` 入口。Stage2 真机这条链路建议复用你已经验证过 SFT/Stage1 的 RLinf OpenPI Python 环境，确保 master 上能正常 import `openpi`、`torch`、`transformers`、`rlinf`，并能读取 SFT actor、Stage1 `rl_token_model.pt` 和 `norm_stats.json`。

在 master 上确认 RLinf 和 OpenPI 模型依赖能被找到：

```bash
python - <<'PY'
import rlinf
print("rlinf ok:", rlinf.__file__)
try:
    import openpi
    print("openpi ok:", openpi.__file__)
except Exception as exc:
    print("openpi import failed:", exc)
PY
```

slave / robot control 节点负责 Franka、相机、GELLO、键盘输入，需要 Franka/ROS/真机依赖：

```bash
cd /path/to/RLinf

# 推荐 Ubuntu 20.04 + ROS Noetic。该命令会安装 franka extra、
# ROS/libfranka/franka_ros/serl_franka_controllers 等真机控制依赖。
bash requirements/install.sh embodied --env franka --venv franka-venv
source franka-venv/bin/activate
```

Franka 固件版本建议先按机器人 Desk 页面确认。`requirements/install.sh` 默认使用 `LIBFRANKA_VERSION=0.15.0` 和 `FRANKA_ROS_VERSION=0.10.0`；如果你的固件需要其他版本，请在安装前显式设置：

```bash
export LIBFRANKA_VERSION=<compatible-libfranka-version>
export FRANKA_ROS_VERSION=<compatible-franka-ros-version>
bash requirements/install.sh embodied --env franka --venv franka-venv
```

如果 slave 已经手动安装好 ROS Noetic、libfranka、franka_ros 和 serl_franka_controllers，可以跳过 ROS 相关自动安装：

```bash
export SKIP_ROS=1
bash requirements/install.sh embodied --env franka --venv franka-venv
source franka-venv/bin/activate
source /opt/ros/noetic/setup.bash
source <your_catkin_ws>/devel/setup.bash
```

无论采用哪种安装方式，启动 Ray 前都要确认 slave 当前 shell 能找到真机依赖：

```bash
python - <<'PY'
import evdev
import pyrealsense2
import serial
import rlinf
print("realworld deps ok")
PY
```

### 2.2 Ray 启动前环境变量

`ray_utils/realworld/setup_before_ray.sh` 是模板脚本。第一次使用前，分别在 master 和 slave 上打开它，至少改两处：

```bash
# 可选：只有需要固定通信网卡时才设置，例如 eth0、eno1、enp134s0f0 等
export RLINF_COMM_NET_DEVICES="<nic_name>"

# 按本机实际 venv 修改。master 指向 RLinf OpenPI 训练 venv，
# slave 指向 Franka/ROS venv。
source <your_venv_path>/bin/activate
```

如果你不想改脚本，也可以把脚本里的 `source <your_venv_path>/bin/activate` 注释掉，然后在 source 该脚本前手动激活 venv。不要保留这个占位符原样。

master / GPU head 节点：

```bash
cd /path/to/RLinf
source <your_rlinf_openpi_venv>/bin/activate
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<head_ip>
```

slave / robot control 节点：

```bash
cd /path/to/RLinf
source franka-venv/bin/activate
source ray_utils/realworld/setup_before_ray.sh

# 如果没有通过 franka-venv/bin/activate 自动 source ROS/catkin，
# 或者你使用的是手动安装的 ROS workspace，则需要显式 source：
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
```

除了 `RLINF_NODE_RANK` 之外，上面这些硬件项都可以直接写进 YAML，或者通过
`cluster.node_groups[].env_configs[].env_vars` 这类配置下发。保留成 shell 环境变量，
主要是为了现场切换更方便。`RLINF_NODE_RANK` 本身仍然必须在 `ray start` 前作为机器环境变量设置。

不要在 `ray start` 之后才换 venv、source ROS workspace 或修改相机/GELLO/键盘变量；Ray worker 不一定能看到这些变化。改完环境后需要 `ray stop` 再重新 `ray start`。

找键盘 event：

```bash
ls -l /dev/input/by-id/*-event-kbd
sudo chmod 666 /dev/input/eventX
```

`/dev/input/eventX` 要换成真实键盘设备。设置完后再启动 Ray。

如果 master 和 slave 不共享同一个 RLinf 代码路径，在 master 提交训练前设置：

```bash
export RLINF_CODE_WORKING_DIR=auto
```

确认集群：

```bash
ray status
```

`examples/embodiment/config/rlt_stage2_realworld_joint.yaml` 默认使用 `cluster.num_nodes: 2`，其中 GPU 是 rank 0，realworld 控制节点是 rank 1。

## 3. 真机前检查

运行前检查：

1. Franka Desk 无 error，并已进入可编程控制模式。
2. 机械臂工作空间清空，peg、hole、相机和线缆不会被手臂扫到。
3. `reset_joint_qpos` 是安全复位关节位姿。
4. `critical_phase_reset_joint_qpos` 是 critical-phase 模式下的安全起始关节位姿。
5. `full_task_reset_joint_qpos` 是 full-task 模式下的安全任务起始关节位姿。
6. `target_pos` 是底层 env 自动成功判定参考的 3D xyz 目标位置。在当前 Stage2 配置里，
   train/eval 默认都开启了 `keyboard_reward_wrapper: single_stage`，所以最终成功/失败通常仍由
   键盘奖励主导；但 `target_pos` 仍应填写真实标定值，便于自动成功判定、排障和纯自动 eval。
7. 第一次 smoke 把 `max_joint_delta` 调小，例如 `0.03`。
8. 真机在线训练必须有人盯机器人、日志、键盘奖励和急停。

在 slave 上启动 Ray、设置好硬件环境变量后，先跑统一自检：

```bash
python -m toolkits.realworld_check.check_realworld_rlt_stack
```

这个脚本会一次性检查：

- `rlt_stage2_realworld_joint.yaml` 和 `realworld_rlt_joint_peg_insertion.yaml` 里的关键占位符是否还没替换。
- Franka IP、controller ready、7D joint state、TCP pose 是否可读。
- `main_camera` 和 `wrist_camera` 的 serial/type 是否能逐个取流，并报告实际 FPS。
- GELLO 串口是否存在、是否能读到 7D joint 和 gripper。
- `RLINF_KEYBOARD_DEVICE` 是否可读，并且是否支持 `a/b/c/v`。

如果某台机器只想先做局部检查，可以跳过部分硬件项：

```bash
python -m toolkits.realworld_check.check_realworld_rlt_stack \
  --skip-franka --skip-cameras
```

单项原始脚本仍可用于交互式排查：

```bash
export FRANKA_ROBOT_IP=<Franka IP>
python -m toolkits.realworld_check.test_franka_controller
python -m toolkits.realworld_check.test_franka_camera
python rlinf/envs/realworld/common/gello/gello_expert.py --port /dev/serial/by-id/<your-gello-port>
```

统一自检里任何硬件项 `FAIL` 时都不要开始 Stage2。GELLO 读不到数据时不要打开 `use_gello`。

## 4. YAML 需要改什么

### Stage2 主配置

文件：

```text
examples/embodiment/config/rlt_stage2_realworld_joint.yaml
```

必须按现场修改或通过环境变量提供：

```yaml
cluster:
  node_groups:
    - label: "4090"
      node_ranks: 0
    - label: realworld
      node_ranks: 1
      hardware:
        type: Franka
        configs:
          - robot_ip: "${oc.env:RLT_REALWORLD_ROBOT_IP,ROBOT_IP}"
            camera_infos:
              - name: main_camera
                serial_number: "${oc.env:RLT_REALWORLD_MAIN_CAMERA_SERIAL,MAIN_CAMERA_SERIAL}"
                camera_type: "${oc.env:RLT_REALWORLD_MAIN_CAMERA_TYPE,realsense}"
              - name: wrist_camera
                serial_number: "${oc.env:RLT_REALWORLD_WRIST_CAMERA_SERIAL,WRIST_CAMERA_SERIAL}"
                camera_type: "${oc.env:RLT_REALWORLD_WRIST_CAMERA_TYPE,lumos}"
            node_rank: 1
```

重要路径：

```yaml
actor:
  model:
    model_path: "${oc.env:RLT_REALWORLD_STAGE2_BASE_PATH,/path/to/sft/checkpoints/global_step_xxx/actor}"
    rlt_stage2:
      norm_stats_path: "${oc.env:RLT_REALWORLD_NORM_STATS_PATH,/path/to/norm_stats.json}"
      rl_token_path: "${oc.env:RLT_REALWORLD_STAGE1_RL_TOKEN_PATH,/path/to/rl_token_model.pt}"
```

含义：

- `RLT_REALWORLD_STAGE2_BASE_PATH` 指向 SFT 的 `actor/` 目录，不是 `full_weights.pt`，也不是 `rl_token_model.pt`。
- `RLT_REALWORLD_STAGE1_RL_TOKEN_PATH` 指向 Stage1 产出的 `actor/rl_token/rl_token_model.pt`。
- `RLT_REALWORLD_NORM_STATS_PATH` 必须和训练 SFT/Stage1 的真实数据集一致。

Stage2 warmup 和在线切换：

```yaml
algorithm:
  warmup_min_size: 100
  warmup_post_collect_updates: 1000
  intervention:
    enable: True
    mode: human_override
```

含义：

- `warmup_min_size`：replay 至少收多少 transition 后才允许在线训练进入 ready 条件。
- `warmup_post_collect_updates`：replay 满后，后台至少更新多少步才允许 online actor 上场。
- `intervention.mode: human_override`：用真机人在环动作作为干预来源。

遥操和键盘：

```yaml
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
```

如果 eval 要纯 policy 跑，可以把 `env.eval.use_gello` 改成 `False`。Stage2 训练现场建议先保留键盘奖励。

### 任务 env 配置

文件：

```text
examples/embodiment/config/env/realworld_rlt_joint_peg_insertion.yaml
```

必须替换占位符：

```yaml
override_cfg:
  target_pos: TARGET_POS
  reset_joint_qpos: RESET_JOINT_QPOS
  critical_phase_reset_joint_qpos: CRITICAL_PHASE_RESET_JOINT_QPOS
  full_task_reset_joint_qpos: FULL_TASK_RESET_JOINT_QPOS
  max_joint_delta: 0.08
```

字段含义：

- `target_pos`：底层 env 自动成功判定参考的空间目标点。当前 Stage2 默认主要由键盘奖励判成功，
  但这个字段仍建议认真标定。
- `reset_joint_qpos`：兜底 reset 关节角，7D Franka qpos，单位 radian。
- `critical_phase_reset_joint_qpos`：`task_mode: critical_phase` 时实际使用的 reset 关节角。
- `full_task_reset_joint_qpos`：`task_mode: full_task` 时实际使用的 reset 关节角。
- `max_joint_delta`：每步允许的最大关节变化，第一次真机 smoke 建议用 `0.03`，确认稳定后再考虑默认 `0.08`。

相机语义：

```yaml
main_image_key: "${oc.env:RLT_REALWORLD_MAIN_IMAGE_KEY,main_camera}"
wrist_image_key: "${oc.env:RLT_REALWORLD_WRIST_IMAGE_KEY,wrist_camera}"
```

数据语义必须和数据集一致：

```text
extra_view_image -> main_camera，第三人称视角，实验室当前是 D435
image            -> wrist_camera，腕部视角，实验室当前是鱼眼相机
```

状态语义：

```text
state = gripper
      + joint_pos(7)
      + joint_vel(7)
      + tcp_force(3 xyz)
      + tcp_pose(7 xyz+quat xyzw)
      + tcp_torque(3)
      + tcp_vel(6 lin3+ang3)
```

## 5. SFT 训练

配置：

```text
examples/sft/config/rlt_realworld_joint_pi05_sft.yaml
```

先在 YAML 里改：

- `data.train_data_paths[].dataset_path`
- `actor.openpi_data.repo_id`
- `actor.openpi_data.norm_stats_path`
- `actor.model.model_path`
- `runner.max_steps`
- `runner.save_interval`

启动：

```bash
bash examples/sft/run_vla_sft.sh rlt_realworld_joint_pi05_sft
```

关键产物：

```text
logs/<time>-rlt_realworld_joint_pi05_sft/rlt_realworld_joint_pi05_sft/checkpoints/global_step_xxx/actor
```

后续 SFT eval、Stage1、Stage2 都使用这个 `actor/` 目录。

## 6. SFT 真机 Eval

配置：

```text
examples/embodiment/config/rlt_realworld_joint_pi05_sft_eval.yaml
```

先在 YAML 里改：

- `cluster.node_groups[].hardware.configs[].robot_ip`
- `cluster.node_groups[].hardware.configs[].camera_infos`
- `actor.model.model_path`
- `actor.model.openpi.config_name`
- `actor.model.openpi_data.norm_stats_path`
- `env.eval.override_cfg.target_pos`
- `env.eval.override_cfg.reset_joint_qpos`
- `env.eval.override_cfg.critical_phase_reset_joint_qpos`
- `env.eval.override_cfg.full_task_reset_joint_qpos`
- `env.eval.override_cfg.max_joint_delta`
- `env.eval.max_episode_steps`
- `env.eval.max_steps_per_rollout_epoch`

启动：

```bash
bash examples/embodiment/run_realworld_eval.sh rlt_realworld_joint_pi05_sft_eval
```

现场判断：

1. 机器人会先 reset 到当前模式选择的 joint qpos。
2. 看第一段动作方向。反向、突跳、接近撞击就立刻停。
3. SFT 能把 peg 大致带到孔附近，再上 Stage2。
4. SFT eval 明显不对时，不要上 Stage2；优先查 checkpoint、norm stats、相机语义和 action 维度。

SFT eval 默认 `use_gello: False`、`use_spacemouse: False`，这一步是纯 policy eval。

## 7. Stage1 RL-token

配置：

```text
examples/sft/config/rlt_stage1_realworld_joint.yaml
```

先在 YAML 里改：

- `data.train_data_paths[].dataset_path`
- `actor.openpi_data.repo_id`
- `actor.openpi_data.norm_stats_path`
- `actor.model.model_path`
- `actor.model.rlt_stage1.config_name`
- `runner.max_steps`
- `runner.save_interval`

启动：

```bash
bash examples/sft/train_rlt_stage1.sh rlt_stage1_realworld_joint
```

Stage1 是离线训练，不需要连真机。

关键产物：

```text
logs/<time>/rlt_stage1_realworld_joint/checkpoints/global_step_xxx/actor/rl_token/rl_token_model.pt
```

把这个路径写到 Stage2 YAML 的 `actor.model.rlt_stage2.rl_token_path`，或在 master 提交训练前设置：

```bash
export RLT_REALWORLD_STAGE1_RL_TOKEN_PATH=/path/to/rl_token_model.pt
```

## 8. Stage2 在线真机训练

配置：

```text
examples/embodiment/config/rlt_stage2_realworld_joint.yaml
```

先按第 4 节改好 YAML，尤其是：

- `cluster.num_nodes`
- `cluster.node_groups`
- `actor.model.model_path`
- `actor.model.rlt_stage2.norm_stats_path`
- `actor.model.rlt_stage2.rl_token_path`
- `algorithm.warmup_min_size`
- `algorithm.warmup_post_collect_updates`
- `algorithm.intervention`
- `env.train.task_mode`
- `env.train.keyboard_reward_wrapper`
- `env.train.use_gello`
- `env.train.gello_action_mode`
- `env.train.override_cfg.max_joint_delta`
- `env.eval` 下对应字段

启动：

```bash
bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint
```

如果要区分 smoke、critical-phase 正式跑、full-task 正式跑，建议自己复制 YAML 成不同配置文件，例如：

```text
examples/embodiment/config/rlt_stage2_realworld_joint_smoke.yaml
examples/embodiment/config/rlt_stage2_realworld_joint_full_task.yaml
```

然后仍然用同一种启动方式：

```bash
bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_smoke
bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_full_task
```

## 9. Critical Phase 和 Full Task

默认配置是：

```yaml
env:
  train:
    task_mode: critical_phase
    critical_phase_key: v
    record_prefix_before_critical_phase: false
```

`critical_phase`：

- reset 后立刻进入 critical phase。
- warmup 期间仍然收老策略/参考策略数据。
- online ready 后，Stage2 actor 可以从 episode 开始就控制。
- 适合把机器人 reset 到孔口附近，只训练对孔/插入这段。

`full_task`：

- reset 后先是非关键 prefix。
- prefix 由 SFT/base/reference 走，Stage2 actor 不控制。
- 操作者看到 peg 到孔口附近或到达自己定义的 critical phase 边界时，按 `v`。
- 按 `v` 后进入 critical phase，replay 默认从这里开始记录。
- online ready 后，Stage2 actor 只在 critical phase 接管。

full-task 需要在 YAML 里改：

```yaml
env:
  train:
    task_mode: full_task
    critical_phase_key: v
    record_prefix_before_critical_phase: false
  eval:
    task_mode: full_task
    critical_phase_key: v
    record_prefix_before_critical_phase: false
```

通常不建议把 `record_prefix_before_critical_phase` 改成 `true`，否则 prefix 会稀释 critical phase 的学习信号，按照论文的思路，RL只在critical phase学，RLT 主要优化 critical phase，也就是最难、最精细、最影响成功率的阶段。

## 10. Stage2 现场操作

Stage2 运行时会在日志里输出固定格式的在线切换状态：

```text
[RLT_STATUS][actor] phase=warmup ready=0 buffer_ready=0 replay=42/100 update=0/1000 pending=0
[RLT_STATUS][env] phase=warmup ready=0 critical=1.00 record=1.00 student=0.00
```

同时会写只读状态文件：

```text
${runner.logger.log_path}/status/rlt_actor_status_rank0.json
${runner.logger.log_path}/status/rlt_env_status_rank0.json
```

现场可以直接查看：

```bash
watch -n 1 'cat ../results/status/rlt_actor_status_rank0.json; cat ../results/status/rlt_env_status_rank0.json'
```

`phase` 有三个值：

1. `warmup`：replay 还没达到 `warmup_min_size`，还在收 warmup 数据。
2. `warmup_wait_online`：replay 已经够了，后台 learner 正在补 `warmup_post_collect_updates`。
3. `online`：`update_step >= warmup_post_collect_updates`，`ready_for_online=true`，后续尝试可以让 Stage2 actor 上场。

Stage2 现场操作按四个阶段看：

1. `replay_size < warmup_min_size`：收 warmup 数据。看老策略表现，GELLO 只救命，不做纠偏教学。
2. `replay_size >= warmup_min_size` 但 `update_step < warmup_post_collect_updates`：材料够了但还在训练。继续按 warmup 规则操作，或让机器人停着等后台更新。
3. 两个条件都满足：从下一次尝试开始 online。actor 做得对就别碰，actor 偏了再用 GELLO 拉回来。
4. online 之后循环：actor 尝试，操作者必要纠偏，键盘判成功/失败，数据进 replay，后台继续训练。

键盘：

```text
c = 成功，reward=1，结束当前 episode
a = 失败/危险/放弃，reward=-1，结束当前 episode
b = reward=0，不结束
v = full_task 中进入 critical phase
```

现场规则：

1. 每个 episode 前把 peg/hole 放回初始状态。
2. `critical_phase` 模式不用按 `v`。
3. `full_task` 模式等 peg 到孔口附近或 critical phase 边界时按 `v`。
4. 老策略正常跑时不要动 GELLO。
5. 老策略有点偏但安全时也先别动，让系统看到老策略问题。
6. 要撞、卡死、危险时用 GELLO 或急停救一下，然后按 `a`。
7. online 后，GELLO 从保险变成纠偏；拉回来后尽快松手，让 actor 继续。
8. 成功插入按 `c`，失败或无意义继续按 `a`。

当前 joint-target wrapper 没有“按住按钮才接管”的显式开关。只要 GELLO 读数和 policy action 差异超过阈值，就可能覆盖 policy action，所以操作者必须盯着 GELLO 当前姿态。

## 11. 奖励和成功判定

自动成功判定主要看：

```yaml
override_cfg:
  target_pos: [...]
  reward_threshold: [0.015, 0.015, 0.03, 0.2, 0.2, 0.2]
  check_orientation_success: false
  success_hold_steps: 1
```

当前 joint peg insertion 默认主要看 xyz 是否进入阈值，因为 `check_orientation_success: false`。

真机早期建议打开：

```yaml
env:
  train:
    keyboard_reward_wrapper: single_stage
  eval:
    keyboard_reward_wrapper: single_stage
```

如果 `target_pos` 还没完全校准，人工键盘奖励比自动 reward 更稳。当前 Stage2 默认就是这一路径：
train/eval 都保留 `keyboard_reward_wrapper: single_stage`，由操作员按 `a/b/c` 给出最终奖励。
等自动 success threshold 校准后，再考虑关闭键盘奖励。

## 12. 每次运行前后的 SOP

运行前：

1. Franka Desk 无 error。
2. slave 已 source ROS workspace。
3. `RLINF_NODE_RANK` 必须在 `ray start` 前设置。`RLT_REALWORLD_ROBOT_IP`、相机 serial、
   GELLO port、`RLINF_KEYBOARD_DEVICE` 等硬件项也必须在 env worker 启动前可见；
   它们可以在 `ray start` 前 `export`，也可以写进 YAML / `env_vars`。
4. `ray status` 看到两个节点。
5. `RLT_REALWORLD_STAGE2_BASE_PATH` 指向 SFT `actor/` 目录。
6. `RLT_REALWORLD_STAGE1_RL_TOKEN_PATH` 指向 Stage1 `rl_token_model.pt`。
7. `RLT_REALWORLD_NORM_STATS_PATH` 指向对应数据集的 `norm_stats.json`。
8. GELLO 能读数，键盘 event 可读。
9. `max_joint_delta` 第一次 smoke 用小值。
10. 如果跑 full-task，确认 `v` 没有和奖励键 `a/b/c` 冲突。

运行中：

1. 一个 episode 一个 episode 地看，不要离开。
2. 看机器人实际动作，不要只看日志。
3. 成功及时按 `c`，明显失败及时按 `a`。
4. 需要接管时只接管必要片段，尤其是 critical phase。
5. 出现异常速度、异常方向、控制器报错、相机掉帧，停止当前 run，先排硬件。

运行后：

1. 看 `logs/.../video/train` 或 `logs/.../video/eval`。
2. 看 reward 是否符合现场按键和实际成功。
3. 看 replay 是否增长。
4. 看 intervention 是否被记录。
5. smoke 有问题时不要扩大训练步数。

## 13. 常见坑

- `RLT_REALWORLD_STAGE2_BASE_PATH` 应该指向 SFT 的 `actor/` 目录，不是 `rl_token_model.pt`。
- `RLT_REALWORLD_STAGE1_RL_TOKEN_PATH` 才是 Stage1 的 `rl_token_model.pt`。
- `RLINF_NODE_RANK`、ROS workspace 必须在 `ray start` 前设置。`RLINF_KEYBOARD_DEVICE`、
  GELLO 串口、相机 serial 等硬件项也必须在 env worker 启动前可见；它们既可以在
  `ray start` 前 `export`，也可以写进 YAML / `env_vars`。
- `use_gello: True` 时必须设置有效 `gello_port`。
- joint RLT 优先用 GELLO，不要直接把 SpaceMouse 当 joint-target 接管设备。
- 如果 reward 一直是 0，先查 `target_pos`、`reward_threshold`、`check_orientation_success`，或临时保留 `keyboard_reward_wrapper: single_stage`。
- 如果按键没反应，先查键盘是否接在 slave、`RLINF_KEYBOARD_DEVICE` 是否在 slave `ray start` 前设置、event 设备是否有读权限。
- 如果 full-task 一直没有 actor 接管，先看日志/指标里的 `env/rlt_in_critical_phase` 和 `env/rlt_record_transition`。它们一直是 0，通常说明没按到 `v` 或键盘 event 没被 env worker 读到。
- 如果 worker 找不到 Franka/ROS 包，多半是 slave 在 `ray start` 前没有 source ROS workspace。
- 如果机器人 reset 都不安全，先修 joint qpos，不要继续训练。

## 14. Stage2 Resume

Stage2 checkpoint：

```text
logs/<time>-rlt_stage2_realworld_joint/rlt_stage2_realworld_joint/checkpoints/global_step_xxx/actor
```

继续训练时，在 YAML 里设置：

```yaml
runner:
  resume_dir: /path/to/checkpoints/global_step_xxx
```

然后仍然用同一个入口：

```bash
bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint
```

如果想保留原配置，建议复制一个 resume 配置，例如：

```text
examples/embodiment/config/rlt_stage2_realworld_joint_resume.yaml
```

然后启动：

```bash
bash examples/embodiment/run_embodiment.sh rlt_stage2_realworld_joint_resume
```
