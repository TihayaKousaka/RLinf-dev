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

典型是两台机器：

```text
master / GPU head，RLINF_NODE_RANK=0
  - 跑 actor
  - 跑 rollout
  - 提交训练入口

slave / robot control，RLINF_NODE_RANK=1
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

Ray 会在 `ray start` 时捕获环境变量。所有机器人、相机、GELLO、键盘相关变量都要在 `ray start` 前设置。

master / GPU head：

```bash
cd /path/to/RLinf
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<head_ip>
```

slave / robot control：

```bash
cd /path/to/RLinf
source ray_utils/realworld/setup_before_ray.sh
source <your_catkin_ws>/devel/setup.bash

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
6. `target_pos` 是孔位或成功判定目标位置，3D xyz。
7. 第一次 smoke 把 `max_joint_delta` 调小，例如 `0.03`。
8. 真机在线训练必须有人盯机器人、日志、键盘奖励和急停。

检查 Franka controller：

```bash
export FRANKA_ROBOT_IP=<Franka IP>
python -m toolkits.realworld_check.test_franka_controller
```

脚本提示后可以输入 `getpos_euler` 查看当前末端位姿。RLT joint 配置真正用于 reset 的是 7D joint qpos，不是 EE reset pose；如果需要记录当前关节角，请用现场控制/状态工具读取 Franka 7 个关节位置后填入 YAML。

检查相机：

```bash
python -m toolkits.realworld_check.test_franka_camera
```

检查 GELLO：

```bash
ls /dev/serial/by-id/
python rlinf/envs/realworld/common/gello/gello_expert.py --port /dev/serial/by-id/<your-gello-port>
```

GELLO 读不到数据时不要打开 `use_gello`。

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

- `target_pos`：3D xyz，成功判定目标位置。
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

Stage2 分四个阶段看：

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

如果 `target_pos` 还没完全校准，人工键盘奖励比自动 reward 更稳。等自动 success threshold 校准后，再考虑关闭键盘奖励。

## 12. 每次运行前后的 SOP

运行前：

1. Franka Desk 无 error。
2. slave 已 source ROS workspace。
3. `RLINF_NODE_RANK`、`RLT_REALWORLD_ROBOT_IP`、相机 serial、GELLO port、`RLINF_KEYBOARD_DEVICE` 都在 `ray start` 前设置。
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
- `RLINF_NODE_RANK`、ROS workspace、`RLINF_KEYBOARD_DEVICE`、GELLO 串口、相机 serial 都要在 `ray start` 前设置。
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
