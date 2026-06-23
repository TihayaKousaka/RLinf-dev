# 真机 RLT 运行说明

现在实际要跑的流程：

1. `SFT`
2. `SFT eval`
3. `Stage1`
4. `Stage2`

其中：

- `SFT` 和 `Stage1` 只在推理/GPU 节点运行。
- `SFT eval` 和 `Stage2` 需要推理/GPU 节点 + Franka 控制节点一起运行。



## 一、开始前先确认

需要的数据在i-kaizhi/rollout/RLinf-dev/data里面，yaml还没同步路径，需要：

1. 数据集路径已经填对。
2. `norm_stats.json` 路径已经填对。
3. 真机环境 YAML 里的机器人 IP、相机序列号、reset 位姿都已经填对。
4. `pi05_base` 或对应的 SFT/Stage1 checkpoint 路径已经填对。

这次流程对应的配置文件是：

- `examples/sft/config/rlt_realworld_ee_pi05_sft.yaml`
- `examples/embodiment/config/rlt_realworld_ee_pi05_sft_eval.yaml`
- `examples/sft/config/rlt_stage1_realworld_ee.yaml`
- `examples/embodiment/config/rlt_stage2_realworld_ee.yaml`
- `examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml`

## 二、Franka 真机两机准备

这一节用于 `SFT eval` 和 `Stage2`。

### 1. 先改好 `setup_before_ray.sh`

文件：

- `ray_utils/realworld/setup_before_ray.sh`

至少要改这两项：

```bash
source <your_venv_path>/bin/activate
# source <your_catkin_ws>/devel/setup.bash
```

说明：

- 两个节点都要能激活同一个 RLinf 运行环境。
- 控制节点如果没有把 `franka_ros` / `serl_franka_controllers` 写进虚拟环境启动链里，就要额外 `source <your_catkin_ws>/devel/setup.bash`。
- `ray start` 会记住当前 Python 环境和环境变量，所以每次启动 Ray 之前都要先 `source` 这份脚本。

### 2. 确认 Franka 硬件配置

`Stage2` 的硬件配置在：

- `examples/embodiment/config/rlt_stage2_realworld_ee.yaml`
- `examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml`

至少要确认下面这些字段已经填对：

```yaml
cluster:
  num_nodes: 2
  node_groups:
    - label: "4090"
      node_ranks: 0
    - label: realworld
      node_ranks: 1
      hardware:
        type: Franka
        configs:
          - robot_ip: 172.16.0.2
            gripper_type: robotiq
            gripper_connection: "/dev/ttyUSB0"
            camera_infos:
              - name: main_camera
                serial_number: "141722070657"
                camera_type: "realsense"
              - name: wrist_camera
                serial_number: "usb-XVisio_Technology_XVisio_vSLAM_250801DR48FB26001216-video-index0"
                camera_type: "lumos"
            node_rank: 1
```

任务位姿相关字段在：

```yaml
override_cfg:
  target_ee_pose: [...]
  joint_reset_qpos: [...]
```

如果这些没填对，后面真机不会正常工作。

### 3. SpaceMouse 和键盘

当前真机 `Stage2` 配置是：

```yaml
env:
  train:
    use_spacemouse: True
```

含义：

- `SpaceMouse` 用来做人类接管纠偏。
- 进入 `critical phase` 用的是键盘 `v`。

当前环境配置在：

```yaml
task_mode: full_task
critical_phase_key: v
record_prefix_before_critical_phase: false
```

所以：

- 一开始默认跑 `BASE`
- 按 `v` 进入 `critical phase`，这个是键盘按的，按一下就行
- 进入后才会开始真正执行 `RL`
- `SpaceMouse` 一动，就会接管并把人类动作写进 replay，什么时候动都行，最好是插销快进孔的时候，或者偏得太离谱的时候，按v前后都可以，算法细节写好了已经。尽量早一点进入关键阶段，就是别等到失败已成定局的时候，早点没关系。

### 4. 控制节点键盘设置

`v` 键依赖 Linux 键盘输入设备，所以控制节点要在 `ray start` 之前设置好键盘设备。

先在控制节点查看键盘：

```bash
ls -l /dev/input/by-id/*-event-kbd
```

例如如果看到：

```bash
usb-Logitech_USB_Keyboard-event-kbd -> ../event20
```

那么就执行：

```bash
chmod 666 /dev/input/event20
export RLINF_KEYBOARD_DEVICE=/dev/input/event20
```

建议把这条 `export RLINF_KEYBOARD_DEVICE=...` 也写进 `ray_utils/realworld/setup_before_ray.sh`，这样控制节点以后每次启动 Ray 都会继承这个设置。

## 三、SFT

用途：
先训练一个能直接做真机 EE 动作输出的 OpenPI policy。

运行命令：

```bash
cd path/to/RLinf-dev

bash examples/sft/run_vla_sft.sh rlt_realworld_ee_pi05_sft
```

训练完成后，重点看这个目录：

```bash
logs/rlt_realworld_ee_pi05_sft/checkpoints/global_step_5000/actor
```

这就是后面 `SFT eval` 和 `Stage1` 要用的 SFT checkpoint。

## 四、SFT Eval

这一步开始需要两台节点一起跑。

### 两机启动顺序

#### 推理/GPU 节点

```bash
cd /Users/lixiaoqun/Downloads/pixiv/rlt-openpi-sim/RLinf-dev
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=192.168.120.43
```

#### Franka 控制节点

```bash
cd /Users/lixiaoqun/Downloads/pixiv/rlt-openpi-sim/RLinf-dev
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=1
export RLINF_KEYBOARD_DEVICE=/dev/input/event20
ray start --address='192.168.120.43:6379'
```

如果 IP 变化，把 `192.168.120.43` 改成新的 head 节点 IP。

用途：
在真机上验证 SFT policy 能不能单独完成任务。

运行命令：

```bash
cd RLinf-dev

bash evaluations/run_eval.sh rlt_realworld_ee_pi05_sft_eval
```

操作时注意：

- 这一步默认不是人在环评估。
- 先确认机器人 reset 位姿正常。
- 先确认主视角和腕部相机画面正常。

如果这一步都跑不通，不要继续跑 Stage1/Stage2。

## 五、Stage1

用途：
训练 RL token。

它读取上一步的 SFT checkpoint，输出后面 Stage2 要用的 `rl_token_model.pt`。

运行命令：

```bash
cd /Users/lixiaoqun/Downloads/pixiv/rlt-openpi-sim/RLinf-dev

bash examples/sft/train_rlt_stage1.sh rlt_stage1_realworld_ee
```

训练完成后，重点看这个文件：

```bash
logs/rlt_stage1_realworld_ee/checkpoints/global_step_5000/actor/rl_token/rl_token_model.pt
```

这就是 Stage2 要加载的 RL token checkpoint。

如果你想离线检查 Stage1 重建效果，可以再跑：

```bash
cd /Users/lixiaoqun/Downloads/pixiv/rlt-openpi-sim/RLinf-dev

PYTHONPATH=$(pwd) \
python toolkits/realworld_rlt/evaluate_stage1_reconstruction.py \
  --dataset-path /mnt/public2/xiekaizhi/rlt-openpi-sim/data/realworld_ee_lerobot \
  --vla-checkpoint /mnt/public2/xiekaizhi/rlt-openpi-sim/tmp/RLinf-dev/logs/rlt-realworld-ee-sft-step-2500/global_step_5000/actor \
  --rl-token-checkpoint /mnt/public2/xiekaizhi/rlt-openpi-sim/tmp/RLinf-dev/logs/20260622-20:22:35/rlt_stage1_realworld_ee/checkpoints/global_step_15000/actor/rl_token/rl_token_model.pt \
  --norm-stats-path /mnt/public2/xiekaizhi/rlt-openpi-sim/data/realworld_ee_lerobot/norm_stats.json
```

## 六、Stage2

这一步也需要两台节点一起跑。

### 两机启动顺序

#### 推理/GPU 节点

```bash
cd RLinf-dev
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=192.168.120.43
```

#### Franka 控制节点

```bash
cd RLinf-dev
source ray_utils/realworld/setup_before_ray.sh
export RLINF_NODE_RANK=1
export RLINF_KEYBOARD_DEVICE=/dev/input/event20
ray start --address='192.168.120.43:6379'
```

用途：
跑真机 RLT 在线训练。

正常运行命令：

```bash
cd RLinf-dev

bash examples/embodiment/run_realworld.sh rlt_stage2_realworld_ee
```

说明：

- 这条脚本会自动新建一个带时间戳的日志目录。
- 训练日志和视频都会写到那个新目录下面。

真机操作方法：

1. 系统开始后，前缀阶段默认跑 `BASE` 动作。
2. 到需要进入 RLT 阶段时，按 `v` 进入 `critical phase`。
3. 进入 `critical phase` 之后，系统才会开始用 `RL` 动作。
4. 如果快失败了，直接动 `SpaceMouse` 接管纠偏。
5. `SpaceMouse` 不动之后，系统会继续回到自动执行。

现在这版真机 Stage2 的核心就是：

- `v` 控制是否进入 `critical phase`
- `SpaceMouse` 控制是否做人类接管
- replay 会记录真实执行过的动作，包括人类纠偏动作

### 如果要把 replay 落盘调试

正常训练默认不会把 replay buffer 快照落盘。

如果你要调试 replay 内容，在 head 节点直接运行下面这条命令：

```bash
cd RLinf-dev

LOG_DIR="$(pwd)/logs/$(date +'%Y%m%d-%H:%M:%S')-rlt_stage2_realworld_ee_replay"

PYTHONPATH=$(pwd) \
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name rlt_stage2_realworld_ee \
  runner.logger.log_path="${LOG_DIR}" \
  algorithm.replay_buffer.auto_save=true \
  algorithm.replay_buffer.auto_save_path="${LOG_DIR}/replay_dump" \
  algorithm.replay_buffer.trajectory_format=pt
```

跑起来以后，replay 会落到：

```bash
${LOG_DIR}/replay_dump/rank_0
```

这个命令只是在现有 `Stage2` 基础上打开 replay 落盘，不改其他训练逻辑，适合排查 replay 里到底写进去了什么。

## 七、Stage2 运行时重点看什么

操作人员重点只看下面几件事：

1. 机器人 reset 是否正常。
2. 相机画面是否正常。
3. 按 `v` 之后是否真的进入了 `critical phase`。（应该会打log）
4. 动 `SpaceMouse` 时，机器人是否真的被人工接管。
5. 日志里 replay size 是否在增长。
6. 日志里 `ready_for_online` 是否最终变成 1。

如果出现下面情况，先停下来排查：

1. `SFT eval` 本身就执行不稳。
2. 按 `v` 没反应。
3. `SpaceMouse` 动了但机器人没接管。
4. replay size 一直不增长。
5. Stage1 没产出 `rl_token_model.pt`。

## 八、这四步的依赖关系

顺序不要跳：

1. 先跑 `SFT`
2. 再跑 `SFT eval`
3. 再跑 `Stage1`
4. 最后跑 `Stage2`

对应关系是：

- `SFT` 产出 `SFT checkpoint`
- `SFT eval` 验证这个 `SFT checkpoint`
- `Stage1` 读取 `SFT checkpoint`，产出 `rl_token_model.pt`
- `Stage2` 读取 `Stage1 checkpoint` 和 `rl_token_model.pt`
