# Realworld RLT EE-action workflow

This variant keeps the realworld RLT state at 34D and converts only the
demonstration actions from 8D joint targets to 7D EE-delta actions.

The YAMLs are filled for the current real robot machine:

- `examples/sft/config/rlt_realworld_ee_pi05_sft.yaml`
- `examples/embodiment/config/rlt_realworld_ee_pi05_sft_eval.yaml`
- `examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml`
- `examples/sft/config/rlt_stage1_realworld_ee.yaml`
- `examples/embodiment/config/rlt_stage2_realworld_ee.yaml`

## 1. Backfill Dataset

```bash
cd /home/i-tingying/RLinf-dev2

python toolkits/realworld_rlt/backfill_ee_delta_actions.py \
  --src /home/i-tingying/rlt_id3_id4_dataset/collected_data_id3_id4/rank_0/id_4 \
  --dst /home/i-tingying/rlt_id3_id4_dataset/collected_data_id3_id4/rank_0/id_4_ee_action
```

The output dataset keeps `state` as 34D and rewrites `actions` to 7D. The
script also writes
`/home/i-tingying/rlt_id3_id4_dataset/collected_data_id3_id4/rank_0/id_4_ee_action/norm_stats.json`.

## 2. Recompute Norm Stats

Usually the backfill output already has `norm_stats.json`. If you want to
recompute it with the OpenPI dataloader path:

```bash
cd /home/i-tingying/RLinf-dev2

python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_rlt_realworld_ee \
  --repo-id /home/i-tingying/rlt_id3_id4_dataset/collected_data_id3_id4/rank_0/id_4_ee_action
```

This writes stats under the OpenPI assets directory printed by the script.
Fill the resulting `norm_stats.json` path into the YAMLs.

## 3. SFT

The converted dataset path and norm stats path are already filled in
`examples/sft/config/rlt_realworld_ee_pi05_sft.yaml`. Check
`actor.model.model_path` points at the local Pi0.5 base checkpoint, then run:

```bash
cd /home/i-tingying/RLinf-dev2

PYTHONPATH=/home/i-tingying/RLinf-dev2 \
python examples/sft/train_vla_sft.py \
  --config-path /home/i-tingying/RLinf-dev2/examples/sft/config \
  --config-name rlt_realworld_ee_pi05_sft \
  runner.logger.log_path=/home/i-tingying/RLinf-dev2/logs
```

## 4. SFT Eval

The SFT actor checkpoint, norm stats path, camera serials, robot IP,
`target_ee_pose`, and `joint_reset_qpos` are filled in the eval/env YAMLs.
Run:

```bash
cd /home/i-tingying/RLinf-dev2

PYTHONPATH=/home/i-tingying/RLinf-dev2 \
python examples/embodiment/eval_embodied_agent.py \
  --config-path /home/i-tingying/RLinf-dev2/examples/embodiment/config \
  --config-name rlt_realworld_ee_pi05_sft_eval \
  runner.logger.log_path=/home/i-tingying/RLinf-dev2/logs
```

This eval uses `PegInsertionEnv-v1` with 7D EE-delta actions and gripper
enabled.

## 5. Stage1

Stage1 reads the SFT actor from
`/home/i-tingying/RLinf-dev2/logs/rlt_realworld_ee_pi05_sft/checkpoints/global_step_5000/actor`
and writes the RL-token checkpoint under the fixed log path below.

```bash
cd /home/i-tingying/RLinf-dev2

PYTHONPATH=/home/i-tingying/RLinf-dev2 \
python examples/sft/train_rlt_stage1.py \
  --config-path /home/i-tingying/RLinf-dev2/examples/sft/config \
  --config-name rlt_stage1_realworld_ee \
  runner.logger.log_path=/home/i-tingying/RLinf-dev2/logs
```

## 6. Stage2

Stage2 uses the original RLinf Franka `PegInsertionEnv-v1` EE controller path.
GELLO is disabled by default because `/dev/ttyUSB0` is already the Robotiq
gripper. Set `env.train.use_gello=true` and `env.train.gello_port=<real_gello_port>`
only after the actual GELLO serial port is known.

```bash
cd /home/i-tingying/RLinf-dev2

PYTHONPATH=/home/i-tingying/RLinf-dev2 \
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/i-tingying/RLinf-dev2/examples/embodiment/config \
  --config-name rlt_stage2_realworld_ee \
  runner.logger.log_path=/home/i-tingying/RLinf-dev2/logs
```
