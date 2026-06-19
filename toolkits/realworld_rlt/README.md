# Realworld RLT EE-action workflow

This variant keeps the realworld RLT state at 34D and converts only the
demonstration actions from 8D joint targets to 7D EE-delta actions.

Edit paths directly in these YAML files before running:

- `examples/sft/config/rlt_realworld_ee_pi05_sft.yaml`
- `examples/embodiment/config/rlt_realworld_ee_pi05_sft_eval.yaml`
- `examples/embodiment/config/env/realworld_rlt_ee_peg_insertion.yaml`

## 1. Backfill Dataset

```bash
python toolkits/realworld_rlt/backfill_ee_delta_actions.py \
  --src /path/to/realworld_joint_lerobot \
  --dst /path/to/realworld_ee_lerobot
```

The output dataset keeps `state` as 34D and rewrites `actions` to 7D. The
script also writes `/path/to/realworld_ee_lerobot/norm_stats.json`.

## 2. Recompute Norm Stats

Usually the backfill output already has `norm_stats.json`. If you want to
recompute it with the OpenPI dataloader path:

```bash
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_rlt_realworld_ee \
  --repo-id /path/to/realworld_ee_lerobot
```

This writes stats under the OpenPI assets directory printed by the script.
Fill the resulting `norm_stats.json` path into the YAMLs.

## 3. SFT

Fill the converted dataset path and norm stats path in
`examples/sft/config/rlt_realworld_ee_pi05_sft.yaml`, then run:

```bash
bash examples/sft/run_vla_sft.sh rlt_realworld_ee_pi05_sft
```

## 4. SFT Eval

Fill the SFT actor checkpoint, norm stats path, camera serials, robot IP,
`target_ee_pose`, and `joint_reset_qpos` in the eval/env YAMLs, then run:

```bash
bash examples/embodiment/run_realworld_eval.sh rlt_realworld_ee_pi05_sft_eval
```

This eval uses `PegInsertionEnv-v1` with 7D EE-delta actions and gripper
enabled. It does not use `FrankaJointPegInsertionEnv-v1`.
