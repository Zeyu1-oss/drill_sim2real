# DP3 — InspireDrill task (what changed + how to integrate)

This branch (`dp3` of `drill_sim2real`) = a **code-only snapshot** of
[YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) plus my changes to
train DP3 on the **InspireDrill** task (dexterous drill grasping; data collected in IsaacLab by the
separate `inspire_drill` repo). It **excludes** `third_party/`, datasets, checkpoints, and the bundled
CUDA installer. This repo is only the **policy-training** side.

The changes are tiny and surgical (7 files, ~36 lines) plus a few new files for the task.

---

## 1. What I changed (and why)

### Modified files
| file | change | why |
|---|---|---|
| `diffusion_policy_3d/policy/simple_dp3.py` | comment out `import pytorch3d.ops`; in `predict_action()` add `obs_dict = dict_apply(obs_dict, lambda x: x.to(self.device))`; drop dead debug prints | pytorch3d isn't installed (it crashed the import); move inference inputs to the model device so deploy can pass CPU tensors |
| `diffusion_policy_3d/policy/dp3.py` | same two changes (pytorch3d import off + move `obs_dict` to device) | same |
| `diffusion_policy_3d/config/simple_dp3.yaml` | `n_action_steps: 8 → 4`; dataloader `num_workers 8→16`, `persistent_workers False→True`; `checkpoint_every 200→1`; `save_ckpt False→True`; `val_every 1→5` | execute 4-step chunks; GPU was starved by the dataloader; save every epoch so you can deploy-test early; val reads point clouds so run it less often |
| `train.py` | `if env_runner is None:` → `if 'test_mean_score' not in step_log:` | we train **offline** with a `DummyRunner` (no in-sim eval). This makes the top-k checkpoint metric `test_mean_score` fall back to `-train_loss`, so checkpointing-by-metric still works |
| `diffusion_policy_3d/dataset/__init__.py` | `+ from .inspire_drill_dataset import InspireDrillDataset` | register the new dataset so hydra can resolve it |
| `diffusion_policy_3d/env/__init__.py` | replace `adroit / dexart / metaworld` imports with `from .inspire_drill.inspire_drill_wrapper import InspireDrillEnv` | so importing the `env` package doesn't require those sim deps |
| `visualizer/visualizer/pointcloud.py` | enable `aspectmode='cube'`, fixed `[-1,1]` axes, white bg in saved html | nicer point-cloud snapshots |

> ⚠️ The `env/__init__.py` change **removes** the adroit/dexart/metaworld imports. If you still want
> those tasks, **merge** instead of overwriting (keep their imports and just add the `InspireDrillEnv` line).

### New files (additions)
| file | what it is |
|---|---|
| `diffusion_policy_3d/config/task/inspire_drill.yaml` | the task config: `shape_meta` (point_cloud `[P,3]`, agent_pos `[13]`, action `[13]`), `dataset._target_ = InspireDrillDataset`, `env_runner = DummyRunner` |
| `diffusion_policy_3d/config/task/drill.yaml` | a task variant |
| `diffusion_policy_3d/dataset/inspire_drill_dataset.py` | **disk-backed** zarr loader → `(point_cloud, agent_pos, action)`, plus `get_normalizer()` (datasets can exceed RAM) |
| `diffusion_policy_3d/env/inspire_drill/{__init__,inspire_drill_wrapper,viz}.py` | minimal env wrapper + viz (offline training never steps the sim) |
| `diffusion_policy_3d/env_runner/dummy_runner.py` | no-op runner (no in-sim eval during training) |
| `diffusion_policy_3d/env_runner/inspire_drill_runner.py`, `env_runner/__init__.py` | runner glue |
| `scripts/train_policy_inspire_drill.sh` | training launcher |
| `scripts/test_ckpt_inference.py` | checkpoint inference smoke test |

---

## 2. How to put this into a cloned official DP3

Assume you already did:
```bash
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git
cd 3D-Diffusion-Policy           # repo root: has inner 3D-Diffusion-Policy/, third_party/, visualizer/, scripts/
# ...install per its INSTALL.md (conda env `dp3`, etc.)
```

### Method 1 — pull just my files in with git (recommended)
```bash
git remote add drill git@github.com:Zeyu1-oss/drill_sim2real.git
git fetch drill dp3
# extract exactly my custom files into your working tree (paths line up with the official layout):
git checkout drill/dp3 -- \
  3D-Diffusion-Policy/train.py \
  3D-Diffusion-Policy/diffusion_policy_3d/policy/simple_dp3.py \
  3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py \
  3D-Diffusion-Policy/diffusion_policy_3d/config/simple_dp3.yaml \
  3D-Diffusion-Policy/diffusion_policy_3d/config/task/inspire_drill.yaml \
  3D-Diffusion-Policy/diffusion_policy_3d/config/task/drill.yaml \
  3D-Diffusion-Policy/diffusion_policy_3d/dataset/__init__.py \
  3D-Diffusion-Policy/diffusion_policy_3d/dataset/inspire_drill_dataset.py \
  3D-Diffusion-Policy/diffusion_policy_3d/env/__init__.py \
  3D-Diffusion-Policy/diffusion_policy_3d/env/inspire_drill \
  3D-Diffusion-Policy/diffusion_policy_3d/env_runner \
  scripts/train_policy_inspire_drill.sh \
  scripts/test_ckpt_inference.py \
  visualizer/visualizer/pointcloud.py
```
`git checkout <ref> -- <paths>` copies files from my branch into your tree **without** merging the
(unrelated) histories. Review the env/`__init__.py` overwrite note above before committing.

### Method 2 — clone my branch and copy files over
```bash
git clone -b dp3 git@github.com:Zeyu1-oss/drill_sim2real.git /tmp/dp3_drill
# from your official repo root, copy the same paths listed above, e.g.:
cp /tmp/dp3_drill/3D-Diffusion-Policy/train.py 3D-Diffusion-Policy/train.py
cp -r /tmp/dp3_drill/3D-Diffusion-Policy/diffusion_policy_3d/env/inspire_drill \
      3D-Diffusion-Policy/diffusion_policy_3d/env/
# ...(repeat for each path in the list above)
```

---

## 3. Data format

A zarr produced by the `inspire_drill` repo's `collect_dp3_data.py`:
```
data/
  point_cloud  (T, P, 3)    # P must equal shape_meta.point_cloud (e.g. 3810)
  state        (T, 13)      # agent_pos = controlled joint positions
  action       (T, 13)
meta/
  episode_ends (n_episodes,)
```
`InspireDrillDataset` reads it disk-backed. Variable-length episodes are fine (a `horizon` window
slides over each). Keep `P` equal to what you deploy with for a clean train↔deploy match.

## 4. Train

```bash
# args:  task           data.zarr                          config_name  seed  gpu
bash scripts/train_policy_inspire_drill.sh inspire_drill /path/to/your.zarr simple_dp3 0 0
```
which runs (hydra entry `train.py`, `config_path = diffusion_policy_3d/config`):
```bash
python train.py --config-name=simple_dp3.yaml task=inspire_drill \
    hydra.run.dir=data/outputs/<exp>_seed0 \
    training.seed=0 training.device=cuda:0 exp_name=<exp> \
    checkpoint.save_ckpt=True task.dataset.zarr_path=/path/to/your.zarr
```
- Checkpoints → `3D-Diffusion-Policy/data/outputs/<exp>_seed0/checkpoints/latest.ckpt`
  (`model` + `ema_model` + `normalizer`).
- No in-sim eval during training (`DummyRunner`). Evaluate / deploy with the `inspire_drill` repo's
  `scripts/deploy_dp3_sim.py` (loads EMA weights + normalizer from the ckpt).

## 5. Key config knobs

| key | value | note |
|---|---|---|
| `horizon` | 16 | diffusion prediction length |
| `n_obs_steps` | 2 | obs frames as condition |
| `n_action_steps` | 4 | actions executed per inference |
| `shape_meta.point_cloud` | `[P, 3]` | **P must match your zarr** (e.g. 3810) |
| `shape_meta.agent_pos` / `action` | `[13]` | controlled joints |
| `training.use_ema` | True | ckpt gets `ema_model`; deploy uses EMA |
