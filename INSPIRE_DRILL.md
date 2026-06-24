# DP3 — InspireDrill task (customizations over official 3D-Diffusion-Policy)

This branch (`dp3` of `drill_sim2real`) is a **code-only snapshot** of
[YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) plus my changes to
train DP3 on the **InspireDrill** task (dexterous drill grasping; data is collected in IsaacLab by the
separate `inspire_drill` repo). It **excludes** `third_party/`, datasets, checkpoints, and the bundled
CUDA installer. This repo is only the **policy-training** side; data collection / deployment live in
the `inspire_drill` repo.

---

## What is custom vs official (the delta)

**Modified files** (overwrite the official ones):
- `3D-Diffusion-Policy/diffusion_policy_3d/policy/simple_dp3.py`
- `3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py`
- `3D-Diffusion-Policy/diffusion_policy_3d/config/simple_dp3.yaml`
- `3D-Diffusion-Policy/diffusion_policy_3d/dataset/__init__.py`   ← registers `InspireDrillDataset`
- `3D-Diffusion-Policy/diffusion_policy_3d/env/__init__.py`
- `3D-Diffusion-Policy/train.py`
- `visualizer/visualizer/pointcloud.py`

**New files** (add them):
- `3D-Diffusion-Policy/diffusion_policy_3d/config/task/inspire_drill.yaml`, `task/drill.yaml`
- `3D-Diffusion-Policy/diffusion_policy_3d/dataset/inspire_drill_dataset.py`
- `3D-Diffusion-Policy/diffusion_policy_3d/env/inspire_drill/{__init__,inspire_drill_wrapper,viz}.py`
- `3D-Diffusion-Policy/diffusion_policy_3d/env_runner/{dummy_runner,inspire_drill_runner,__init__}.py`
- `scripts/train_policy_inspire_drill.sh`, `scripts/test_ckpt_inference.py`
- `3D-Diffusion-Policy/dp3_real_robot_demo/DP3_CUSTOM_TASK_GUIDE.md`

---

## Option A — use this branch directly

```bash
git clone -b dp3 git@github.com:Zeyu1-oss/drill_sim2real.git dp3_drill && cd dp3_drill
```
`third_party/` is **not** in this branch — install the official DP3 deps per its `INSTALL.md`
(conda env `dp3`, etc.). `pytorch3d_simplified` is only referenced for the fused FPS kernel and is
optional (the pipeline falls back to pure-torch FPS if absent).

## Option B — drop into a fresh official clone

```bash
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git
# install per its INSTALL.md, then copy the custom files from this branch over the official tree:
#   - overwrite the 7 MODIFIED files
#   - add the NEW files (task configs, inspire_drill_dataset.py, env/inspire_drill/,
#     env_runner/{dummy_runner,inspire_drill_runner}.py, scripts/train_policy_inspire_drill.sh)
```
The two `__init__.py` edits (`dataset/`, `env/`) are what register `InspireDrillDataset` and the
runner, so they **must** be applied or hydra can't resolve `task=inspire_drill`.

---

## Data format

A zarr produced by the `inspire_drill` repo's `collect_dp3_data.py`:
```
data/
  point_cloud  (T, P, 3)    # P must equal shape_meta.point_cloud (e.g. 3810)
  state        (T, 13)      # agent_pos = controlled joint positions
  action       (T, 13)
meta/
  episode_ends (n_episodes,)
```
`InspireDrillDataset` reads it **disk-backed** (datasets can exceed RAM). Episodes may be variable
length; the sequence sampler slides a window of `horizon` over each (episodes shorter than ~`horizon`
are dropped). Point clouds are count-agnostic in the PointNet encoder (max-pool), but for a clean
train↔deploy match keep `P` equal to what you deploy with.

---

## Train

```bash
# args:  task           data.zarr                          config_name  seed  gpu
bash scripts/train_policy_inspire_drill.sh inspire_drill /path/to/your.zarr simple_dp3 0 0
```
which runs (entry: `train.py`, hydra `config_path = diffusion_policy_3d/config`):
```bash
python train.py --config-name=simple_dp3.yaml task=inspire_drill \
    hydra.run.dir=data/outputs/<exp>_seed0 \
    training.seed=0 training.device=cuda:0 exp_name=<exp> \
    checkpoint.save_ckpt=True \
    task.dataset.zarr_path=/path/to/your.zarr
```
- Checkpoints → `3D-Diffusion-Policy/data/outputs/<exp>_seed0/checkpoints/latest.ckpt`
  (contains `model`, `ema_model`, and the `normalizer`).
- **No in-sim eval during training** (`DummyRunner`). Evaluate / deploy with the `inspire_drill`
  repo's `scripts/deploy_dp3_sim.py` (it loads the EMA weights + normalizer from the ckpt).

---

## Key config knobs (`config/simple_dp3.yaml` + `config/task/inspire_drill.yaml`)

| key | value | note |
|---|---|---|
| `horizon` | 16 | diffusion prediction length |
| `n_obs_steps` | 2 | obs frames as condition |
| `n_action_steps` | 4 | actions executed per inference |
| `shape_meta.point_cloud` | `[P, 3]` | **P must match your zarr** (e.g. 3810) |
| `shape_meta.agent_pos` / `action` | `[13]` | controlled joints |
| `training.use_ema` | True | ckpt gets `ema_model`; deploy uses EMA |
