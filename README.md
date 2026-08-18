# inspire_drill

A simulation research platform for **dexterous power-drill manipulation**: a Franka arm with an Inspire five-finger hand (13 controlled joints — 7 arm + 6 hand) must pick a drill up off the table and then align it to a target pose in front of a plate.

## What this repo is for

Reinforcement learning solves this task easily *if* you let the policy cheat — feed it the drill's exact pose, contact forces, and velocities from the simulator. None of that exists on a real robot. So the workflow here is **teacher → student distillation**:

```
  PPO teacher  ──rollout──>  zarr dataset  ──train──>  DP3 student  ──eval──>  success rate
 (privileged state)         (point cloud +            (point cloud +          (fixed Sobol
                             proprioception)           proprioception)         pose set)
```

1. Train a privileged RL teacher (rl_games PPO) on full simulator state.
2. Roll the teacher out and record only what a real robot could observe: a depth-camera point cloud plus joint proprioception, paired with the joint targets the teacher executed.
3. Train a [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) (DP3) student on that dataset (in the sibling repo).
4. Deploy the student back into the same environment and score it against fixed initial poses.

The reason the code is structured this way is **ablation**: the whole point is asking *what does the student actually need to see?* The observation is assembled from swappable segments — fixed camera vs. wrist camera, whether to include a robot-FK point cloud, a ground-truth handle mask channel, joint positions alone vs. positions + applied joint torques, and a contact-gated torque branch supervised by privileged per-link contact forces. Data collection and deployment share the same `perception/` code so that a change in the observation is the *only* difference between two runs, and every checkpoint is graded on the same policy-independent Sobol pose set (`--init_pose_file`) so success rates are directly comparable.

Three drill variants (`drill2`, `drill_blue`, `drill_yellow`) with different geometry and scale are trained and evaluated together, and success is reported per variant.

---

## The two tasks

| | **Task 1 — grasp** | **Task 2 — align** |
|---|---|---|
| env | `tasks/grasp_drill_env.py` (`GraspDrillEnv`) | `tasks/stage2_env.py` (`Stage2Env`) |
| start state | drill on the table at a randomized pose | drill **already grasped** (reset from a pkl of successful grasp end-states) |
| goal | lift the drill off the table and hold it | bring the drill to a target orientation in front of the plate |
| success | drill held for ≥20 of the last 50 steps | alignment held for `--success_hold_stop` consecutive steps |

`tasks/chained_env.py` (`ChainedEnv`) runs both in one episode: it starts in phase 0 (grasp) and switches to phase 1 (align) the moment the grasp criterion fires. `--stage1_only` uses the same env but terminates at that switch point — the grasp-only setting used for most of the DP3 work here.

---

## Installation

`scripts/` runs inside the **Isaac Lab Python environment**; the DP3 student trains in a **separate** env (its dependencies conflict with Isaac Sim's).

### 1. Isaac Sim + Isaac Lab

Full instructions: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip

# Isaac Sim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# PyTorch (CUDA 12.8, x86_64)
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git --branch main
cd IsaacLab
./isaaclab.sh --install
```

Verify with `python -c "import isaaclab; print(isaaclab.__version__)"`. This repo is developed against **Python 3.11 / isaaclab 0.53.1**.

### 2. Extra packages in the Isaac Lab env

```bash
pip install rl_games==1.6.1 zarr numcodecs dill omegaconf trimesh
```

`rl_games` is the PPO backend · `zarr`/`numcodecs` write the DP3 dataset · `dill`/`omegaconf` load DP3 checkpoints at deploy time · `trimesh` is only needed by `tools/build_robot_pointcloud.py`.

### 3. DP3 student environment (separate)

Clone [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) next to this repo and follow its install. Developed against:

```
Python 3.8 · torch 2.4.1 (cu124) · diffusers 0.36 · zarr 2.16 · hydra-core 1.3.2
```

### 4. Assets

USD files for the drills / robot / plate go in `assets/` (gitignored). The canonical robot point cloud is a one-off build:

```bash
python tools/build_robot_pointcloud.py     # -> assets/inspire_tac/robot_canonical_points.npz
```

`data/`, `collected_data/`, `runs/` and `output/` are gitignored as well.

---

## Scripts

### Training the RL teachers (privileged state, PPO via rl_games)

| script | trains | notes |
|---|---|---|
| `scripts/train_with_rl_games.py` | **Task 1 (grasp)** | `--num_envs 256 --headless`, resume with `--checkpoint`, config `config/agents/rl_games_ppo_cfg.yaml` |
| `scripts/collect_success_data.py` | — | banks successful grasp end-states into a pkl; Task 2 resets from it |
| `scripts/train2.py` | **Task 2 (align)** | `--dataset collected_data/success_data.pkl`, `--num_envs 4096`, target pose via `--target_pos/--target_quat` |

```bash
# Task 1
python scripts/train_with_rl_games.py --headless --num_envs 256

# grasp end-states for Task 2
python scripts/collect_success_data.py --headless --num_envs 2048 \
    --checkpoint runs/<stage1_run>/nn/<stage1_run>.pth \
    --output collected_data/success_data.pkl

# Task 2
python scripts/train2.py --headless --num_envs 4096 \
    --dataset collected_data/success_data.pkl
```

Checkpoints land in `runs/<run_name>/nn/*.pth`.

### Deploying the privileged RL policy (state observations, no cameras)

| script | runs |
|---|---|
| `scripts/play_drill.py` | Task 1 teacher alone, on screen |
| `scripts/play_stage2.py` | Task 2 teacher alone, resets from `--success_dataset` |
| `scripts/play_chained.py` | both teachers chained in one episode (`--stage1_checkpoint` + `--stage2_checkpoint`) |
| `scripts/deploy_dp3_sim.py --policy rl` | the teacher inside the **DP3 evaluation loop** — same env, same init poses, same success counting, so the number is directly comparable to a student's |

```bash
python scripts/play_drill.py   --num_envs 4 --checkpoint runs/<stage1_run>/nn/<x>.pth --real_time
python scripts/play_stage2.py  --num_envs 8 --checkpoint runs/<stage2_run>/nn/<x>.pth
python scripts/play_chained.py --num_envs 3 \
    --stage1_checkpoint runs/<stage1_run>/nn/<x>.pth \
    --stage2_checkpoint runs/<stage2_run>/nn/<x>.pth

# teacher measured in the student's eval harness
python scripts/deploy_dp3_sim.py --policy rl --stage1_only --headless --num_envs 70 \
    --stage1_checkpoint runs/<stage1_run>/nn/<x>.pth \
    --init_pose_file data/eval_sobol_100_pos010.npz
```

### DP3 student: collection, training, deployment

| script | does |
|---|---|
| `scripts/collect_dp3_data.py` | rolls out the RL teacher and writes a DP3 zarr (point cloud + joint state + executed joint targets) |
| `tools/make_sobol_init_poses.py` | generates a policy-independent Sobol eval pose set (no simulator needed) |
| `scripts/deploy_dp3_sim.py` | runs / evaluates a trained DP3 checkpoint in the same env |

**Collect** (grasp task, fixed camera only, no robot-FK segment, with joint torques and contact labels):

```bash
python scripts/collect_dp3_data.py --stage1_only --headless \
    --num_envs 64 --episodes_per_variant 1000 \
    --disable_cam2 --no_robot --force_state --save_contact \
    --stage1_checkpoint runs/<stage1_run>/nn/<x>.pth \
    --output data/norobot.zarr
```

Only successful episodes are written. One row = the observation at decision time plus the joint target `q*` executed that step:

```
data/point_cloud  (T, N, 3)   env-local fused cloud
data/state        (T, 13|26)  joint pos [| applied torque]
data/action       (T, 13)     joint targets q*
data/contact      (T, 13)     per-link contact force (privileged training label)
meta/episode_ends (E,)
```

**Train the student** in the sibling repo, with a task yaml whose `shape_meta` matches the zarr:

```bash
cd ../3D-Diffusion-Policy/3D-Diffusion-Policy
python train.py --config-name=simple_dp3.yaml \
    task=inspire_drill_grasp_norobot_contact \
    task.dataset.zarr_path=[/home/zeyu/inspire_drill/data/norobot.zarr]
```

**Deploy / evaluate**:

```bash
python scripts/deploy_dp3_sim.py --stage1_only --headless --num_envs 70 \
    --dp3_ckpt <run>/checkpoints/epoch_0150.ckpt \
    --disable_cam2 --no_robot \
    --init_pose_file data/eval_sobol_100_pos010.npz
```

`--init_pose_file` replays a fixed pose set, each pose exactly once, so different checkpoints are graded on identical initial conditions; success is printed overall and per drill variant.

Other modes of the same script:

- `--stage2_only` — align task only, drill starts already grasped from `--success_dataset`
- `--stage2_dp3_ckpt` — two-stage student: the grasp checkpoint hands off to the align checkpoint
- `--stage2_rl` — grasp by DP3, align by the privileged teacher (upper bound on what the hand-off supports)
- `--dual` — student and teacher on the same trajectory, logging the per-step action difference
- `--collect_success_data` — like `collect_success_data.py`, but with DP3 driving the grasp
- `--dump_obs_zarr` — record what the deployed policy actually sees, in the collection layout, for a 1:1 diff against the training data

---

## Important: collect and deploy must agree

The observation is built by the same `perception/` code on both sides, but its **composition is chosen by flags**. A mismatch is silent — the policy simply sees something it was never trained on. Only the point-cloud *size* is checked against the checkpoint at startup.

| Flag | Effect |
|------|--------|
| `--disable_cam2` | camera segment = fixed cam1 alone (2048 pts) instead of cam1 + wrist cam |
| `--no_robot` | drop the robot FK segment |
| `--robot_pc_points` / `--robot_pc_per_link` / `--robot_pc_hand_only` | robot segment size and which links it covers |
| `--ground_points` | synthetic ground segment |
| `--force_state` (collect) | `agent_pos` is 26-d `[joint pos \| applied torque]` instead of 13-d |
| `--mask_threshold` | is-handle radius, i.e. the definition of the 4th point-cloud channel |
| `--stage1_only` / `--chained` / `--stage2_only` | which env, which cameras, and the success criterion |

`agent_pos` dimension and point-cloud channel count are read back from the checkpoint config at deploy time, so those two need not be passed again.

---

## Layout

```
tasks/          Isaac Lab environments
  grasp_drill_env.py    Task 1: grasp (DirectRLEnv — scene, obs, reward, termination)
  stage2_env.py         Task 2: align, resets into an already-grasped state
  chained_env.py        Task 1 -> Task 2 in one episode (and --stage1_only)
  config/config.py      all hyperparameters (scene, randomization, perception)

perception/     observation construction, shared byte-for-byte by collect and deploy
  dp3_pointcloud.py     camera depth -> cropped fixed-size point cloud
  robot_pointcloud.py   robot FK point cloud + synthetic ground
  student_obs.py        agent_pos = joint pos (13) [| applied torque (13)]
  target_drive.py       direct joint-target drive; raw action <-> q* conversion
  groundtruth_mask.py   GT is-handle label for the optional 4th channel

scripts/        training / play / collect / deploy (see above)
tools/          one-off offline utilities
config/         drill variants, drill config, rl_games agent configs
```
