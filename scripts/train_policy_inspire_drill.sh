#!/bin/bash
# Train DP3 on the InspireDrill task.
#
# Usage:
#   bash scripts/train_policy_inspire_drill.sh inspire_drill /path/to/data.zarr 0322 0 0
#
# Args:
#   $1 = task_name (e.g. inspire_drill)
#   $2 = data_path  (path to the zarr dataset)
#   $3 = config_name (e.g. 0322)
#   $4 = seed
#   $5 = gpu_id


DEBUG=False
save_ckpt=True

task_name=${1:-inspire_drill}
data_path=${2:-/home/zeyu/inspire_drill/data/inspire_drill_dp3.zarr}
config_name=${3:-simple_dp3}
addition_info=${3:-simple}
seed=${4:-0}
gpu_id=${5:-0}

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train InspireDrill] task=${task_name} data=${data_path} exp=${exp_name} gpu=${gpu_id}\033[0m"

cd 3D-Diffusion-Policy

export PYTHONPATH="${HOME}/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/third_party/pytorch3d_simplified"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

/home/zeyu/anaconda3/envs/dp3/bin/python train.py --config-name=${config_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=online \
    checkpoint.save_ckpt=${save_ckpt} \
    task.dataset.zarr_path=${data_path}
