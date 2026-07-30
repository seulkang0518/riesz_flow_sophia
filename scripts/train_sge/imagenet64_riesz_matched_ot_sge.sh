#!/usr/bin/env bash
#$ -N imagenet64_riesz_matched
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 2
#$ -l tmem=10G
#$ -l h_rt=48:00:00
#$ -R y
#$ -j y
#$ -o /home/zongchen/

set -eo pipefail

REPO_DIR=${RIESZ_FLOW_REPO_DIR:-/home/zongchen/riesz_flow_hudson}
SHARED_WFLOW_CACHE=${SHARED_WFLOW_CACHE:-/home/zongchen/riesz_flow_hudson/.cache}

eval "$(/home/zongchen/miniconda3/condabin/conda shell.bash hook)"
conda activate mmd_flow

cd "$REPO_DIR"

export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-$SHARED_WFLOW_CACHE/sdvae_hf_root}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-$SHARED_WFLOW_CACHE/drifting_hf_root}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-$SHARED_WFLOW_CACHE/torch_hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export NGPU=${NGPU:-2}
export MASTER_PORT=${MASTER_PORT:-6667}
export CONFIG=${CONFIG:-configs/gen/imagenet64_riesz_matched_ot.yaml}
export RUN_NAME=${RUN_NAME:-imagenet64_riesz_matched_ot}
export WORKDIR=${WORKDIR:-/SAN/intelsys/imagenet_mmd_flow/$RUN_NAME}
export DRIFT_COMPILE=${DRIFT_COMPILE:-0}
export DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "Job:        ${JOB_ID:-N/A}"
echo "Node:       ${HOSTNAME:-N/A}"
echo "SGE slots:  ${NSLOTS:-unknown}"
echo "GPUs:       $NGPU"
echo "Run name:   $RUN_NAME"
echo "Config:     $CONFIG"
echo "Workdir:    $WORKDIR"

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  train_riesz_support.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"
