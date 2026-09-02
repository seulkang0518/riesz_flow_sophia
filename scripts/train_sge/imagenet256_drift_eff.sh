#!/usr/bin/env bash
#$ -N imagenet256_drift_eff
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 4
#$ -l tmem=10G
#$ -l h_rt=12:00:00
#$ -R y
#$ -j y
#$ -o /home/zongchen/

set -eo pipefail

REPO_DIR=${RIESZ_FLOW_REPO_DIR:-/home/zongchen/riesz_flow_sophia}
SHARED_WFLOW_CACHE=${SHARED_WFLOW_CACHE:-/home/zongchen/riesz_flow_sophia/.cache}

eval "$(/home/zongchen/miniconda3/condabin/conda shell.bash hook)"
conda activate mmd_flow

cd "$REPO_DIR"

export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-$SHARED_WFLOW_CACHE/sdvae_hf_root}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-$SHARED_WFLOW_CACHE/drifting_hf_root}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-$SHARED_WFLOW_CACHE/torch_hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export CONFIG=${CONFIG:-configs/gen/imagenet256_drift_1node_eff.yaml}
export RUN_NAME=${RUN_NAME:-imagenet256_drift_eff_4gpu_official30k}
export WORKDIR=${WORKDIR:-/SAN/intelsys/imagenet_mmd_flow/$RUN_NAME}

if [[ ! -f train_drift_eff.py ]]; then
  echo "Error: train_drift_eff.py was not found in repository directory: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found: $REPO_DIR/$CONFIG" >&2
  exit 1
fi

python -m py_compile train_drift_eff.py

if [[ ! -f "$WFLOW_VAE_HF_PATH/config.json" ]]; then
  echo "Error: SD-VAE config not found at $WFLOW_VAE_HF_PATH/config.json" >&2
  exit 1
fi

MAE_METADATA="$WFLOW_DRIFTING_HF_ROOT/models/mae/jax/mae_latent_256/metadata.json"
if [[ ! -f "$MAE_METADATA" ]]; then
  echo "Error: latent MAE metadata not found at $MAE_METADATA" >&2
  exit 1
fi

export NGPU=${NGPU:-${NSLOTS:-4}}
if [[ "$NGPU" -lt 1 ]]; then
  echo "Error: invalid GPU process count: $NGPU" >&2
  exit 1
fi

export MASTER_PORT=${MASTER_PORT:-$((20000 + ${JOB_ID:-1} % 20000))}
export DRIFT_COMPILE=${DRIFT_COMPILE:-1}
export DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export CPU_THREADS_PER_GPU=${CPU_THREADS_PER_GPU:-4}
export WFLOW_NUM_WORKERS=${WFLOW_NUM_WORKERS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_THREADS_PER_GPU}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$CPU_THREADS_PER_GPU}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$CPU_THREADS_PER_GPU}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$CPU_THREADS_PER_GPU}

echo "============================================"
echo "Job:          ${JOB_ID:-N/A}"
echo "Node:         ${HOSTNAME:-N/A}"
echo "SGE slots:    ${NSLOTS:-unknown}"
echo "GPUs:         $NGPU"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repository:   $REPO_DIR"
echo "Python:       train_drift_eff.py"
echo "Config:       $CONFIG"
echo "Workdir:      $WORKDIR"
echo "Drift diagnostic settings:"
grep -n -A3 "drift_eff_kwargs" "$CONFIG" || true
echo "============================================"

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  train_drift_eff.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"
