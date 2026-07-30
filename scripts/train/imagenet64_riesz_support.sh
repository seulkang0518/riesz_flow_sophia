#!/usr/bin/env bash
#$ -l tmem=80G
#$ -l h_vmem=80G
#$ -l h_rt=24:0:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/zongchen/riesz_flow_hudson
#$ -N riesz_support
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -P aihub_ucl

source /home/zongchen/miniconda3/etc/profile.d/conda.sh
conda activate mmd_flow

set -euo pipefail

NGPU=${NGPU:-1}
MASTER_PORT=${MASTER_PORT:-6668}
CONFIG=configs/gen/imagenet64_riesz_genbank.yaml
RUN_NAME=${RUN_NAME:-imagenet64_riesz_support}
WORKDIR=${WORKDIR:-/SAN/intelsys/imagenet_mmd_flow/$RUN_NAME}

mkdir -p "$WORKDIR"
echo "Saving run to: $WORKDIR"

DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1} \
NCCL_DEBUG=WARN \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train_riesz_support.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"
