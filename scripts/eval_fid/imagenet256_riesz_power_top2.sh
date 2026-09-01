#!/usr/bin/env bash
#$ -N imagenet256_rpower
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 2
#$ -l tmem=20G
#$ -l h_rt=6:00:00
#$ -R y
#$ -j y
#$ -o /home/zongchen/

set -eo pipefail

REPO_DIR=${REPO_DIR:-/home/zongchen/riesz_flow_sophia}
SHARED_WFLOW_CACHE=${SHARED_WFLOW_CACHE:-/home/zongchen/riesz_flow_sophia/.cache}

eval "$(/home/zongchen/miniconda3/condabin/conda shell.bash hook)"
conda activate mmd_flow

cd "$REPO_DIR"

export WFLOW_CACHE_ROOT=${WFLOW_CACHE_ROOT:-$SHARED_WFLOW_CACHE}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-$SHARED_WFLOW_CACHE/drifting_hf_root}
export WFLOW_HF_ROOT=${WFLOW_HF_ROOT:-$SHARED_WFLOW_CACHE/wflow_hf_root}
export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-$SHARED_WFLOW_CACHE/sdvae_hf_root}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-$SHARED_WFLOW_CACHE/torch_hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export NGPU=${NGPU:-${NSLOTS:-4}}
export MASTER_PORT=${MASTER_PORT:-28780}

# ---- EDIT THESE FOR THE RUN YOU WANT ----
export CONFIG=${CONFIG:-configs/gen/imagenet256_riesz_power_top2.yaml}
export RUN_DIR=${RUN_DIR:-/SAN/intelsys/imagenet_mmd_flow/imagenet256_riesz_power_top2_1node_4gpu_official30k}
export CKPT_DIR=${CKPT_DIR:-$RUN_DIR/checkpoints}
export WORK_ROOT=${WORK_ROOT:-$RUN_DIR/fid_inference_ours_cfg1p19_work}
export OUT_ROOT=${OUT_ROOT:-$RUN_DIR/fid_inference_ours_cfg1p19_json}
export FID_REF=${FID_REF:-$WFLOW_HF_ROOT/stats/jit_in256_stats.npz}
# ----------------------------------------

export CFG_SCALE=${CFG_SCALE:-1.19}
export NUM_SAMPLES=${NUM_SAMPLES:-50000}
export GEN_BSZ=${GEN_BSZ:-64}
export SEED=${SEED:-0}
export KEEP_SAMPLES=${KEEP_SAMPLES:-0}
export SKIP_EXISTING=${SKIP_EXISTING:-1}

if [[ ! -f inference_ours.py ]]; then
  echo "Error: inference_ours.py not found in $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: missing config: $CONFIG" >&2
  exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Error: missing checkpoint dir: $CKPT_DIR" >&2
  exit 1
fi

if [[ ! -f "$FID_REF" ]]; then
  echo "Error: missing FID reference: $FID_REF" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT"
mkdir -p "$OUT_ROOT"

echo "Job:              ${JOB_ID:-N/A}"
echo "Node:             ${HOSTNAME:-N/A}"
echo "Start time:       $(date)"
echo "Repository:       $REPO_DIR"
echo "Config:           $CONFIG"
echo "Run dir:          $RUN_DIR"
echo "Checkpoint dir:   $CKPT_DIR"
echo "FID reference:    $FID_REF"
echo "CFG scale:        $CFG_SCALE"
echo "Samples per ckpt: $NUM_SAMPLES"
echo "Batch per GPU:    $GEN_BSZ"
echo "GPUs:             $NGPU"
echo "Work root:        $WORK_ROOT"
echo "Output root:      $OUT_ROOT"
echo "MASTER_PORT:      $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

found_ckpt=0

for STEPNUM in 00010000; do
  CKPT="$CKPT_DIR/state_${STEPNUM}.pt"

  if [[ ! -f "$CKPT" ]]; then
    echo "Skipping missing checkpoint: $CKPT"
    continue
  fi

  found_ckpt=1

  CKPT_NAME=$(basename "$CKPT" .pt)
  WORKDIR="$WORK_ROOT/$CKPT_NAME/cfg${CFG_SCALE}"
  OUTDIR="$OUT_ROOT/$CKPT_NAME"
  JSON_OUT="$OUTDIR/results_cfg${CFG_SCALE}.json"

  if [[ "$SKIP_EXISTING" == "1" && -f "$JSON_OUT" ]]; then
    echo "Skipping existing result: $JSON_OUT"
    continue
  fi

  mkdir -p "$WORKDIR"
  mkdir -p "$OUTDIR"

  echo "========================================"
  echo "Evaluating checkpoint: $CKPT"
  echo "Workdir:               $WORKDIR"
  echo "JSON out:              $JSON_OUT"
  echo "CFG:                   $CFG_SCALE"
  echo "Samples:               $NUM_SAMPLES"
  echo "========================================"

  CMD=(
    torchrun
    --nproc_per_node="$NGPU"
    --master_port="$MASTER_PORT"
    inference_ours.py
    evaluate
    --ckpt "$CKPT"
    --config "$CONFIG"
    --cfg-scale "$CFG_SCALE"
    --num-samples "$NUM_SAMPLES"
    --gen-bsz "$GEN_BSZ"
    --fid-ref "$FID_REF"
    --seed "$SEED"
    --workdir "$WORKDIR"
    --json-out "$JSON_OUT"
  )

  if [[ "$KEEP_SAMPLES" == "1" ]]; then
    CMD+=(--keep-samples)
  fi

  NCCL_DEBUG=WARN "${CMD[@]}"

  echo ""
  echo "Result JSON:"
  cat "$JSON_OUT"
  echo ""

  MASTER_PORT=$((MASTER_PORT + 1))
done

if [[ "$found_ckpt" == "0" ]]; then
  echo "Error: no checkpoints found in $CKPT_DIR" >&2
  exit 1
fi

echo "End time: $(date)"