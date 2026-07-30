#$ -l tmem=160G
#$ -l h_vmem=160G
#$ -l h_rt=24:0:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/zongchen/riesz_flow_sophia
#$ -N imagenet256_riesz_8gpu
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 8
#$ -P aihub_ucl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmd_flow
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

export IMAGENET_CACHE_PATH=/SAN/intelsys/imagenet_mmd_flow/imagenet256-latents-sdvae
export WFLOW_VAE_HF_PATH=/SAN/intelsys/imagenet_mmd_flow/cache/sdvae_hf_root
export WFLOW_DRIFTING_HF_ROOT=/SAN/intelsys/imagenet_mmd_flow/cache/drifting_hf_root

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DRIFT_COMPILE=0

date
which python
nvidia-smi

torchrun \
  --standalone \
  --nproc_per_node=8 \
  train.py \
  --config configs/gen/imagenet256_riesz_8gpu.yaml \
  --workdir /SAN/intelsys/imagenet_mmd_flow/imagenet256_riesz_sophia_8gpu
