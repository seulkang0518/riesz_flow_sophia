#$ -l tmem=80G
#$ -l h_vmem=80G
#$ -l h_rt=24:0:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/zongchen/riesz_flow_sophia
#$ -N imagenet256_latents
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -P aihub_ucl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmd_flow

export WFLOW_VAE_HF_PATH=/SAN/intelsys/imagenet_mmd_flow/cache/sdvae_hf_root

python -m dataset.latent \
  --data-path /SAN/intelsys/imagenet_mmd_flow/ILSVRC/Data/CLS-LOC \
  --target-path /SAN/intelsys/imagenet_mmd_flow/imagenet256-latents-sdvae \
  --local-batch-size 32 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --pin-memory
