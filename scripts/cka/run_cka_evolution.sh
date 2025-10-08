#!/bin/bash --login
#SBATCH --job-name=cka_evolution
#SBATCH --output=logs/cka/cka_evolution_%j.out
#SBATCH --error=logs/cka/cka_evolution_%j.err
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=01:00:00

mkdir -p logs/cka

WORK_DIR="/lus/lfs1aip2/scratch/u5bb/han00.u5bb/workspace/cgrep"
cd $WORK_DIR

source /scratch/u5bb/han00.u5bb/miniforge3/etc/profile.d/conda.sh
conda activate cgrep

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'cgrep'"
    exit 1
fi

echo "Job started: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID | Node: $(hostname)"

python scripts/cka/cka_evolution.py \
  --pretrained_dir artifacts/bigcarp/bigcarp_models/run_esm_init \
  --random_dir artifacts/bigcarp/bigcarp_models/run_random_init \
  --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
  --fvocab data/processed/vocabularies/pfam_vocab_present.json \
  --d_embedding 1280 \
  --d_model 256 \
  --n_layers 32 \
  --kernel_size 3 \
  --r 128 \
  --batch_size 64 \
  --gpu 0 \
  --output_dir results/cka \
  --max_checkpoints 99 \
  --n_batches 16 \
  --unconditional

EXIT_CODE=$?

echo "Job completed: $(date) | Exit code: $EXIT_CODE"

exit $EXIT_CODE
