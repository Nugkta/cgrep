#!/bin/bash --login
#SBATCH --job-name=bigcarp_train
#SBATCH --output=logs/bigcarp_train/bigcarp_train_%j.out
#SBATCH --error=logs/bigcarp_train/bigcarp_train_%j.err
#SBATCH --gpus=1                # 1 GPU
#SBATCH --ntasks-per-gpu=1      # 1 task per GPU
#SBATCH --time=23:59:00         # Wallclock time limit
#SBATCH --exclusive             # Ensure exclusive access to allocated resources

# Create task-specific logs directory
mkdir -p logs/bigcarp_train

# Define working directory
WORK_DIR="/home/u5bb/han00.u5bb/workspace/cgrep"

# Change to the working directory
cd $WORK_DIR
echo "Working in directory: $WORK_DIR"

# Initialize conda and activate environment (before loading any modules)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cgrep

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'cgrep'"
    exit 1
fi

# Verify conda environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Define the training command
COMMAND="python scripts/train_bigcarp/train_BC.py \
  --out_fpath artifacts/bigcarp/bigcarp_models \
  --gpu 0 \
  --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
  --fvocab data/processed/vocabularies/pfam_vocab_present.json \
  --fdata None \
  --unconditional \
  --epochs 200 \
  --batch_size 512 \
  --esm_emb_fpath artifacts/bigcarp/esm_embeddings/esm1b_pfam_embs_present.pt \
  --pretrain
  "
#  --pretrain--freeze

# Log system information
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU Info:"
nvidia-smi
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run training
echo "Running command: $COMMAND"
echo "Training started at: $(date)"
echo "Note: Checkpoints will be saved under artifacts/bigcarp/bigcarp_models/"
echo ""

$COMMAND
TRAIN_EXIT_CODE=$?

echo "Training finished at: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "Job completed at: $(date)"
echo "Results saved in $WORK_DIR"

exit $TRAIN_EXIT_CODE