#!/bin/bash --login 
#SBATCH -p gpuA              # Select the type of GPU (gpuA for A100 GPUs)
#SBATCH -G 1                 # 1 GPU
#SBATCH -n 4                 # Select the no. of CPU (host) cores (fewer than training)
#SBATCH -t 0-2:00:00         # Job "wallclock" time: 2 hours (CKA analysis is faster than training)
#SBATCH --mem=16G            # Memory requirement (less than training since no gradient computation)
#SBATCH -J cka_analysis      # Job name
#SBATCH -o logs/cka_analysis/cka_analysis_%j.out  # Standard output log
#SBATCH -e logs/cka_analysis/cka_analysis_%j.err  # Standard error log

# Load required modules
module purge
module load libs/cuda/11.7.0 

# Create task-specific logs directory
mkdir -p logs/cka_analysis

# Define working directory
SCRATCH_DIR="/mnt/iusers01/mace01/j56806hx/scratch/Embedded_Subclusters"

# Create results directory in scratch
mkdir -p $SCRATCH_DIR/results/bigcarp/cka

echo "Working directly from scratch directory: $SCRATCH_DIR"

# Change to the working directory
cd $SCRATCH_DIR

# Activate conda environment
source activate bigcarp

# Check if conda environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'bigcarp'"
    exit 1
fi

# Project should already be installed in editable mode in the conda environment

# Define the CKA analysis command - can now run directly or use console script
# Option 1: Run script directly
COMMAND="python scripts/cka/cka_bigcarp.py \
  --model1_path artifacts/bigcarp/bigcarp_models/run_20250515_132525_rd_pfam_present2/checkpoint0.tar \
  --model2_path artifacts/bigcarp/bigcarp_models/run_20250515_132525_rd_pfam_present2/checkpoint0.tar \
  --fcorpus data/processed/bgc_corpus/antidb_pfam_BC.csv \
  --fvocab data/processed/vocabularies/pfam_vocab_present.json \
  --d_embedding 1280 \
  --d_model 256 \
  --n_layers 32 \
  --kernel_size 3 \
  --r 128 \
  --batch_size 32 \
  --gpu 0 \
  --output_dir results/bigcarp/cka \
  --unconditional"

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

# Run the CKA analysis command
echo "Running command: $COMMAND"
echo "CKA analysis started at: $(date)"
echo ""
echo "Note: Results will be saved to a timestamped directory under results/bigcarp/cka/"
echo "Example: results/bigcarp/cka/cka_run_20250123_143022/"
echo ""

$COMMAND

CKA_EXIT_CODE=$?

echo "CKA analysis finished at: $(date)"
echo "CKA analysis exit code: $CKA_EXIT_CODE"

# List the CKA results structure for verification
echo "CKA analysis results structure:"
ls -la $SCRATCH_DIR/results/bigcarp/cka/ 2>/dev/null || echo "No results generated yet"

echo "Job completed at: $(date)"
echo "Results saved directly to $SCRATCH_DIR/results/bigcarp/cka/"
exit $CKA_EXIT_CODE
