#!/bin/bash --login 
#SBATCH -p gpuA              # Select the type of GPU (gpuA for A100 GPUs)
#SBATCH -G 1                 # 1 GPU
#SBATCH -n 4                 # Select the no. of CPU (host) cores (fewer than training)
#SBATCH -t 0-4:00:00         # Job "wallclock" time: 4 hours (longer than single CKA since processing multiple checkpoints)
#SBATCH --mem=20G            # Memory requirement (more than single CKA since loading multiple models)
#SBATCH -J cka_evolution     # Job name
#SBATCH -o logs/cka_evolution/cka_evolution_%j.out  # Standard output log
#SBATCH -e logs/cka_evolution/cka_evolution_%j.err  # Standard error log

# Load required modules
module purge
module load libs/cuda/11.7.0 

# Create task-specific logs directory
mkdir -p logs/cka_evolution

# Define working directory
SCRATCH_DIR="/mnt/iusers01/mace01/j56806hx/scratch/Embedded_Subclusters"

# Create results directory in scratch
mkdir -p $SCRATCH_DIR/results/bigcarp/cka_evolution

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

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================

# UPDATE THESE PATHS TO MATCH YOUR ACTUAL CHECKPOINT DIRECTORIES:
PRETRAINED_DIR="artifacts/bigcarp/bigcarp_models/run_20250404_020145_pt_pfam_present"
RANDOM_DIR="artifacts/bigcarp/bigcarp_models/run_20250515_132525_rd_pfam_present2"

# Data paths (usually don't need to change these)
CORPUS="data/processed/bgc_corpus/antidb_pfam_BC.csv"
VOCAB="data/processed/vocabularies/pfam_vocab_present.json"

# Model parameters (should match your trained models)
D_EMBEDDING=1280
D_MODEL=256
N_LAYERS=32
KERNEL_SIZE=3
R=128
BATCH_SIZE=32

# Analysis parameters
MAX_CHECKPOINTS=60  # Maximum number of checkpoints to analyze
N_BATCHES=8         # Number of batches for CKA computation (more = more accurate but slower)

# GPU settings
GPU=0

# Set to true if your models use frozen embeddings, false otherwise
USE_FROZEN_EMBEDDINGS=false

# ==========================================
# BUILD COMMAND BASED ON CONFIGURATION
# ==========================================

BASE_COMMAND="python scripts/cka/cka_evolution.py \
  --pretrained_dir $PRETRAINED_DIR \
  --random_dir $RANDOM_DIR \
  --fcorpus $CORPUS \
  --fvocab $VOCAB \
  --d_embedding $D_EMBEDDING \
  --d_model $D_MODEL \
  --n_layers $N_LAYERS \
  --kernel_size $KERNEL_SIZE \
  --r $R \
  --batch_size $BATCH_SIZE \
  --gpu $GPU \
  --output_dir results/bigcarp/cka_evolution \
  --max_checkpoints $MAX_CHECKPOINTS \
  --n_batches $N_BATCHES \
  --unconditional"

# Add frozen embedding flags if needed
if [ "$USE_FROZEN_EMBEDDINGS" = true ]; then
    COMMAND="$BASE_COMMAND --pretrained_frozen --random_frozen"
else
    COMMAND="$BASE_COMMAND"
fi

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

# ==========================================
# DISPLAY CONFIGURATION AND START ANALYSIS
# ==========================================

echo "Configuration Summary:"
echo "======================"
echo "Pretrained checkpoints dir: $PRETRAINED_DIR"
echo "Random checkpoints dir: $RANDOM_DIR"
echo "Max checkpoints to analyze: $MAX_CHECKPOINTS"
echo "Batches for CKA computation: $N_BATCHES"
echo "Using frozen embeddings: $USE_FROZEN_EMBEDDINGS"
echo "Model parameters: d_embedding=$D_EMBEDDING, d_model=$D_MODEL, n_layers=$N_LAYERS"
echo ""
echo "The analysis will:"
echo "1. Load checkpoint 0 as reference (pretrained & random embedders)"
echo "2. Compare pretrained ref embedder vs embedders at different checkpoints (pretrained)"
echo "3. Compare pretrained ref embedder vs last layers at different checkpoints (pretrained)"  
echo "4. Compare pretrained ref embedder vs last layers at different checkpoints (random)"
echo "5. Compare random ref embedder vs last layers at different checkpoints (random)"
echo "6. Generate plot with four curves showing similarity evolution"
echo ""
echo "Results will be saved to: results/bigcarp/cka_evolution/cka_evolution_YYYYMMDD_HHMMSS/"
echo ""

# Run the CKA evolution analysis command
echo "Running command: $COMMAND"
echo "CKA evolution analysis started at: $(date)"
echo ""

$COMMAND

CKA_EXIT_CODE=$?

echo "CKA evolution analysis finished at: $(date)"
echo "CKA evolution analysis exit code: $CKA_EXIT_CODE"

# List the CKA results structure for verification
echo "CKA evolution analysis results structure:"
ls -la $SCRATCH_DIR/results/bigcarp/cka_evolution/ 2>/dev/null || echo "No results generated yet"

# If successful, show the latest results directory
if [ $CKA_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Latest results directory contents:"
    LATEST_DIR=$(ls -t $SCRATCH_DIR/results/bigcarp/cka_evolution/cka_evolution_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "Directory: $LATEST_DIR"
        ls -la "$LATEST_DIR"
        echo ""
        echo "Generated files:"
        echo "  - cka_evolution_plot.png: Main visualization with four curves"
        echo "  - cka_evolution_results.npy: Raw numerical results"
        echo "  - results_summary.txt: Summary of findings"
        echo "  - run_config.txt: Configuration used for this run"
    fi
fi

echo "Job completed at: $(date)"
echo "Results saved directly to $SCRATCH_DIR/results/bigcarp/cka_evolution/"
exit $CKA_EXIT_CODE
