#!/bin/bash --login 
#SBATCH -p gpuV              # Select the type of GPU (gpuA for A100 GPUs)
#SBATCH -G 1                 # 1 GPU
#SBATCH -n 4                 # Select the no. of CPU (host) cores
#SBATCH -t 0-2:00:00         # Job "wallclock" time: 2 hours 
#SBATCH --mem=16G            # Memory requirement
#SBATCH -J cka_difference    # Job name
#SBATCH -o logs/cka_difference/cka_difference_%j.out  # Standard output log
#SBATCH -e logs/cka_difference/cka_difference_%j.err  # Standard error log

# Load required modules
module purge
module load libs/cuda/11.7.0 

# Create task-specific logs directory
mkdir -p logs/cka_difference

# Define working directory
SCRATCH_DIR="/mnt/iusers01/mace01/j56806hx/scratch/Embedded_Subclusters"

# Create results directory in scratch
mkdir -p $SCRATCH_DIR/results/bigcarp/cka_difference

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

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================

# UPDATE THESE PATHS TO MATCH YOUR SPECIFIC CHECKPOINTS:
# Use checkpoint 50 or whatever checkpoint number you want to compare
PRETRAINED_CHECKPOINT="artifacts/bigcarp/bigcarp_models/run_20250515_132525_rd_pfam_present2/checkpoint60.tar"
RANDOM_CHECKPOINT="artifacts/bigcarp/bigcarp_models/run_20250515_132525_rd_pfam_present2/checkpoint80.tar"

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
N_BATCHES=8         # Number of batches for CKA computation (more = more accurate but slower)

# GPU settings
GPU=0

# Set to true if your models use frozen embeddings, false otherwise
USE_FROZEN_EMBEDDINGS=false

# ==========================================
# BUILD COMMAND BASED ON CONFIGURATION
# ==========================================

BASE_COMMAND="python scripts/cka/cka_difference_heatmap.py \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
  --random_checkpoint $RANDOM_CHECKPOINT \
  --fcorpus $CORPUS \
  --fvocab $VOCAB \
  --d_embedding $D_EMBEDDING \
  --d_model $D_MODEL \
  --n_layers $N_LAYERS \
  --kernel_size $KERNEL_SIZE \
  --r $R \
  --batch_size $BATCH_SIZE \
  --gpu $GPU \
  --output_dir results/bigcarp/cka_difference \
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
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Random checkpoint: $RANDOM_CHECKPOINT"
echo "Batches for CKA computation: $N_BATCHES"
echo "Using frozen embeddings: $USE_FROZEN_EMBEDDINGS"
echo "Model parameters: d_embedding=$D_EMBEDDING, d_model=$D_MODEL, n_layers=$N_LAYERS"
echo ""
echo "The analysis will:"
echo "1. Load pretrained model at specified checkpoint"
echo "2. Load random model at specified checkpoint"
echo "3. Compute CKA matrix for pretrained model vs itself (all layers vs all layers)"
echo "4. Compute CKA matrix for random model vs itself (all layers vs all layers)"
echo "5. Create difference heatmap (pretrained - random) to show coordination convergence"
echo "6. Generate individual heatmaps and statistical analysis"
echo ""
echo "Results will be saved to: results/bigcarp/cka_difference/cka_difference_YYYYMMDD_HHMMSS/"
echo ""

# Run the CKA difference analysis command
echo "Running command: $COMMAND"
echo "CKA difference analysis started at: $(date)"
echo ""

$COMMAND

CKA_EXIT_CODE=$?

echo "CKA difference analysis finished at: $(date)"
echo "CKA difference analysis exit code: $CKA_EXIT_CODE"

# List the CKA results structure for verification
echo "CKA difference analysis results structure:"
ls -la $SCRATCH_DIR/results/bigcarp/cka_difference/ 2>/dev/null || echo "No results generated yet"

# If successful, show the latest results directory
if [ $CKA_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Latest results directory contents:"
    LATEST_DIR=$(ls -t $SCRATCH_DIR/results/bigcarp/cka_difference/cka_difference_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "Directory: $LATEST_DIR"
        ls -la "$LATEST_DIR"
        echo ""
        echo "Generated files:"
        echo "  - cka_difference_heatmap.png: Main difference plot (RED: pretrained>random, BLUE: random>pretrained)"
        echo "  - pretrained_cka_heatmap.png: Pretrained model self-similarity heatmap"
        echo "  - random_cka_heatmap.png: Random model self-similarity heatmap"
        echo "  - cka_difference_matrix.npy: Raw difference matrix data"
        echo "  - pretrained_cka_matrix.npy: Raw pretrained CKA matrix"
        echo "  - random_cka_matrix.npy: Raw random CKA matrix"
        echo "  - analysis_summary.txt: Statistical analysis and interpretation guide"
        echo "  - run_config.txt: Configuration used for this run"
        echo ""
        echo "Interpretation:"
        echo "  - Small differences suggest the models converge to similar layer coordination"
        echo "  - Large differences suggest distinct coordination patterns between pretrained/random"
        echo "  - The difference plot shows where embedding space coordination differs most"
    fi
fi

echo "Job completed at: $(date)"
echo "Results saved directly to $SCRATCH_DIR/results/bigcarp/cka_difference/"
exit $CKA_EXIT_CODE
