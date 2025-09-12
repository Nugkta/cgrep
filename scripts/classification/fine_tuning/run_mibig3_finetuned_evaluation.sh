#!/bin/bash --login
#SBATCH --job-name=bigcarp_arch_comparison
#SBATCH --output=logs/bigcarp_architecture_comparison/bigcarp_arch_comparison_%j.out
#SBATCH --error=logs/bigcarp_architecture_comparison/bigcarp_arch_comparison_%j.err
#SBATCH --gpus=1                # 1 GPU for BigCarp fine-tuning
#SBATCH --ntasks-per-gpu=1      # 1 task per GPU
#SBATCH --time=10:00:00         # Extended time for comprehensive comparison
#SBATCH --exclusive             # Ensure exclusive access to allocated resources

# Create task-specific logs directory
mkdir -p logs/bigcarp_architecture_comparison

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

# Define the evaluation command
OUTPUT_DIR="results/bigcarp_architecture_comparison"
COMMAND="python scripts/classification/fine_tuning/bigcarp_architecture_comparison.py \
  --artifacts_dir artifacts/classification/mibig3 \
  --outdir $OUTPUT_DIR \
  --vocab_path /home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json \
  --seed 42 \
  --batch_size 16 \
  --epochs 10 \
  --lr 1e-4 \
  --patience 3 \
  --freeze_encoder"

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

# Check available BigCarp model files
echo "Available BigCarp model files:"
ls -la artifacts/bigcarp/bigcarp_models/paper_models/
echo ""

# Run evaluation
echo "Running command: $COMMAND"
echo "Architecture comparison started at: $(date)"
echo "Note: Results will be saved under $OUTPUT_DIR/"
echo ""

$COMMAND
EVAL_EXIT_CODE=$?

echo "Architecture comparison finished at: $(date)"
echo "Architecture comparison exit code: $EVAL_EXIT_CODE"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Architecture comparison completed successfully!"
    echo "📊 Results summary:"
    if [ -f "$OUTPUT_DIR/architecture_comparison.csv" ]; then
        echo "Architecture comparison table:"
        cat "$OUTPUT_DIR/architecture_comparison.csv"
    fi
else
    echo "❌ Architecture comparison failed with exit code: $EVAL_EXIT_CODE"
fi

echo "Job completed at: $(date)"
echo "Results saved in $WORK_DIR/$OUTPUT_DIR/"

exit $EVAL_EXIT_CODE