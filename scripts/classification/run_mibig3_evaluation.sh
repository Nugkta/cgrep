#!/bin/bash --login
#SBATCH --job-name=mibig3_eval
#SBATCH --output=logs/mibig3_evaluation/mibig3_eval_%j.out
#SBATCH --error=logs/mibig3_evaluation/mibig3_eval_%j.err
#SBATCH --gpus=1                # 1 GPU for BiLSTM models
#SBATCH --ntasks-per-gpu=1      # 1 task per GPU
#SBATCH --time=12:00:00         # 12 hours should be sufficient
#SBATCH --exclusive             # Ensure exclusive access to allocated resources

# Create task-specific logs directory
mkdir -p logs/mibig3_evaluation

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
COMMAND="python scripts/classification/mibig3_stratified_evaluation.py \
  --artifacts_dir artifacts/classification/mibig3 \
  --outdir results/mibig3_classification \
  --seed 42"

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

# Check available embedding files
echo "Available MIBiG 3.0 embedding files:"
ls -la artifacts/classification/mibig3/
echo ""

# Run evaluation
echo "Running command: $COMMAND"
echo "Evaluation started at: $(date)"
echo "Note: Results will be saved under results/mibig3_classification/"
echo ""

$COMMAND
EVAL_EXIT_CODE=$?

echo "Evaluation finished at: $(date)"
echo "Evaluation exit code: $EVAL_EXIT_CODE"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "📊 Results summary:"
    if [ -f "results/mibig3_classification/mibig3_comparison.csv" ]; then
        echo "Model comparison table:"
        cat results/mibig3_classification/mibig3_comparison.csv
    fi
else
    echo "❌ Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi

echo "Job completed at: $(date)"
echo "Results saved in $WORK_DIR/results/mibig3_classification/"

exit $EVAL_EXIT_CODE