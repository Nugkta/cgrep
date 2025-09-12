#!/bin/bash --login
#SBATCH --job-name=mibig3_bootstrap
#SBATCH --output=logs/mibig3_evaluation/mibig3_bootstrap_%j.out
#SBATCH --error=logs/mibig3_evaluation/mibig3_bootstrap_%j.err
#SBATCH --gpus=1                # 1 GPU for BiLSTM models
#SBATCH --ntasks-per-gpu=1      # 1 task per GPU
#SBATCH --time=24:00:00         # 24 hours for multiple runs
#SBATCH --exclusive             # Ensure exclusive access to allocated resources

# Create task-specific logs directory
mkdir -p logs/mibig3_evaluation

# Define working directory
WORK_DIR="/home/u5bb/han00.u5bb/workspace/cgrep"

# Change to the working directory
cd $WORK_DIR

# Initialize conda and activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cgrep

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'cgrep'"
    exit 1
fi

# Define the bootstrap evaluation command
COMMAND="python scripts/classification/bootstrap_evaluation.py \
  --dataset mibig3 \
  --n_seeds 10 \
  --focus_metric macro_auc \
  --base_seed 42"

# Log system information (minimal)
echo "🚀 MIBiG 3.0 Bootstrap Evaluation Started"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "Environment: $CONDA_DEFAULT_ENV | Python: $(which python)"
echo ""

# Run bootstrap evaluation
echo "🔄 Running bootstrap evaluation with 10 random seeds..."
echo "Progress will be shown below:"
echo ""
$COMMAND
EVAL_EXIT_CODE=$?

echo ""
echo "📊 Bootstrap Evaluation Results:"
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Bootstrap evaluation completed successfully!"
    
    # Show summary results
    if [ -f "results/mibig3_bootstrap_evaluation/bootstrap_analysis/mibig3_bootstrap_summary.csv" ]; then
        echo ""
        echo "📈 Performance Summary (Macro AUC Focus):"
        echo "Model,Macro_AUC_Mean,Macro_AUC_Std,95%_CI_Lower,95%_CI_Upper"
        tail -n +2 results/mibig3_bootstrap_evaluation/bootstrap_analysis/mibig3_bootstrap_summary.csv | \
        awk -F',' '{printf "%-35s %8.4f ± %6.4f [%7.4f, %7.4f]\n", $1, $2, $3, $4, $5}' | sort -k2 -nr
    fi
else
    echo "❌ Bootstrap evaluation failed with exit code: $EVAL_EXIT_CODE"
fi

echo ""
echo "🏁 Job completed at: $(date)"
echo "📁 Results: $WORK_DIR/results/mibig3_bootstrap_evaluation/"

exit $EVAL_EXIT_CODE