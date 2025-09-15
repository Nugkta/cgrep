#!/bin/bash --login
#SBATCH --job-name=halogen_prediction
#SBATCH --output=logs/halogen_prediction/halogen_prediction_%j.out
#SBATCH --error=logs/halogen_prediction/halogen_prediction_%j.err
#SBATCH --gpus=1                # 1 GPU (allocates 72 CPU cores + 115GB RAM automatically)
#SBATCH --ntasks-per-gpu=1      # 1 task per GPU
#SBATCH --time=23:59:00         # Wallclock time limit
#SBATCH --exclusive             # Ensure exclusive access to allocated resources

# Create task-specific logs directory
mkdir -p logs/halogen_prediction

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

# Default parameters - modify as needed
DATA_PATH="data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl"
OUTPUT_DIR="results/halogen_prediction"
N_BOOTSTRAP=10000

# Parse command line argument for run mode
MODE=${1:-full}

case $MODE in
    "quick")
        echo "Running QUICK mode (1000 bootstrap samples)"
        N_BOOTSTRAP=1000
        OUTPUT_DIR="results/halogen_prediction/quick"
        ;;
    "full")
        echo "Running FULL mode (10000 bootstrap samples)"
        N_BOOTSTRAP=10000
        OUTPUT_DIR="results/halogen_prediction/full"
        ;;
    "test")
        echo "Running TEST mode (100 bootstrap samples)"
        N_BOOTSTRAP=100
        OUTPUT_DIR="results/halogen_prediction/test"
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: sbatch $0 [quick|full|test]"
        exit 1
        ;;
esac

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    echo "Please ensure the dataset is available at the expected location"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define the prediction command
COMMAND="python scripts/halogen_prediction/halogen_embedding_comparison_deep_mlp.py"

# Log system information
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU Info:"
nvidia-smi
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""
echo "Prediction Parameters:"
echo "  Data path: $DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Bootstrap samples: $N_BOOTSTRAP"
echo "  Mode: $MODE"
echo ""

# Modify the Python script to use current parameters
export DATASET_PATH="$DATA_PATH"
export N_BOOTSTRAP_SAMPLES="$N_BOOTSTRAP"
export OUTPUT_DIRECTORY="$OUTPUT_DIR"

# Run prediction
echo "Running command: $COMMAND"
echo "Prediction started at: $(date)"
echo "Note: Results will be saved under $OUTPUT_DIR/"
echo ""

$COMMAND
PREDICTION_EXIT_CODE=$?

echo "Prediction finished at: $(date)"
echo "Prediction exit code: $PREDICTION_EXIT_CODE"

if [ $PREDICTION_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Halogen presence prediction completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - $OUTPUT_DIR/embedding_comparison_results.csv"
    echo "  - $OUTPUT_DIR/paired_bootstrap_esm_vs_esm_bc_mean.json"
    echo "  - $OUTPUT_DIR/evaluation_predictions.pkl"
    echo "  - $OUTPUT_DIR/comprehensive_embedding_comparison.png"
    echo ""

    # Quick summary of results
    if [ -f "$OUTPUT_DIR/embedding_comparison_results.csv" ]; then
        echo "QUICK SUMMARY - Top 3 performing embeddings:"
        echo "---------------------------------------------"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('$OUTPUT_DIR/embedding_comparison_results.csv')
    top3 = df.head(3)
    for _, row in top3.iterrows():
        print(f'{row[\"embedding_name\"]:25} | AUC: {row[\"bootstrap_mean_auc\"]:.4f} [{row[\"bootstrap_ci_lower\"]:.4f}-{row[\"bootstrap_ci_upper\"]:.4f}]')
except Exception as e:
    print(f'Could not generate summary: {e}')
"
    fi
else
    echo "ERROR: Prediction failed with exit code $PREDICTION_EXIT_CODE"
fi

echo "Job completed at: $(date)"
echo "Results saved in $WORK_DIR/$OUTPUT_DIR"

exit $PREDICTION_EXIT_CODE