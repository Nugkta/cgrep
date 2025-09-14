#!/bin/bash --login
#SBATCH --job-name=adomain_prediction
#SBATCH --output=logs/adomain_prediction/adomain_prediction_%j.out
#SBATCH --error=logs/adomain_prediction/adomain_prediction_%j.err
#SBATCH --gpus=1                # 1 GPU (allocates 72 CPU cores + 115GB RAM automatically)
#SBATCH --ntasks-per-gpu=1      # 1 task per GPU
#SBATCH --time=23:59:00         # Wallclock time limit
#SBATCH --exclusive             # Ensure exclusive access to allocated resources

# Create task-specific logs directory
mkdir -p logs/adomain_prediction

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
DATA_PATH="/home/u5bb/han00.u5bb/workspace/cgrep/data/processed/adomain_prediction/adomain_training_dataset_full.pkl"
OUTPUT_DIR="results/adomain_prediction/bootstrap"
N_BOOTSTRAP=10
N_FOLDS=5
EPOCHS=50

# Parse command line argument for run mode
MODE=${1:-full}

case $MODE in
    "quick")
        echo "Running QUICK mode (3 bootstrap seeds, 25 epochs)"
        N_BOOTSTRAP=3
        EPOCHS=25
        OUTPUT_DIR="results/adomain_prediction/bootstrap_quick"
        ;;
    "full")
        echo "Running FULL mode (10 bootstrap seeds, 50 epochs)"
        N_BOOTSTRAP=10
        EPOCHS=50
        OUTPUT_DIR="results/adomain_prediction/bootstrap_full"
        ;;
    "test")
        echo "Running TEST mode (2 bootstrap seeds, 10 epochs)"
        N_BOOTSTRAP=2
        EPOCHS=10
        OUTPUT_DIR="results/adomain_prediction/bootstrap_test"
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
COMMAND="python scripts/adomain_prediction/adomain_properties_prediction_bootstrap.py \
  --data-path $DATA_PATH \
  --output-dir $OUTPUT_DIR \
  --n-bootstrap $N_BOOTSTRAP \
  --n-folds $N_FOLDS \
  --epochs $EPOCHS"

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
echo "Prediction Parameters:"
echo "  Data path: $DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Bootstrap seeds: $N_BOOTSTRAP"
echo "  CV folds: $N_FOLDS"
echo "  Epochs: $EPOCHS"
echo "  Mode: $MODE"
echo ""

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
    echo "SUCCESS: A-domain properties prediction completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - $OUTPUT_DIR/summary_results.json (means & confidence intervals)"
    echo "  - $OUTPUT_DIR/raw_results.json (all bootstrap runs)"
    echo ""

    # Quick summary of best methods
    if [ -f "$OUTPUT_DIR/summary_results.json" ]; then
        echo "QUICK SUMMARY - Best F1 scores by property:"
        echo "---------------------------------------------"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/summary_results.json', 'r') as f:
        results = json.load(f)
    for prop in results:
        best_f1 = 0
        best_method = ''
        for method in results[prop]:
            if 'f1' in results[prop][method]:
                f1_mean = results[prop][method]['f1']['mean']
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_method = method
        if best_method:
            ci_lower = results[prop][best_method]['f1']['ci_lower']
            ci_upper = results[prop][best_method]['f1']['ci_upper']
            print(f'{prop:20} | {best_method:20} | {best_f1:.3f} [{ci_lower:.3f}-{ci_upper:.3f}]')
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