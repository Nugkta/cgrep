#!/bin/bash --login
#SBATCH --job-name=adomain_prediction
#SBATCH --output=logs/adomain_prediction/adomain_prediction_%j.out
#SBATCH --error=logs/adomain_prediction/adomain_prediction_%j.err
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=23:59:00
#SBATCH --exclusive

mkdir -p logs/adomain_prediction

WORK_DIR="/lus/lfs1aip2/scratch/u5bb/han00.u5bb/workspace/cgrep"
cd $WORK_DIR

source /scratch/u5bb/han00.u5bb/miniforge3/etc/profile.d/conda.sh
conda activate cgrep

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'cgrep'"
    exit 1
fi

DATA_PATH="data/processed/adomain_prediction/adomain_training_dataset_full.pkl"
OUTPUT_DIR="results/adomain_prediction/bootstrap"
N_BOOTSTRAP=10
N_FOLDS=5

MODE=${1:-full}

case $MODE in
    "quick")
        N_BOOTSTRAP=3
        OUTPUT_DIR="results/adomain_prediction/bootstrap_quick"
        ;;
    "full")
        N_BOOTSTRAP=10
        OUTPUT_DIR="results/adomain_prediction/bootstrap_full"
        ;;
    "test")
        N_BOOTSTRAP=2
        OUTPUT_DIR="results/adomain_prediction/bootstrap_test"
        ;;
    *)
        echo "Invalid mode: $MODE. Usage: sbatch $0 [quick|full|test]"
        exit 1
        ;;
esac

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Job started: $(date)"
echo "Mode: $MODE | Bootstrap seeds: $N_BOOTSTRAP | CV folds: $N_FOLDS"
echo "SLURM Job ID: $SLURM_JOB_ID | Node: $(hostname)"
echo "Data path: $DATA_PATH"

python scripts/adomain_prediction/adomain_properties_prediction.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --n-bootstrap $N_BOOTSTRAP \
  --n-folds $N_FOLDS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_DIR/summary_results.json" ]; then
    echo ""
    echo "BEST METHODS (by AUC-ROC):"
    echo "-------------------------------------------"
    python -c "
import json
with open('$OUTPUT_DIR/summary_results.json', 'r') as f:
    results = json.load(f)
for prop in results:
    best_auc, best_method = 0, ''
    for method in results[prop]:
        if 'auc' in results[prop][method]:
            auc = results[prop][method]['auc']['mean']
            if auc > best_auc:
                best_auc, best_method = auc, method
    if best_method:
        ci_l = results[prop][best_method]['auc']['ci_lower']
        ci_u = results[prop][best_method]['auc']['ci_upper']
        print(f'{prop:20} | {best_method:20} | {best_auc:.3f} [{ci_l:.3f}-{ci_u:.3f}]')
"
fi

echo ""
echo "Job completed: $(date) | Exit code: $EXIT_CODE"

exit $EXIT_CODE