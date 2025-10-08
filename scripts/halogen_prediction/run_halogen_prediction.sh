#!/bin/bash --login
#SBATCH --job-name=halogen_prediction
#SBATCH --output=logs/halogen_prediction/halogen_prediction_%j.out
#SBATCH --error=logs/halogen_prediction/halogen_prediction_%j.err
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=23:59:00
#SBATCH --exclusive

mkdir -p logs/halogen_prediction

WORK_DIR="/lus/lfs1aip2/scratch/u5bb/han00.u5bb/workspace/cgrep"
cd $WORK_DIR

source /scratch/u5bb/han00.u5bb/miniforge3/etc/profile.d/conda.sh
conda activate cgrep

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'cgrep'"
    exit 1
fi

DATA_PATH="data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl"
N_BOOTSTRAP=10000

MODE=${1:-full}

case $MODE in
    "quick")
        N_BOOTSTRAP=1000
        ;;
    "full")
        N_BOOTSTRAP=10000
        ;;
    "test")
        N_BOOTSTRAP=100
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

echo "Job started: $(date)"
echo "Mode: $MODE | Bootstrap samples: $N_BOOTSTRAP"
echo "SLURM Job ID: $SLURM_JOB_ID | Node: $(hostname)"

export DATASET_PATH="$DATA_PATH"
export N_BOOTSTRAP_SAMPLES="$N_BOOTSTRAP"

python scripts/halogen_prediction/halogen_embedding_comparison.py
EXIT_CODE=$?

echo "Job completed: $(date) | Exit code: $EXIT_CODE"

exit $EXIT_CODE