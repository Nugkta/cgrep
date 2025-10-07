#!/bin/bash --login
#SBATCH --job-name=mibig3_bootstrap
#SBATCH --output=logs/mibig3_evaluation/mibig3_bootstrap_%j.out
#SBATCH --error=logs/mibig3_evaluation/mibig3_bootstrap_%j.err
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=24:00:00
#SBATCH --exclusive

# Setup environment
mkdir -p logs/mibig3_evaluation
cd "/lus/lfs1aip2/scratch/u5bb/han00.u5bb/workspace/cgrep"
source /scratch/u5bb/han00.u5bb/miniforge3/etc/profile.d/conda.sh
conda activate cgrep

# Run bootstrap evaluation
python scripts/classification/bootstrap_evaluation.py \
  --dataset mibig3 \
  --n_seeds 10 \
  --focus_metric macro_auc \
  --base_seed 42

# Output summary
echo ""
echo "Bootstrap evaluation completed at: $(date)"