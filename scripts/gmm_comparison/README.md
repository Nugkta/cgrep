# GMM Comparison Analysis

This directory contains scripts for analyzing whether Gaussian Mixture Models (GMM) better represent known subclusters in the embedding space compared to single Gaussians.

## Motivation

The current TNTM model uses a single Gaussian distribution to represent each topic in the embedding space. However, subclusters might have multi-modal distributions that would be better represented by Gaussian Mixtures. This analysis investigates whether GMMs provide a better fit.

## Files

- `analyze_gmm_fit.py` - Main analysis script
- `run_gmm_analysis.sh` - HPC submission script (SLURM)
- `README.md` - This file

## Analysis Pipeline

1. **Load Data**
   - Subpfam embeddings from BigCarp model
   - Vocabulary mapping
   - Known subclusters

2. **Dimensionality Reduction**
   - Apply UMAP to reduce embeddings to 10 dimensions
   - Preserves local structure while making GMM fitting tractable

3. **GMM Fitting**
   - For each known subcluster:
     - Extract embeddings of domains in that subcluster
     - Fit GMMs with 1, 2, 3, 4 components
     - Compare using BIC, AIC, and log-likelihood

4. **Analysis & Visualization**
   - Compare model fits across different numbers of components
   - Identify optimal number of components for each cluster
   - Generate summary statistics and visualizations

## Running on HPC

### Submit the job:
```bash
cd /lus/lfs1aip2/scratch/u5bb/han00.u5bb/workspace/cgrep
sbatch scripts/gmm_comparison/run_gmm_analysis.sh
```

### Check job status:
```bash
squeue -u $USER
```

### Monitor output:
```bash
tail -f logs/gmm_comparison/gmm_analysis_<JOB_ID>.out
```

### Check errors:
```bash
tail -f logs/gmm_comparison/gmm_analysis_<JOB_ID>.err
```

## Running Locally

```bash
python scripts/gmm_comparison/analyze_gmm_fit.py \
    --embedding_path artifacts/bigcarp/average_embeddings/subpfam/embeddings_checkpoint_best_last.pt \
    --vocab_path data/processed/vocabularies/antiDB_subpfam_vocab_BC.json \
    --known_subclusters_path data/raw/known_subc_domains.txt \
    --output_dir results/gmm_comparison \
    --umap_dims 10 \
    --max_components 4 \
    --min_cluster_size 3
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding_path` | Required | Path to subpfam embeddings .pt file |
| `--vocab_path` | Required | Path to vocabulary JSON file |
| `--known_subclusters_path` | Required | Path to known subclusters file |
| `--output_dir` | `results/gmm_comparison` | Output directory |
| `--umap_dims` | `10` | Number of UMAP dimensions |
| `--max_components` | `4` | Maximum number of GMM components to test |
| `--min_cluster_size` | `3` | Minimum cluster size to analyze |
| `--random_seed` | `42` | Random seed for reproducibility |

## Output Files

The analysis generates the following outputs in `results/gmm_comparison/`:

### Main Results
- `summary_report.txt` - Text summary of key findings
- `gmm_comparison.png` - Overview plots comparing all models
- `gmm_comparison_data.csv` - Tabular data of all model fits
- `full_results.pkl` - Complete results (pickle format)

### Per-Cluster Visualizations
- `cluster_<N>_gmm_fits.png` - Visualization of GMM fits for each cluster

### Intermediate Files
- `embeddings_umap.npy` - UMAP-reduced embeddings (10D)

## Interpreting Results

### BIC (Bayesian Information Criterion)
- **Lower is better**
- Penalizes model complexity
- Conservative: tends to favor simpler models

### AIC (Akaike Information Criterion)
- **Lower is better**
- Less penalty for complexity than BIC
- May prefer more complex models

### Log-Likelihood
- **Higher is better**
- Measures model fit without complexity penalty
- Always improves with more components

### Key Questions to Answer

1. **What proportion of clusters are best fit by GMMs with >1 component?**
   - If >50%: Suggests GMMs may be superior
   - If <50%: Single Gaussians appear sufficient

2. **What is the average optimal number of components?**
   - ~1: Single Gaussian sufficient
   - >2: Strong evidence for mixture models

3. **Which clusters benefit most from GMMs?**
   - Identifies which subclusters have multi-modal structure
   - May inform model design choices

## Next Steps Based on Results

### If GMMs are Superior (>1 component for most clusters):

**Option A: Full GMM Decoder**
- Replace single Gaussian with GMM in decoder
- Each topic represented as mixture of Gaussians
- More flexible but more parameters

**Option B: Hybrid Approach**
- Use single Gaussian for simple topics
- Use GMM for complex topics
- Adaptive model complexity

### If Single Gaussian is Sufficient (1 component for most clusters):

- Keep current architecture
- May increase embedding dimensions instead
- Focus on other improvements (like hybrid beta)

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- umap-learn
- tqdm

All dependencies should be in the `subcluster` conda environment.

## Troubleshooting

### Memory Issues
- Reduce `--max_components`
- Reduce `--umap_dims`
- Increase SLURM memory allocation

### UMAP Taking Too Long
- Reduce number of neighbors in UMAP (edit script)
- Use PCA instead of UMAP for initial testing

### GMM Fitting Fails
- Check if clusters have enough samples (increase `--min_cluster_size`)
- Check for NaN values in embeddings
- Try simpler covariance type (edit script: 'full' → 'diag')

## References

- UMAP: McInnes et al. (2018) "UMAP: Uniform Manifold Approximation and Projection"
- GMM: Bishop (2006) "Pattern Recognition and Machine Learning", Chapter 9
- Model Selection: Schwarz (1978) "Estimating the Dimension of a Model" (BIC)

## Contact

For questions or issues, contact the research team.
