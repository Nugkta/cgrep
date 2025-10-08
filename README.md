# cgrep: Cross-Granularity Representation Learning for Biosynthetic Gene Clusters

cgrep implements the experimental codebase behind our cross-granularity representation learning work on biosynthetic gene clusters (BGCs). The repository centers on **BiGCARP**, a ByteNet-based masked language model trained on Pfam domain sequences, and a suite of downstream evaluations covering product class classification, substrate-property prediction, halogenation prediction, and representation analyses (CKA, UMAP). For methodological background, see the accompanying manuscript in `cgrep_paper.pdf`.

## Repository Layout
| Path | Description |
| --- | --- |
| `cgrep/` | Core Python package with shared data utilities (`utils.py`), BiGCARP data loaders (`bigcarp_functions.py`), and helper models for downstream tasks (`models_multiclass.py`). |
| `scripts/train_bigcarp/` | Training entrypoints for BiGCARP (`train_BC.py`) with extensive CLI options for GPU training, checkpointing, and embedding initialisation. |
| `scripts/classification/` | Product-class evaluation pipelines (`mibig*_stratified_evaluation.py`, `bootstrap_evaluation.py`) and fine-tuning scripts for MIBiG datasets. |
| `scripts/cka/` | Centered Kernel Alignment studies (`cka_difference.py`, `cka_evolution.py`) comparing random vs. ESM initialisation during training. |
| `scripts/umap/` | Embedding extraction and UMAP visualisation tooling for checkpointed models. |
| `scripts/adomain_prediction/`, `scripts/halogen_prediction/` | Downstream biochemical property predictors for NRPS A-domains and halogenated products. |
| `data/` | Expected data layout (`raw/`, `processed/`) for corpora, vocabularies, and task-specific datasets. Raw data are not bundled and should be downloaded from XXX. **Probe task data** (A-domain substrate properties, halogenation labels, MIBiG product classifications) can be downloaded from **XXX**. Note: the data curation process is not included in this repository. |
| `artifacts/`, `results/` | Default output locations for trained checkpoints, extracted embeddings, metrics, plots, and tables. |
| `notebooks/` | Jupyter notebooks for MIBiG preprocessing (`bgc_classification/`). |
| `environment_setup.md` | Conda-based environment bootstrap instructions (Python 3.10, PyTorch, bioinformatics dependencies). |
| `external/protein-sequence-models/` | Submodule/clone required for the ByteNet implementation (`sequence_models`), **changed are made on the original repository for this project.** |

## Getting Started
- **Prerequisites:** Python 3.10, Conda (or mamba), CUDA-capable GPU for model training, and adequate disk space for artefacts. Network access is required for installing optional dependencies (`protein-sequence-models`, `centered-kernel-alignment`).
- **Environment:** Follow `environment_setup.md` for a tested dependency stack. Quick start:
- **GPU notebooks (optional):** Submit `start_jupyter.sh` to your SLURM queue and tunnel to the printed hostname.

## Data Preparation
The code expects tokenised BGC corpora and vocabularies in CSV/JSON form:
- Corpus CSV: columns `domains` (semicolon-separated Pfam domains), `function`, and `split`.
- Vocabulary JSON: keys `specials`, `domains`, `size` mapping domain names to token IDs.
- Downstream datasets (MIBiG embeddings, A-domain properties, halogenation labels) live in `data/processed/<task>/`. Populate these directories using your own preprocessing or the provided notebooks (`notebooks/bgc_classification/*`).

## Training BiGCARP
BiGCARP is trained with masked language modelling. Typical invocation:

```bash
python scripts/train_bigcarp/train_BC.py \
  --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
  --fvocab data/processed/vocabularies/pfam_vocab_present.json \
  --out_fpath artifacts/bigcarp/models/run_esm_init \
  --epochs 100 \
  --batch_size 256 \
  --esm_emb_fpath artifacts/bigcarp/esm_embeddings/domain_embeddings.pt \
  --pretrain  # load (and fine-tune) ESM embeddings; use --freeze to keep them fixed
```

Key flags:
- `--conditional` prepends function tokens for conditional modelling.
- `--ar` toggles causal (autoregressive) training.
- `--restart` resumes from `checkpoint_latest.tar` in the output directory.
- `--fcorpus`, `--fvocab`, and `--esm_emb_fpath` should align with your data artefacts.

Checkpoints, metrics (`metrics.csv`), and plots (`loss_plot.png`) are written under `artifacts/bigcarp/models/<run_name>/`.

## Downstream Evaluations
### BGC Product Class Classification (MIBiG)
- **Scripts:** `scripts/classification/mibig3_stratified_evaluation.py`, `scripts/classification/bootstrap_evaluation.py`, `scripts/classification/fine_tuning/finetune_mibig_multilabel.py`.
- **Inputs:** Pickled embedding tables in `artifacts/classification/mibig*/`.
- **Run:** 
  ```bash
  python scripts/classification/mibig3_stratified_evaluation.py \
    --artifacts_dir artifacts/classification/mibig3 \
    --outdir results/classification/mibig3 \
    --seed 42
  ```
- Bootstraped multi-seed comparisons: `python scripts/classification/bootstrap_evaluation.py --dataset mibig3 --n_seeds 10`.
- Fine-tuning on BGC sequences: adjust CLI args in `scripts/classification/fine_tuning/finetune_mibig_multilabel.py` (defaults mirror the evaluation pipeline).

### A-Domain Substrate Property Prediction
- **Script:** `scripts/adomain_prediction/adomain_properties_prediction.py`.
- **Dataset:** `data/processed/adomain_prediction/adomain_training_dataset_full.pkl`.
- **Run:** `python scripts/adomain_prediction/adomain_properties_prediction.py --n-bootstrap 10 --n-folds 5 --output-dir results/adomain_prediction/bootstrap`.
- Produces JSON summaries (`summary_results.json`, `raw_results.json`) with 95% confidence intervals per property and embedding combination.

### Halogenation Detection
- **Script:** `scripts/halogen_prediction/halogen_embedding_comparison.py`.
- **Dataset:** `data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl`.
- **Run:** `python scripts/halogen_prediction/halogen_embedding_comparison.py` (set `DATASET_PATH` and `N_BOOTSTRAP_SAMPLES` env vars to override defaults).
- Outputs CSV/JSON metrics, bootstrap confidence intervals, and comparative plots in `results/halogen_prediction/`.

## Representation Analyses
### CKA Difference & Coordination
- **Script:** `scripts/cka/cka_difference.py`.
- **Usage:** 
  ```bash
  python scripts/cka/cka_difference.py \
    --pretrained_checkpoint artifacts/bigcarp/models/run_esm_init/checkpoint_epoch10.tar \
    --random_checkpoint artifacts/bigcarp/models/run_random/checkpoint_epoch10.tar \
    --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
    --fvocab data/processed/vocabularies/pfam_vocab_present.json \
    --output_dir results/cka/difference_epoch10
  ```
- Generates NumPy matrices and publication-grade heatmaps comparing inter-layer coordination.

### CKA Evolution During Training
- **Script:** `scripts/cka/cka_evolution.py`.
- **Usage:** 
  ```bash
  python scripts/cka/cka_evolution.py \
    --pretrained_dir artifacts/bigcarp/models/run_esm_init \
    --random_dir artifacts/bigcarp/models/run_random \
    --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
    --fvocab data/processed/vocabularies/pfam_vocab_present.json \
    --output_dir results/cka/evolution \
    --max_checkpoints 20
  ```
- Tracks representation drift across checkpoints for both initialisation regimes.

## Embedding Extraction & UMAP Visualisation
- **Script:** `scripts/umap/extract_average_embeddings.py` (bulk or single checkpoint modes).
- **Example:** 
  ```bash
  python scripts/umap/extract_average_embeddings.py bulk \
    --checkpoint-dir artifacts/bigcarp/models/run_esm_init \
    --vocab-path data/processed/vocabularies/pfam_vocab_present.json \
    --corpus-path data/processed/bgc_corpus/antidb_pfam_corpus.csv \
    --save-dir artifacts/bigcarp/average_embeddings/run_esm_init \
    --layer-indices last
  ```
- Visualise with `scripts/umap/plot_UMAP_interact.py`, which loads saved embeddings and produces static or Plotly interactive UMAPs (`results/umap/`).

## Results & Artefacts
- Model checkpoints, embedding dumps, and evaluation tables default to `artifacts/`.
- Publication-ready figures and summary tables (CKA, classification, property prediction) live in `results/`.
- Logs from long training jobs can be routed to `logs/` (not version-controlled by default).

## Notebooks
Preprocessing notebooks for MIBiG releases sit in `notebooks/bgc_classification/`. Launch them inside the configured environment (`conda activate cgrep`) or via the SLURM-based Jupyter job.

## Referencing
If you use this code or reproduce figures, please cite the accompanying work:

> Hanlin et al., *Cross-Granularity Representation Learning for Natural Product Biosynthesis*, 2024. (See `cgrep_paper.pdf`.)

Check the manuscript for experiment-specific hyperparameters and dataset details.

