# cgrep: Cross-Granularity Representation Learning for Biosynthetic Gene Clusters

## Overview
cgrep implements the experimental codebase for investigating **cross-granularity integration** in biological foundation models through a case study of biosynthetic gene clusters (BGCs). Unlike natural language, biological sequences exhibit hierarchical granularity (nucleotides → amino acids → protein domains → genes) where each level encodes distinct functional information. This repository explores how knowledge from models operating at different granularities can be combined to improve both performance and interpretability.

The repository centers on **BiGCARP**, a ByteNet-based masked language model trained at the Pfam domain level on ~127,000 BGCs from the antiSMASH database, and investigates its integration with **ESM**, a state-of-the-art amino acid-level protein language model trained on billions of sequences. Through representation analysis (CKA, UMAP) and a suite of probe tasks (product classification, A-domain substrate prediction, halogenation detection), we demonstrate that:

1. **Fine-grained (ESM) and coarse-grained (BiGCARP) models capture complementary biological knowledge** — ESM encodes local sequence patterns while BiGCARP captures long-range dependencies across entire BGCs.
2. **Cross-granularity integration improves performance** — combining representations from both models yields measurable gains in intermediate-level prediction tasks (BGC product prediction).
3. **Deeper-layer embeddings are more faithful** — last-layer (or higher-layers) representations better capture a model's learned knowledge compared to initial embedding layers, explaining why straightforward embedding initialization strategies may fail.

For detailed methodology and experimental results, see the accompanying manuscript in `cgrep_paper.pdf`.

## Repository Layout
The `scripts/` directory contains the most important scripts for running the probe tasks (product classification, A-domain substrate prediction, halogenation detection) and representation analyses (CKA, UMAP).

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
- **Environment:** Follow `environment_setup.md` for a tested dependency stack.

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



<!-- ## Referencing
If you use this code or reproduce figures, please cite the accompanying work:

> Hanlin et al., *Cross-Granularity Representation Learning for Natural Product Biosynthesis*, 2024. (See `cgrep_paper.pdf`.)

Check the manuscript for experiment-specific hyperparameters and dataset details. -->

