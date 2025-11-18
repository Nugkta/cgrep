#!/usr/bin/env python
"""
TNTM Training Script for HPC/SLURM

This script loads the corpus, vocabulary, and embeddings,
filters the corpus, sets up and trains the TNTM model,
and saves the training plots and metrics.

Usage example:
    python scripts/generate_subclusters.py \
        --corpus_path data/processed/bgc_corpus/antidb_subpfam_corpus.csv \
        --vocab_path data/processed/vocabularies/antiDB_subpfam_vocab_BC.json \
        --embedding_path outputs/subpfam_embs.pt \
        --output_dir outputs/TNTM \
        --n_topics 10 \
        --n_dims 8 \
        --epochs 10 \
        --gpu 0 \
        --early_stopping \
        --known_subclusters_path data/known_subc_domains_subpfam.txt
"""

import argparse
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
import random

# Import custom modules
import cgrep.utils as utils
from cgrep.sc_gen import TNTM_trainer
from cgrep.sc_gen import known_sub_evaluation as kse

def load_corpus_from_csv(csv_path):
    """
    Load corpus from CSV file where each row represents a BGC with domains.

    Args:
        csv_path: Path to CSV file where first column is BGC ID and
                 subsequent columns contain domains (semicolon-separated for multi-domain genes)

    Returns:
        Dictionary mapping BGC IDs to lists of domains
    """
    corpus_dict = {}

    # Read line-by-line to handle variable number of columns
    with open(csv_path, 'r') as f:
        for line in f:
            # Split by comma to get all fields
            fields = line.strip().split(',')
            if len(fields) < 2:  # Skip lines with only BGC ID or empty lines
                continue

            bgc_id = fields[0]
            domains = []

            # Process each field (skip first field which is BGC ID)
            for cell_value in fields[1:]:
                # Skip empty cells or '-' placeholders
                if not cell_value or cell_value == '-':
                    continue

                # Split by semicolon if multiple domains in one gene/CDS
                cell_domains = cell_value.split(';')
                domains.extend([d.strip() for d in cell_domains if d.strip() and d.strip() != '-'])

            corpus_dict[bgc_id] = domains

    print(f"Loaded {len(corpus_dict)} BGCs from CSV")
    return corpus_dict

def parse_args():
    """Parse command-line arguments for hyperparameters and file paths."""
    parser = argparse.ArgumentParser(description="Train TNTM on HPC using SLURM")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device id to use")
    parser.add_argument('--corpus_path', type=str, required=True, help="Path to corpus CSV file")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to vocabulary JSON file")
    parser.add_argument('--embedding_path', type=str, required=True, help="Path to embeddings file")
    parser.add_argument('--output_dir', type=str, default="../outputs", help="Directory to save outputs")
    parser.add_argument('--n_topics', type=int, default=1000, help="Number of topics")
    parser.add_argument('--n_dims', type=int, default=8, help="Number of dimensions for embeddings")
    parser.add_argument('--n_hidden_units', type=int, default=200, help="Number of hidden units")
    parser.add_argument('--n_encoder_layers', type=int, default=3, help="Number of encoder layers")
    parser.add_argument('--enc_lr', type=float, default=1e-3, help="Encoder learning rate")
    parser.add_argument('--dec_lr', type=float, default=1e-3, help="Decoder learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--dropout_rate_encoder', type=float, default=0.3, help="Dropout rate for encoder")
    parser.add_argument('--prior_variance', type=float, default=3.0, help="Prior variance")
    parser.add_argument('--reg_lambda', type=float, default=0.5, help="Regularization lambda")
    parser.add_argument('--trace_min', type=float, default=0.5, help="Minimum threshold for the trace of the variance matrix")
    parser.add_argument('--log_diag_init_eps', type=float, default=0.03, help="Initial epsilon for log_diag")
    parser.add_argument('--validation_set_size', type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument('--early_stopping', action='store_true', help="Enable early stopping")
    parser.add_argument('--n_epochs_early_stopping', type=int, default=10, help="Epochs for early stopping")
    parser.add_argument('--subset_fraction', type=float, default=1.0,
                        help="Fraction of corpus to use (e.g., 0.2 for 20%% of data)")
    parser.add_argument('--known_subclusters_path', type=str, default='data/known_subc_domains_subpfam.txt',
                        help="Path to known subclusters")
    parser.add_argument('--similarity_threshold', type=float, default=0.4,
                        help="Threshold for subcluster comparison")
    parser.add_argument('--skip_evaluation', action='store_true',
                        help="Skip the evaluation against known subclusters")
    parser.add_argument('--use_hybrid', action='store_true',
                        help="Use hybrid beta (Gaussian + ProdLDA) for topic modeling")
    parser.add_argument('--prodlda_lr', type=float, default=1e-3,
                        help="Learning rate for ProdLDA parameters")
    parser.add_argument('--prodlda_only', action='store_true',
                        help="Use only ProdLDA decoder (no Gaussian contribution)")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    if args.use_hybrid and args.prodlda_only:
        raise SystemExit("Cannot enable both hybrid and prodlda-only modes.")
    # Print all the hyperparameters
    print(json.dumps(vars(args), indent=2))
    
    # Set GPU device for torch
    torch.cuda.set_device(args.gpu)
    
    # Create a unique output directory using a timestamp
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(args.output_dir) / time_str
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Starting training at", datetime.now())
    print("Output directory:", save_path)
    
    # ------------------------------
    # Load and Preprocess the Corpus
    # ------------------------------
    print(f"Loading corpus from CSV file: {args.corpus_path}")
    database_cleaned = load_corpus_from_csv(args.corpus_path)
    corpus = list(database_cleaned.values())
    
    # Optionally use only a fraction of the corpus for training
    if args.subset_fraction < 1.0:
        subset_len = int(len(corpus) * args.subset_fraction)
        corpus = random.sample(corpus, subset_len)
        print(f"Randomly selecting {args.subset_fraction} of the dataset")
    
    # Load vocabulary from JSON file
    print(f"Loading vocabulary from JSON file: {args.vocab_path}")
    with open(args.vocab_path, 'r') as f:
        json_data = json.load(f)
        # Check if the JSON has the expected structure with a "domains" key
        if isinstance(json_data, dict) and "domains" in json_data:
            # Extract domain names from the domains dictionary
            # Skip the "UNK" token if present
            vocab = [domain for domain in json_data["domains"].keys() if domain != "UNK"]
            print(f"Loaded {len(vocab)} domain terms from JSON vocabulary")
        else:
            # If it's a simple list or different format
            vocab = json_data
            print(f"Loaded vocabulary with {len(vocab)} terms from JSON")
    
    new_embedding = torch.load(args.embedding_path, weights_only=True)
    
    print("Corpus size before filtering:", len(corpus))
    
    # Filter tokens in each document using the vocabulary
    corpus = [[token for token in bgc if token in vocab] for bgc in tqdm(corpus, desc='Filtering tokens')]
    # Remove empty documents
    corpus = [bgc for bgc in tqdm(corpus, desc='Removing empty lists') if bgc]
    
    print("Corpus size after filtering:", len(corpus))
    
    # For this example, we use the entire filtered corpus (or adjust as needed)
    corpus_short = corpus
    
    # ------------------------------
    # Initialize and Train the TNTM Model
    # ------------------------------
    trainer = TNTM_trainer.TNTMTrainer(
        n_topics=args.n_topics,
        save_path=str(save_path),
        n_dims=args.n_dims,
        n_hidden_units=args.n_hidden_units,
        n_encoder_layers=args.n_encoder_layers,
        enc_lr=args.enc_lr,
        dec_lr=args.dec_lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate_encoder=args.dropout_rate_encoder,
        prior_variance=args.prior_variance,
        prior_mean=None,
        n_topwords=50,  # Historical parameter; adjust if needed
        device="cuda",
        validation_set_size=args.validation_set_size,
        early_stopping=args.early_stopping,
        n_epochs_early_stopping=args.n_epochs_early_stopping,
        log_diag_init_eps=args.log_diag_init_eps,
        reg_lambda=args.reg_lambda,
        trace_min=args.trace_min,  # Updated from v_min to trace_min
        use_hybrid=args.use_hybrid,
        prodlda_only=args.prodlda_only,
        prodlda_lr=args.prodlda_lr
    )
    
    # Prepare data (e.g., bag-of-words representation, projected embeddings)
    trainer.prepare_data(corpus_short, vocab, new_embedding)
    
    # Build the model using data-dependent initializations
    trainer.build_model()
    
    # Train the model and retrieve training metrics and top words per topic
    topwords, probs, metrics = trainer.train()
    
    # Analyze model variances (optional)
    utils.analyze_variances_script(trainer.model, save_path)
    
    # ------------------------------
    # Save Training Plots and Metrics
    # ------------------------------
    # Plot training and validation loss curves
    train_losses = metrics['train']['loss']
    val_losses = metrics['validation']['loss']
    
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    loss_fig_path = save_path / "loss_curve.png"
    plt.savefig(loss_fig_path)
    plt.close()
    
    # Plot histogram of topic lengths (number of topwords per topic)
    topic_lens = [len(topics) for topics in topwords.values()]
    plt.figure()
    plt.hist(topic_lens, bins=40)
    plt.xlabel('Number of Topwords per Topic')
    plt.ylabel('Frequency')
    plt.title('Histogram of Topic Lengths')
    hist_fig_path = save_path / "topic_length_hist.png"
    plt.savefig(hist_fig_path)
    plt.close()
    
    # Compute and log the ratio of single-domain topics
    counts_single = sum(1 for length in topic_lens if length == 1)
    ratio_single = counts_single / len(topic_lens)
    print('Ratio of 1-domain topics:', ratio_single)

    # Skip evaluation if requested
    if args.skip_evaluation:
        print("Skipping evaluation against known subclusters (--skip_evaluation flag used)")
    else:
        fpath_known_subclusters = args.known_subclusters_path
        threshold = args.similarity_threshold

        domains_list = utils.read_known_clusters(fpath_known_subclusters)

        # Compare detected subclusters to known subclusters
        detected_subclusters = topwords
        ratio_type = 'known'  # or 'jaccard' if desired

        comparison = kse.compare_subclusters(detected_subclusters,
                                            domains_list,
                                            threshold=threshold,
                                            ratio_type=ratio_type)

        # (Optional) check for duplicates or overlaps
        duplicate_data_comp = utils.find_duplicate_values(comparison)

        # Compute and print proportion of known subclusters that are detected
        count = sum(value is not None for value in comparison.values())
        print(f"Proportion of known subclusters that are detected: {count / len(comparison)}")

        # Print detailed matches
        print("KNOWN SUBCLUSTER vs DETECTED SUBCLUSTER MATCHES:\n")
        match_count = 0
        for known_idx, detected_idx in comparison.items():
            if detected_idx is not None:
                match_count += 1
                known_domains = domains_list[known_idx]
                detected_domains = detected_subclusters[detected_idx]
                print(f"Match #{match_count}")
                print(f"Known subcluster {known_idx}: {known_domains}")
                print(f"Detected subcluster {detected_idx}: {detected_domains}")
                print("-" * 80)

        print(f"\nTotal matches found: {match_count} out of {len(comparison)} known subclusters")
        print(f"Match rate: {match_count/len(comparison)*100:.2f}%")

if __name__ == '__main__':
    main()
