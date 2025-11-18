#!/usr/bin/env python
"""
Extract topics from a trained TNTM model with custom n_topwords and run evaluation.

This script:
1. Loads a trained TNTM model from a specified directory
2. Extracts top words for each topic with a custom n_topwords parameter
3. Saves the topwords and probabilities to CSV files
4. Evaluates the detected subclusters against known subclusters

Usage:
    python scripts/subcluster_generation/extract_topics_and_evaluate.py \
        --model_path results/subcluster_generation/2025-10-31_10-23-15 \
        --n_topwords 90 \
        --known_subclusters_path data/raw/known_subc_domains.txt \
        --similarity_threshold 0.4 \
        --output_suffix n90
"""

import argparse
import torch
import pickle
import pandas as pd
from pathlib import Path

# Import custom modules
import cgrep.utils as utils
from cgrep.sc_gen import TNTM_inference
from cgrep.sc_gen import known_sub_evaluation as kse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract topics from trained TNTM model and evaluate"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to trained model directory (containing model_state.pth and metadata.pkl)"
    )
    parser.add_argument(
        '--n_topwords',
        type=int,
        default=90,
        help="Number of top words to extract per topic (default: 90)"
    )
    parser.add_argument(
        '--known_subclusters_path',
        type=str,
        default='data/raw/known_subc_domains.txt',
        help="Path to known subclusters file"
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.4,
        help="Threshold for subcluster comparison (default: 0.4)"
    )
    parser.add_argument(
        '--ratio_type',
        type=str,
        default='known',
        choices=['known', 'jaccard'],
        help="Type of similarity ratio to use: 'known' or 'jaccard' (default: known)"
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='',
        help="Suffix to add to output files (e.g., 'n90' for topwords_n90.csv)"
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help="GPU device id to use (default: 0)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model_path = Path(args.model_path)
    print(f"Loading model from: {model_path}")

    # Load the trained model and metadata
    print("Loading trained TNTM model...")
    model, train_config = TNTM_inference.load_tntm_model(str(model_path), device=device)

    # Load metadata to get vocabulary and other info
    metadata_path = model_path / "metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    vocab = metadata['vocab']
    idx2word = metadata['idx2word']
    embeddings_proj = torch.tensor(metadata['embeddings_proj'], dtype=torch.float32).to(device)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of topics: {train_config.n_topics}")
    print(f"Extracting top {args.n_topwords} words per topic...")

    # Extract model parameters
    mus = model.decoder.mus
    L_lower = model.decoder.L_lower
    log_diag = model.decoder.log_diag

    # Check if model uses hybrid mode
    log_beta_prodlda = None
    lambda_logit = None
    if hasattr(model.decoder, 'log_beta_prodlda'):
        log_beta_prodlda = model.decoder.log_beta_prodlda
        lambda_logit = model.decoder.lambda_logit
        print("Model uses hybrid mode (Gaussian + ProdLDA)")

    # Extract topwords and probabilities
    topwords_arr, probs_arr = TNTM_inference.get_topwords(
        n_topwords=args.n_topwords,
        mus_res=mus,
        L_lower_res=L_lower,
        D_log_res=log_diag,
        emb_vocab_mat=embeddings_proj,
        idx2word=idx2word,
        config=train_config,
        log_beta_prodlda=log_beta_prodlda,
        lambda_logit=lambda_logit
    )

    # Convert to dictionary format for evaluation
    topwords_dict = {}
    probs_dict = {}

    for topic_idx in range(len(topwords_arr)):
        # Filter out topics with only 1 domain (if desired)
        words = topwords_arr[topic_idx].tolist()
        probs = probs_arr[topic_idx].tolist()

        topwords_dict[topic_idx] = words
        probs_dict[topic_idx] = probs

    # Determine output file names
    if args.output_suffix:
        topwords_file = model_path / f"topwords_{args.output_suffix}.csv"
        probs_file = model_path / f"probs_{args.output_suffix}.csv"
        eval_file = model_path / f"evaluation_{args.output_suffix}.txt"
    else:
        topwords_file = model_path / "topwords.csv"
        probs_file = model_path / "probs.csv"
        eval_file = model_path / "evaluation.txt"

    # Save topwords to CSV
    print(f"Saving topwords to: {topwords_file}")
    topwords_df = pd.DataFrame.from_dict(topwords_dict, orient='index')
    topwords_df.to_csv(topwords_file, index=True)

    # Save probabilities to CSV
    print(f"Saving probabilities to: {probs_file}")
    probs_df = pd.DataFrame.from_dict(probs_dict, orient='index')
    probs_df.to_csv(probs_file, index=True)

    # Run evaluation against known subclusters
    print("\n" + "="*80)
    print("EVALUATING AGAINST KNOWN SUBCLUSTERS")
    print("="*80)

    fpath_known_subclusters = args.known_subclusters_path
    threshold = args.similarity_threshold

    print(f"Loading known subclusters from: {fpath_known_subclusters}")
    domains_list = utils.read_known_clusters(fpath_known_subclusters)
    print(f"Number of known subclusters: {len(domains_list)}")

    # Compare detected subclusters to known subclusters
    detected_subclusters = topwords_dict
    ratio_type = args.ratio_type

    comparison = kse.compare_subclusters(
        detected_subclusters,
        domains_list,
        threshold=threshold,
        ratio_type=ratio_type
    )

    # Compute proportion of known subclusters that are detected
    count = sum(value is not None for value in comparison.values())
    proportion = count / len(comparison) if len(comparison) > 0 else 0

    print(f"\nProportion of known subclusters detected: {proportion:.4f} ({count}/{len(comparison)})")
    print(f"Match rate: {proportion*100:.2f}%")

    # Save detailed results to file
    print(f"\nSaving evaluation results to: {eval_file}")
    with open(eval_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TOPIC EXTRACTION AND EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of topics: {train_config.n_topics}\n")
        f.write(f"Number of top words per topic: {args.n_topwords}\n")
        f.write(f"Known subclusters file: {fpath_known_subclusters}\n")
        f.write(f"Similarity threshold: {threshold}\n")
        f.write(f"Ratio type: {ratio_type}\n")
        f.write("\n")
        f.write(f"Total known subclusters: {len(comparison)}\n")
        f.write(f"Detected known subclusters: {count}\n")
        f.write(f"Detection rate: {proportion*100:.2f}%\n")
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED MATCHES\n")
        f.write("="*80 + "\n\n")

        match_count = 0
        for known_idx, detected_idx in comparison.items():
            if detected_idx is not None:
                match_count += 1
                known_domains = domains_list[known_idx]
                detected_domains = detected_subclusters[detected_idx]

                f.write(f"Match #{match_count}\n")
                f.write(f"Known subcluster {known_idx}: {known_domains}\n")
                f.write(f"Detected subcluster {detected_idx}: {detected_domains[:20]}...\n")  # Show first 20 words
                f.write("-" * 80 + "\n")

    # Print detailed matches to console
    print("\n" + "="*80)
    print("DETAILED MATCHES (showing first 10)")
    print("="*80 + "\n")

    match_count = 0
    matches_shown = 0
    for known_idx, detected_idx in comparison.items():
        if detected_idx is not None:
            match_count += 1
            if matches_shown < 10:  # Show only first 10 matches in console
                known_domains = domains_list[known_idx]
                detected_domains = detected_subclusters[detected_idx]

                print(f"Match #{match_count}")
                print(f"Known subcluster {known_idx}: {known_domains}")
                print(f"Detected subcluster {detected_idx}: {detected_domains[:20]}...")  # Show first 20
                print("-" * 80)
                matches_shown += 1

    if match_count > 10:
        print(f"\n... and {match_count - 10} more matches (see {eval_file} for full details)")

    print("\n" + "="*80)
    print("EXTRACTION AND EVALUATION COMPLETE")
    print("="*80)
    print(f"Topwords saved to: {topwords_file}")
    print(f"Probabilities saved to: {probs_file}")
    print(f"Evaluation results saved to: {eval_file}")


if __name__ == '__main__':
    main()
