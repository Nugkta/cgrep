#!/usr/bin/env python
"""
GMM Comparison Analysis for Subpfam Embeddings

This script investigates whether Gaussian Mixture Models (GMM) better represent
known subclusters in the embedding space compared to single Gaussians.

Usage:
    python scripts/gmm_comparison/analyze_gmm_fit.py \
        --embedding_path artifacts/bigcarp/average_embeddings/subpfam/embeddings_checkpoint_best_last.pt \
        --vocab_path data/processed/vocabularies/antiDB_subpfam_vocab_BC.json \
        --known_subclusters_path data/raw/known_subc_domains.txt \
        --output_dir results/gmm_comparison
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import umap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GMM comparison analysis for subpfam embeddings")
    parser.add_argument('--embedding_path', type=str, required=True,
                        help="Path to subpfam embeddings .pt file")
    parser.add_argument('--vocab_path', type=str, required=True,
                        help="Path to vocabulary JSON file")
    parser.add_argument('--known_subclusters_path', type=str, required=True,
                        help="Path to known subclusters file")
    parser.add_argument('--output_dir', type=str, default="results/gmm_comparison",
                        help="Directory to save outputs")
    parser.add_argument('--umap_dims', type=int, default=10,
                        help="Number of UMAP dimensions")
    parser.add_argument('--max_components', type=int, default=4,
                        help="Maximum number of GMM components to test")
    parser.add_argument('--min_cluster_size', type=int, default=3,
                        help="Minimum size of cluster to analyze")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def load_embeddings(embedding_path):
    """Load embeddings from .pt file."""
    print(f"Loading embeddings from: {embedding_path}")
    embeddings = torch.load(embedding_path, map_location='cpu', weights_only=True)

    if isinstance(embeddings, dict) and 'embeddings' in embeddings:
        embeddings = embeddings['embeddings']

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def load_vocabulary(vocab_path):
    """Load vocabulary from JSON file."""
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    if isinstance(vocab_data, dict) and "domains" in vocab_data:
        vocab = [domain for domain in vocab_data["domains"].keys() if domain != "UNK"]
    else:
        vocab = vocab_data

    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_known_subclusters(known_path):
    """Load known subclusters from file."""
    print(f"Loading known subclusters from: {known_path}")
    subclusters = []

    with open(known_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse the line - could be comma or space separated
            if ',' in line:
                domains = [d.strip() for d in line.split(',') if d.strip()]
            else:
                domains = [d.strip() for d in line.split() if d.strip()]

            if domains:
                subclusters.append(domains)

    print(f"Loaded {len(subclusters)} known subclusters")
    return subclusters


def apply_umap(embeddings, n_components, random_seed):
    """Apply UMAP dimension reduction."""
    print(f"\nApplying UMAP to reduce to {n_components} dimensions...")

    # Use all available cores for UMAP (reads from OMP_NUM_THREADS env var if set to -1)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=random_seed,
        n_jobs=-1,  # Use all available cores
        verbose=True
    )

    embeddings_umap = reducer.fit_transform(embeddings)
    print(f"UMAP embeddings shape: {embeddings_umap.shape}")

    return embeddings_umap, reducer


def fit_gmm_to_cluster(cluster_embeddings, n_components, random_seed):
    """
    Fit a GMM with n_components to the cluster embeddings.

    Returns:
        dict with keys: 'n_components', 'bic', 'aic', 'log_likelihood', 'model'
    """
    if len(cluster_embeddings) < n_components:
        return None

    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_seed,
            max_iter=200,
            n_init=10
        )
        gmm.fit(cluster_embeddings)

        bic = gmm.bic(cluster_embeddings)
        aic = gmm.aic(cluster_embeddings)
        log_likelihood = gmm.score(cluster_embeddings) * len(cluster_embeddings)

        return {
            'n_components': n_components,
            'bic': bic,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'model': gmm,
            'converged': gmm.converged_
        }
    except Exception as e:
        print(f"  Warning: GMM with {n_components} components failed: {e}")
        return None


def analyze_cluster(cluster_idx, cluster_domains, vocab, embeddings_umap,
                    max_components, random_seed, vocab_to_idx):
    """
    Analyze a single cluster by fitting GMMs with different numbers of components.
    """
    # Get indices of domains in the cluster
    cluster_indices = []
    missing_domains = []

    for domain in cluster_domains:
        if domain in vocab_to_idx:
            cluster_indices.append(vocab_to_idx[domain])
        else:
            missing_domains.append(domain)

    if missing_domains:
        print(f"  Warning: {len(missing_domains)} domains not in vocabulary: {missing_domains[:3]}...")

    if len(cluster_indices) == 0:
        print(f"  Skipping cluster {cluster_idx}: no domains found in vocabulary")
        return None

    # Extract embeddings for this cluster
    cluster_embeddings = embeddings_umap[cluster_indices]

    print(f"\nAnalyzing cluster {cluster_idx}:")
    print(f"  Domains in cluster: {len(cluster_domains)}")
    print(f"  Domains found in vocab: {len(cluster_indices)}")
    print(f"  Embedding shape: {cluster_embeddings.shape}")

    # Fit GMMs with different numbers of components
    results = []
    for n_comp in range(1, max_components + 1):
        print(f"  Fitting GMM with {n_comp} component(s)...", end=' ')
        result = fit_gmm_to_cluster(cluster_embeddings, n_comp, random_seed)

        if result is not None:
            results.append(result)
            print(f"BIC={result['bic']:.2f}, AIC={result['aic']:.2f}, LL={result['log_likelihood']:.2f}")
        else:
            print("Failed")

    if not results:
        return None

    # Find best model by BIC (lower is better)
    best_by_bic = min(results, key=lambda x: x['bic'])
    best_by_aic = min(results, key=lambda x: x['aic'])

    return {
        'cluster_idx': cluster_idx,
        'n_domains': len(cluster_domains),
        'n_found': len(cluster_indices),
        'domains': cluster_domains,
        'found_domains': [vocab[i] for i in cluster_indices],
        'embeddings': cluster_embeddings,
        'gmm_results': results,
        'best_n_components_bic': best_by_bic['n_components'],
        'best_n_components_aic': best_by_aic['n_components'],
        'best_bic': best_by_bic['bic'],
        'best_aic': best_by_aic['aic']
    }


def plot_gmm_comparison(all_results, output_dir):
    """Plot comparison of GMM fits across all clusters."""

    # Prepare data for plotting
    plot_data = []
    for result in all_results:
        cluster_idx = result['cluster_idx']
        for gmm_result in result['gmm_results']:
            plot_data.append({
                'Cluster': cluster_idx,
                'N_Components': gmm_result['n_components'],
                'BIC': gmm_result['bic'],
                'AIC': gmm_result['aic'],
                'Log_Likelihood': gmm_result['log_likelihood']
            })

    df = pd.DataFrame(plot_data)

    # Plot 1: BIC comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # BIC by number of components
    ax = axes[0, 0]
    for cluster_idx in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_idx]
        ax.plot(cluster_data['N_Components'], cluster_data['BIC'],
                marker='o', label=f'Cluster {cluster_idx}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('BIC (lower is better)')
    ax.set_title('BIC vs Number of GMM Components')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # AIC by number of components
    ax = axes[0, 1]
    for cluster_idx in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_idx]
        ax.plot(cluster_data['N_Components'], cluster_data['AIC'],
                marker='s', label=f'Cluster {cluster_idx}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('AIC (lower is better)')
    ax.set_title('AIC vs Number of GMM Components')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Log-likelihood by number of components
    ax = axes[1, 0]
    for cluster_idx in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_idx]
        ax.plot(cluster_data['N_Components'], cluster_data['Log_Likelihood'],
                marker='^', label=f'Cluster {cluster_idx}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Log-Likelihood (higher is better)')
    ax.set_title('Log-Likelihood vs Number of GMM Components')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Distribution of best number of components by BIC
    ax = axes[1, 1]
    best_components = [result['best_n_components_bic'] for result in all_results]
    counts = pd.Series(best_components).value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Best Number of Components (by BIC)')
    ax.set_xticks(range(1, max(best_components) + 1))
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'gmm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_dir / 'gmm_comparison.png'}")
    plt.close()

    return df


def plot_cluster_embeddings(result, output_dir):
    """Plot embeddings for a single cluster with GMM fit."""
    cluster_idx = result['cluster_idx']
    embeddings = result['embeddings']

    if embeddings.shape[1] < 2:
        print(f"  Cannot plot cluster {cluster_idx}: need at least 2D embeddings")
        return

    # Use first 2 dimensions for visualization
    X = embeddings[:, :2]

    fig, axes = plt.subplots(1, len(result['gmm_results']),
                             figsize=(5 * len(result['gmm_results']), 5))

    if len(result['gmm_results']) == 1:
        axes = [axes]

    for ax, gmm_result in zip(axes, result['gmm_results']):
        n_comp = gmm_result['n_components']
        gmm = gmm_result['model']

        # Plot data points
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)

        # Plot GMM contours (only for 2D)
        if embeddings.shape[1] >= 2:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))

            # For plotting, we need to project GMM to 2D if embeddings are higher-D
            if embeddings.shape[1] == 2:
                Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contour(xx, yy, Z, levels=10, alpha=0.3)

        ax.set_title(f'{n_comp} Component(s)\nBIC={gmm_result["bic"]:.1f}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Cluster {cluster_idx} - GMM Fits ({result["n_found"]} domains)',
                 fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / f'cluster_{cluster_idx}_gmm_fits.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved cluster plot to {plot_path}")
    plt.close()


def generate_summary_report(all_results, df_comparison, output_dir):
    """Generate a text summary report."""
    report_path = output_dir / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GMM COMPARISON ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Total clusters analyzed: {len(all_results)}\n\n")

        # Overall statistics
        best_components_bic = [r['best_n_components_bic'] for r in all_results]
        best_components_aic = [r['best_n_components_aic'] for r in all_results]

        f.write("BEST NUMBER OF COMPONENTS (by BIC):\n")
        f.write("-" * 40 + "\n")
        for n_comp in range(1, 5):
            count = best_components_bic.count(n_comp)
            pct = 100 * count / len(best_components_bic) if best_components_bic else 0
            f.write(f"  {n_comp} component(s): {count} clusters ({pct:.1f}%)\n")

        f.write("\nBEST NUMBER OF COMPONENTS (by AIC):\n")
        f.write("-" * 40 + "\n")
        for n_comp in range(1, 5):
            count = best_components_aic.count(n_comp)
            pct = 100 * count / len(best_components_aic) if best_components_aic else 0
            f.write(f"  {n_comp} component(s): {count} clusters ({pct:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS PER CLUSTER\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            f.write(f"Cluster {result['cluster_idx']}:\n")
            f.write(f"  Domains: {result['n_found']}/{result['n_domains']}\n")
            f.write(f"  Best model (BIC): {result['best_n_components_bic']} components (BIC={result['best_bic']:.2f})\n")
            f.write(f"  Best model (AIC): {result['best_n_components_aic']} components (AIC={result['best_aic']:.2f})\n")

            f.write(f"  All fits:\n")
            for gmm_res in result['gmm_results']:
                marker = " *" if gmm_res['n_components'] == result['best_n_components_bic'] else ""
                f.write(f"    {gmm_res['n_components']} comp: BIC={gmm_res['bic']:.2f}, "
                       f"AIC={gmm_res['aic']:.2f}, LL={gmm_res['log_likelihood']:.2f}{marker}\n")
            f.write("\n")

        # Key findings
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        single_gauss = sum(1 for x in best_components_bic if x == 1)
        multi_gauss = len(best_components_bic) - single_gauss

        if multi_gauss > single_gauss:
            f.write(f"✓ {multi_gauss}/{len(best_components_bic)} ({100*multi_gauss/len(best_components_bic):.1f}%) "
                   f"clusters are better fit by GMMs with >1 component.\n")
            f.write("  → This suggests that mixture models may be superior to single Gaussians.\n\n")
        else:
            f.write(f"✓ {single_gauss}/{len(best_components_bic)} ({100*single_gauss/len(best_components_bic):.1f}%) "
                   f"clusters are best fit by a single Gaussian.\n")
            f.write("  → Single Gaussian appears sufficient for most clusters.\n\n")

        avg_best = np.mean(best_components_bic)
        f.write(f"Average optimal number of components: {avg_best:.2f}\n\n")

    print(f"\nSaved summary report to {report_path}")

    # Also print key findings to console
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    if multi_gauss > single_gauss:
        print(f"✓ {multi_gauss}/{len(best_components_bic)} ({100*multi_gauss/len(best_components_bic):.1f}%) "
              f"clusters are better fit by GMMs with >1 component.")
        print("  → This suggests that mixture models may be superior to single Gaussians.")
    else:
        print(f"✓ {single_gauss}/{len(best_components_bic)} ({100*single_gauss/len(best_components_bic):.1f}%) "
              f"clusters are best fit by a single Gaussian.")
        print("  → Single Gaussian appears sufficient for most clusters.")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    embeddings = load_embeddings(args.embedding_path)
    vocab = load_vocabulary(args.vocab_path)
    known_subclusters = load_known_subclusters(args.known_subclusters_path)

    # Check alignment
    if len(embeddings) != len(vocab):
        print(f"Warning: Embeddings ({len(embeddings)}) and vocabulary ({len(vocab)}) sizes don't match!")
        min_len = min(len(embeddings), len(vocab))
        embeddings = embeddings[:min_len]
        vocab = vocab[:min_len]
        print(f"Truncated to {min_len} entries")

    # Create vocabulary index mapping
    vocab_to_idx = {domain: idx for idx, domain in enumerate(vocab)}

    # Apply UMAP
    embeddings_umap, umap_reducer = apply_umap(embeddings, args.umap_dims, args.random_seed)

    # Save UMAP embeddings
    np.save(output_dir / 'embeddings_umap.npy', embeddings_umap)
    print(f"Saved UMAP embeddings to {output_dir / 'embeddings_umap.npy'}")

    # Analyze each cluster
    all_results = []
    for cluster_idx, cluster_domains in enumerate(tqdm(known_subclusters, desc="Analyzing clusters")):
        if len(cluster_domains) < args.min_cluster_size:
            print(f"Skipping cluster {cluster_idx}: too few domains ({len(cluster_domains)} < {args.min_cluster_size})")
            continue

        result = analyze_cluster(
            cluster_idx, cluster_domains, vocab, embeddings_umap,
            args.max_components, args.random_seed, vocab_to_idx
        )

        if result is not None:
            all_results.append(result)

            # Plot individual cluster
            plot_cluster_embeddings(result, output_dir)

    if not all_results:
        print("\nNo clusters were successfully analyzed!")
        return

    print(f"\n\nSuccessfully analyzed {len(all_results)} clusters")

    # Generate comparison plots
    df_comparison = plot_gmm_comparison(all_results, output_dir)

    # Save comparison dataframe
    df_comparison.to_csv(output_dir / 'gmm_comparison_data.csv', index=False)
    print(f"Saved comparison data to {output_dir / 'gmm_comparison_data.csv'}")

    # Generate summary report
    generate_summary_report(all_results, df_comparison, output_dir)

    # Save full results as pickle
    results_pkl_path = output_dir / 'full_results.pkl'
    with open(results_pkl_path, 'wb') as f:
        # Remove model objects to reduce file size
        results_to_save = []
        for result in all_results:
            result_copy = result.copy()
            result_copy['gmm_results'] = [
                {k: v for k, v in gmm_res.items() if k != 'model'}
                for gmm_res in result['gmm_results']
            ]
            results_to_save.append(result_copy)
        pickle.dump(results_to_save, f)
    print(f"Saved full results to {results_pkl_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
