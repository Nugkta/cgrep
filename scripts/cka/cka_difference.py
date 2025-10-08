"""
CKA Difference Heatmap Analysis for bigcarp Models
===================================================

This script analyzes internal layer coordination differences between ESM-initialized
and random-initialized models by computing layer-wise CKA similarity matrices and
their difference.

Analysis Strategy:
    1. Compute self-CKA matrix for ESM-initialized model (all layers vs all layers)
    2. Compute self-CKA matrix for random-initialized model (all layers vs all layers)
    3. Calculate difference matrix: CKA_ESM - CKA_Random
    4. Visualize with heatmaps to identify coordination patterns

Interpretation:
    - Diagonal: Layer self-similarity (always 1.0)
    - Off-diagonal: Inter-layer coordination
    - Positive differences: ESM-initialized shows higher coordination
    - Negative differences: Random-initialized shows higher coordination
    - Block patterns: Groups of coordinated layers

Model Types:
    - ESM-initialized: ByteNetLM with pretrained ESM embeddings (optionally frozen)
    - Random-initialized: ByteNetLM with random weight initialization

Metrics:
    - CKA similarity matrices (n_layers * n_layers)
    - Difference statistics (mean, std, min, max, Frobenius norm)

Usage:
    python cka_difference.py \\
        --pretrained_checkpoint checkpoints/esm_init/checkpoint_epoch10.tar \\
        --random_checkpoint checkpoints/random_init/checkpoint_epoch10.tar \\
        --fcorpus data/corpus.txt \\
        --fvocab data/vocab.txt \\
        --n_layers 32 \\
        --n_batches 8

Output Files:
    - pretrained_cka_matrix.npy: ESM-initialized self-CKA matrix
    - random_cka_matrix.npy: Random-initialized self-CKA matrix
    - cka_difference_matrix.npy: Difference matrix
    - pretrained_cka_heatmap.pdf: ESM-initialized heatmap visualization
    - random_cka_heatmap.pdf: Random-initialized heatmap visualization
    - cka_difference_heatmap.pdf: Difference heatmap visualization
    - analysis_summary.txt: Statistical summary of differences
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sequence_models.convolutional import ByteNetLM
from cgrep.bigcarp_functions import load_data, prepare_dataloaders

# Publication-quality plotting setup
plt.style.use('default')  # Use clean default style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': False,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white'
})

def get_parser():
    """
    Create argument parser for CKA difference heatmap analysis.

    Returns:
        argparse.ArgumentParser: Configured parser with all required and optional arguments
    """
    parser = argparse.ArgumentParser(description='CKA difference heatmap analysis of bigcarp models')
    
    # Paths to specific checkpoints
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                        help='Path to ESM-initialised model checkpoint')
    parser.add_argument('--random_checkpoint', type=str, required=True,
                        help='Path to random initialized model checkpoint')
    
    # Data arguments (same as in train_BC.py)
    parser.add_argument('--fcorpus', type=str, required=True,
                        help='Path to the corpus file for evaluation')
    parser.add_argument('--fvocab', type=str, required=True,
                        help='Path to the vocabulary file')
    
    # Model parameters (should match the trained models)
    parser.add_argument('--d_embedding', type=int, default=1280, help='Dimension of embedding')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension within ByteNet model')
    parser.add_argument('--n_layers', type=int, default=32, help='Number of ByteNet layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel width')
    parser.add_argument('--r', type=int, default=128, help='Used to calculate dilation factor')
    parser.add_argument('--wide', action='store_true', 
                        help='If set, use the "wide" version instead of the "slim" version of ByteNet')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--ar', action='store_true',
                        help='If set, use autoregressive model (causal)')
    parser.add_argument('--unconditional', action='store_true',
                        help='If set, do not prepend a special function token')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--output_dir', type=str, default='results/bigcarp/cka_difference',
                        help='Directory to save CKA difference analysis results')
    
    # Model-specific frozen embedding flags
    parser.add_argument('--pretrained_frozen', action='store_true',
                        help='If set, load ESM-initialised model with frozen embeddings')
    parser.add_argument('--random_frozen', action='store_true',
                        help='If set, load random model with frozen embeddings')
    
    # Analysis parameters
    parser.add_argument('--n_batches', type=int, default=8,
                        help='Number of batches to use for CKA computation')
    
    return parser

def load_model(checkpoint_path, n_tokens, mask_idx, domains, is_frozen, args):
    """
    Load a ByteNetLM model from checkpoint file.

    Args:
        checkpoint_path (str): Path to checkpoint .tar file
        n_tokens (int): Vocabulary size
        mask_idx (int): Index for mask token
        domains (list): List of domain tokens for frozen embedding calculation
        is_frozen (bool): Whether to use frozen embeddings
        args (argparse.Namespace): Arguments containing model architecture parameters:
            - d_embedding (int): Embedding dimension
            - d_model (int): Model hidden dimension
            - n_layers (int): Number of ByteNet layers
            - kernel_size (int): Convolution kernel size
            - r (int): Dilation factor base
            - wide (bool): Use wide ByteNet variant
            - ar (bool): Use autoregressive (causal) model

    Returns:
        ByteNetLM: Loaded model with checkpoint weights
    """
    # Calculate number of frozen embeddings if needed
    n_frozen_embs = None
    if is_frozen:
        n_frozen_embs = len(domains) - 1
        print(f"Using frozen embeddings with n_frozen_embs={n_frozen_embs}")
    
    model = ByteNetLM(
        n_tokens=n_tokens,
        d_embedding=args.d_embedding,
        d_model=args.d_model,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        r=args.r,
        slim=(not args.wide),
        padding_idx=mask_idx,
        causal=args.ar,
        final_ln=True,
        activation='gelu',
        n_frozen_embs=n_frozen_embs  # Parameter for frozen embeddings
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def compute_cka_matrix_self(model, dataloader, device, padding_idx, n_batches=8):
    """
    Compute self-CKA matrix for a model (all layers vs all layers).

    Computes pairwise CKA similarity between all layer representations within a single model.
    Uses batch accumulation to efficiently compute HSIC and variance terms across multiple batches.

    Implementation:
        - Accumulates HSIC (Hilbert-Schmidt Independence Criterion) across batches
        - Accumulates variance terms for normalization
        - Final CKA = HSIC / sqrt(var1 * var2) for each layer pair

    Args:
        model (ByteNetLM): Model to analyze
        dataloader (DataLoader): Data loader for evaluation samples
        device (torch.device): Device for computation (CPU or CUDA)
        padding_idx (int): Token ID used for padding (positions to exclude from CKA)
        n_batches (int): Number of batches to use for CKA computation (default: 8)

    Returns:
        numpy.ndarray: Self-CKA matrix of shape (n_layers, n_layers) with values in [0, 1].
            Element [i, j] represents CKA similarity between layer i and layer j.
            Diagonal elements are always 1.0 (perfect self-similarity).
    """
    model.eval()

    hsic_sum = var1_sum = var2_sum = None

    # Create iterator once to properly iterate through different batches
    dataloader_iter = iter(dataloader)

    for batch_idx in range(n_batches):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # If we run out of batches, break early
            break

        src, _, _ = [b.to(device) for b in batch]
        input_mask = (src != padding_idx).float().unsqueeze(-1)
        mask_flat = input_mask.squeeze(-1).reshape(-1).bool()

        with torch.no_grad():
            _, hidden_states = model.embedder(src, input_mask=input_mask, return_all_hidden_states=True)

        if hsic_sum is None:  # allocate on first pass
            n_layers = len(hidden_states)
            hsic_sum = torch.zeros(n_layers, n_layers, device=device)
            var1_sum = torch.zeros(n_layers, device=device)
            var2_sum = torch.zeros(n_layers, device=device)

        # Outer loop over layers (model 1)
        for i, h1 in enumerate(hidden_states):
            f1 = h1.reshape(-1, h1.size(-1))[mask_flat]
            f1 -= f1.mean(0, keepdim=True)
            gram1 = f1 @ f1.T
            var1 = (gram1 * gram1).sum()
            var1_sum[i] += var1

            # Inner loop over layers (model 2 - same model)
            for j, h2 in enumerate(hidden_states):
                f2 = h2.reshape(-1, h2.size(-1))[mask_flat]
                f2 -= f2.mean(0, keepdim=True)
                gram2 = f2 @ f2.T

                if i == 0:  # add each var2 only once per batch
                    var2_sum[j] += (gram2 * gram2).sum()

                hsic_sum[i, j] += (gram1 * gram2).sum()

    # Final normalization
    denom = torch.outer(torch.sqrt(var1_sum), torch.sqrt(var2_sum))
    cka_matrix = (hsic_sum / denom).cpu().numpy()
    return cka_matrix


def create_heatmap_plot(cka_matrix, output_path, model_name=""):
    """
    Create and save a CKA similarity heatmap visualization.

    Generates a square heatmap with viridis colormap showing layer-wise CKA similarities.
    Includes intelligently spaced axis tick labels to avoid overcrowding.

    Args:
        cka_matrix (numpy.ndarray): CKA similarity matrix of shape (n_layers, n_layers)
        output_path (str): Path to save the PDF plot
        model_name (str): Model name for axis labels (default: "")

    Output:
        - Saves PDF file at output_path with 300 DPI resolution
        - Colormap range: [0, 1]
        - Square aspect ratio
        - Publication-quality formatting
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    num_layers = len(cka_matrix)
    step = max(1, num_layers // 8)
    tick_positions = np.arange(0, num_layers, step)
    if num_layers - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_layers - 1)
        tick_positions = np.unique(tick_positions)

    tick_labels = [f"Layer {int(i)}" for i in tick_positions]

    sns.heatmap(
        cka_matrix,
        annot=False,
        cmap='viridis',
        cbar_kws={'label': 'CKA Similarity', 'shrink': 0.8},
        xticklabels=False,
        yticklabels=False,
        square=True,
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0)
    ax.invert_yaxis()

    ax.set_xlabel(f'Layers ({model_name})')
    ax.set_ylabel(f'Layers ({model_name})')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_difference_plot(pretrained_cka, random_cka, output_path):
    """
    Create and save CKA difference heatmap (ESM-initialized - Random).

    Visualizes the difference between ESM-initialized and random-initialized CKA matrices
    using a diverging colormap (red-blue) centered at zero.

    Interpretation:
        - Blue regions: ESM-initialized has lower coordination than random
        - Red regions: ESM-initialized has higher coordination than random
        - White regions: Similar coordination between initializations

    Args:
        pretrained_cka (numpy.ndarray): ESM-initialized CKA matrix (n_layers, n_layers)
        random_cka (numpy.ndarray): Random-initialized CKA matrix (n_layers, n_layers)
        output_path (str): Path to save the PDF plot

    Returns:
        numpy.ndarray: Difference matrix (pretrained_cka - random_cka)

    Output:
        - Saves PDF file at output_path with 300 DPI resolution
        - Colormap: RdBu_r (diverging, centered at 0)
        - Symmetric color limits: [-max_abs_diff, +max_abs_diff]
    """
    difference = pretrained_cka - random_cka

    fig, ax = plt.subplots(figsize=(12, 10))

    num_layers = len(difference)
    step = max(1, num_layers // 8)
    tick_positions = np.arange(0, num_layers, step)
    if num_layers - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_layers - 1)
        tick_positions = np.unique(tick_positions)

    tick_labels = [f"Layer {int(i)}" for i in tick_positions]
    max_abs_diff = np.max(np.abs(difference))

    sns.heatmap(
        difference,
        annot=False,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'CKA Difference (ESM-initialised - Random)', 'shrink': 0.8},
        xticklabels=False,
        yticklabels=False,
        square=True,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        ax=ax
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0)
    ax.invert_yaxis()

    ax.set_xlabel('Layers')
    ax.set_ylabel('Layers')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    return difference

def main():
    """
    Main pipeline for CKA difference analysis.

    Pipeline:
        1. Load data and create dataloader
        2. Load ESM-initialized and random-initialized checkpoint models
        3. Compute self-CKA matrix for ESM-initialized model
        4. Compute self-CKA matrix for random-initialized model
        5. Calculate difference matrix
        6. Generate three heatmap visualizations:
           - ESM-initialized self-CKA
           - Random-initialized self-CKA
           - Difference heatmap
        7. Compute and save statistical summary

    Command-line Arguments:
        Required:
            --pretrained_checkpoint: Path to ESM-initialized checkpoint file
            --random_checkpoint: Path to random-initialized checkpoint file
            --fcorpus: Path to corpus file
            --fvocab: Path to vocabulary file

        Optional:
            --d_embedding: Embedding dimension (default: 1280)
            --d_model: Model dimension (default: 256)
            --n_layers: Number of layers (default: 32)
            --n_batches: Batches for CKA computation (default: 8)
            --batch_size: Batch size (default: 32)
            --gpu: GPU device ID (default: 0)
            --pretrained_frozen: Use frozen embeddings for ESM-initialized
            --random_frozen: Use frozen embeddings for random-initialized

    Output Files:
        - pretrained_cka_matrix.npy: ESM-initialized CKA matrix
        - random_cka_matrix.npy: Random-initialized CKA matrix
        - cka_difference_matrix.npy: Difference matrix
        - pretrained_cka_heatmap.pdf: ESM-initialized heatmap
        - random_cka_heatmap.pdf: Random-initialized heatmap
        - cka_difference_heatmap.pdf: Difference heatmap
        - analysis_summary.txt: Statistics (mean, std, min, max, Frobenius norms)
    """
    parser = get_parser()
    args = parser.parse_args()

    run_output_dir = os.path.join(args.output_dir, "cka_difference")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output will be saved to: {run_output_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)

    print("Loading data...")
    args.fdata = None
    (train_tokens, test_tokens,
     specials, domains, domain_tokens,
     n_tokens, padding_idx, mask_idx,
     _, _) = load_data(args)

    print("Preparing dataloader...")
    dataloaders = prepare_dataloaders(
        test_tokens, test_tokens, domain_tokens, mask_idx,
        padding_idx, batch_size=args.batch_size, num_workers=4
    )
    dl_test = dataloaders[0]

    print(f"Loading ESM-initialised model from {args.pretrained_checkpoint}")
    pretrained_model = load_model(args.pretrained_checkpoint, n_tokens, mask_idx, domains, args.pretrained_frozen, args)
    pretrained_model.to(device)

    print(f"Loading random model from {args.random_checkpoint}")
    random_model = load_model(args.random_checkpoint, n_tokens, mask_idx, domains, args.random_frozen, args)
    random_model.to(device)

    print("Computing CKA matrix for ESM-initialised model vs itself...")
    pretrained_cka = compute_cka_matrix_self(pretrained_model, dl_test, device, padding_idx, args.n_batches)

    print("Computing CKA matrix for random model vs itself...")
    random_cka = compute_cka_matrix_self(random_model, dl_test, device, padding_idx, args.n_batches)

    np.save(os.path.join(run_output_dir, 'pretrained_cka_matrix.npy'), pretrained_cka)
    np.save(os.path.join(run_output_dir, 'random_cka_matrix.npy'), random_cka)

    create_heatmap_plot(
        pretrained_cka,
        os.path.join(run_output_dir, 'pretrained_cka_heatmap.pdf'),
        "ESM-initialised"
    )

    create_heatmap_plot(
        random_cka,
        os.path.join(run_output_dir, 'random_cka_heatmap.pdf'),
        "Random"
    )

    difference_matrix = create_difference_plot(
        pretrained_cka, random_cka,
        os.path.join(run_output_dir, 'cka_difference_heatmap.pdf')
    )

    np.save(os.path.join(run_output_dir, 'cka_difference_matrix.npy'), difference_matrix)

    stats = {
        'difference_mean': float(np.mean(difference_matrix)),
        'difference_std': float(np.std(difference_matrix)),
        'difference_max': float(np.max(difference_matrix)),
        'difference_min': float(np.min(difference_matrix)),
        'difference_abs_mean': float(np.mean(np.abs(difference_matrix))),
        'frobenius_norm_pretrained': float(np.linalg.norm(pretrained_cka, 'fro')),
        'frobenius_norm_random': float(np.linalg.norm(random_cka, 'fro')),
        'frobenius_norm_difference': float(np.linalg.norm(difference_matrix, 'fro'))
    }

    with open(os.path.join(run_output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("CKA Difference Analysis Results\n")
        f.write("===============================\n\n")
        f.write(f"ESM-initialised checkpoint: {args.pretrained_checkpoint}\n")
        f.write(f"Random checkpoint: {args.random_checkpoint}\n\n")

        f.write("Difference Statistics (ESM-initialised - Random):\n")
        f.write(f"Mean difference: {stats['difference_mean']:.6f}\n")
        f.write(f"Std difference: {stats['difference_std']:.6f}\n")
        f.write(f"Max difference: {stats['difference_max']:.6f}\n")
        f.write(f"Min difference: {stats['difference_min']:.6f}\n")
        f.write(f"Mean absolute difference: {stats['difference_abs_mean']:.6f}\n\n")

        f.write("Frobenius Norms:\n")
        f.write(f"ESM-initialised CKA matrix: {stats['frobenius_norm_pretrained']:.6f}\n")
        f.write(f"Random CKA matrix: {stats['frobenius_norm_random']:.6f}\n")
        f.write(f"Difference matrix: {stats['frobenius_norm_difference']:.6f}\n")

    del pretrained_model
    del random_model
    torch.cuda.empty_cache()

    print(f"\nAnalysis complete. Results saved to {run_output_dir}")

if __name__ == "__main__":
    main()
