"""
CKA Evolution Analysis for bigcarp Models
==========================================

This script analyzes how neural network representations evolve during training by
computing Centered Kernel Alignment (CKA) similarity between initial and later
training checkpoints.

Analysis Strategy:
    Compares representations from checkpoint 0 (reference) against subsequent checkpoints:
    - ESM-initialized embedder vs ESM-initialized embedder (layer stability)
    - ESM-initialized embedder vs ESM-initialized last layer (representation evolution)
    - ESM-initialized embedder vs Random-initialized last layer (initialization effects)
    - Random embedder vs Random last layer (baseline evolution)

Model Types:
    - ESM-initialized: bigcarp with pretrained ESM embeddings initialized (optionally frozen)
    - Random-initialized: bigcarp with random initialization

Metrics:
    - Linear CKA similarity (0-1 scale, 1 = identical representations)
    - Tracked across training epochs for embedder and last layer

Usage:
    python cka_evolution.py \\
        --pretrained_dir checkpoints/esm_init \\
        --random_dir checkpoints/random_init \\
        --fcorpus data/corpus.txt \\
        --fvocab data/vocab.txt \\
        --n_layers 32 \\
        --max_checkpoints 20 \\
        --n_batches 8

Output Files:
    - cka_evolution_results.npy: Dictionary with CKA scores across epochs
    - cka_evolution_plot.pdf: Line plot showing similarity evolution
    - results_summary.txt: Summary of final similarities
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import re

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
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.grid': False,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white'
})

def get_parser():
    """
    Create argument parser for CKA evolution analysis.

    Returns:
        argparse.ArgumentParser: Configured parser with all required and optional arguments
    """
    parser = argparse.ArgumentParser(description='CKA evolution analysis of bigcarp models')
    
    # Paths to checkpoint directories
    parser.add_argument('--pretrained_dir', type=str, required=True,
                        help='Path to directory containing ESM-initialised model checkpoints')
    parser.add_argument('--random_dir', type=str, required=True,
                        help='Path to directory containing random initialized model checkpoints')
    
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
    parser.add_argument('--output_dir', type=str, default='results/bigcarp/cka_evolution',
                        help='Directory to save CKA evolution analysis results')
    
    # Replace single frozen embeddings flag with model-specific flags
    parser.add_argument('--pretrained_frozen', action='store_true',
                        help='If set, load pretrained models with frozen embeddings')
    parser.add_argument('--random_frozen', action='store_true',
                        help='If set, load random models with frozen embeddings')
    
    # Checkpoint selection parameters
    parser.add_argument('--max_checkpoints', type=int, default=20,
                        help='Maximum number of checkpoints to analyze')
    parser.add_argument('--n_batches', type=int, default=8,
                        help='Number of batches to use for CKA computation')
    
    return parser

def linear_CKA(X, Y):
    """
    Compute linear Centered Kernel Alignment (CKA) between two feature matrices.

    Linear CKA measures the similarity between two representations by comparing their
    Gram matrices. Range: [0, 1], where 1 indicates identical representations.

    Args:
        X (torch.Tensor): Feature matrix of shape (n_samples, d1)
        Y (torch.Tensor): Feature matrix of shape (n_samples, d2)

    Returns:
        float: CKA similarity score in range [0, 1]
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    XXT = torch.matmul(X, X.T)
    YYT = torch.matmul(Y, Y.T)

    epsilon = 1e-10
    HSIC = torch.sum(XXT * YYT)
    var1 = torch.sqrt(torch.sum(XXT * XXT) + epsilon)
    var2 = torch.sqrt(torch.sum(YYT * YYT) + epsilon)

    return (HSIC / (var1 * var2)).item()

def load_model(checkpoint_path, n_tokens, mask_idx, domains, is_frozen, args):
    """
    Load a bigcarp model from checkpoint file.

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

def extract_epoch_from_filename(filename):
    """
    Extract epoch number from checkpoint filename.

    Tries multiple common checkpoint naming patterns:
    - checkpoint_epoch{N}.tar
    - checkpoint{N}.tar
    - epoch{N}.tar
    Falls back to extracting last number from filename if no pattern matches.

    Args:
        filename (str): Checkpoint filename (e.g., "checkpoint_epoch10.tar")

    Returns:
        int or None: Extracted epoch number, or None if no number found
    """
    patterns = [
        r'checkpoint_epoch(\d+)\.tar',
        r'checkpoint(\d+)\.tar',
        r'epoch(\d+)\.tar'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))

    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])

    return None

def get_checkpoint_files(checkpoint_dir, max_checkpoints=None):
    """
    Get list of checkpoint files from directory, sorted by epoch number.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        max_checkpoints (int, optional): Maximum number of checkpoints to return.
            If specified, returns only the first N checkpoints by epoch.

    Returns:
        list of tuples: List of (epoch, checkpoint_path) tuples sorted by epoch number.
            Example: [(0, "path/checkpoint_epoch0.tar"), (1, "path/checkpoint_epoch1.tar"), ...]
    """
    checkpoint_patterns = ['checkpoint*.tar', 'checkpoint_epoch*.tar']
    all_checkpoints = []

    for pattern in checkpoint_patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        all_checkpoints.extend(files)

    checkpoint_info = []
    for checkpoint_path in all_checkpoints:
        filename = os.path.basename(checkpoint_path)
        epoch = extract_epoch_from_filename(filename)
        if epoch is not None:
            checkpoint_info.append((epoch, checkpoint_path))

    checkpoint_info.sort(key=lambda x: x[0])
    if max_checkpoints:
        checkpoint_info = checkpoint_info[:max_checkpoints]

    return checkpoint_info

def compute_layer_cka(model1, layer1_idx, model2, layer2_idx, dataloader, device, padding_idx, n_batches=8):
    """
    Compute CKA similarity between specific layers of two models.

    Extracts hidden states from specified layers on the same input data,
    then computes linear CKA between the representations.

    Args:
        model1 (ByteNetLM): First model
        layer1_idx (int): Layer index in model1 (0 = embedder, n_layers-1 = last layer)
        model2 (ByteNetLM): Second model
        layer2_idx (int): Layer index in model2
        dataloader (DataLoader): Data loader for evaluation samples
        device (torch.device): Device for computation (CPU or CUDA)
        padding_idx (int): Token ID used for padding (positions to exclude from CKA)
        n_batches (int): Number of batches to use for CKA computation (default: 8)

    Returns:
        float: CKA similarity score between the two layers in range [0, 1]
    """
    model1.eval()
    model2.eval()

    all_features1 = []
    all_features2 = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            src, _, _ = [b.to(device) for b in batch]
            input_mask = (src != padding_idx).float().unsqueeze(-1)
            mask_flat = input_mask.squeeze(-1).reshape(-1).bool()

            _, h1 = model1.embedder(src, input_mask=input_mask, return_all_hidden_states=True)
            features1 = h1[layer1_idx].reshape(-1, h1[layer1_idx].size(-1))[mask_flat]

            _, h2 = model2.embedder(src, input_mask=input_mask, return_all_hidden_states=True)
            features2 = h2[layer2_idx].reshape(-1, h2[layer2_idx].size(-1))[mask_flat]

            all_features1.append(features1.cpu())
            all_features2.append(features2.cpu())

    all_features1 = torch.cat(all_features1, dim=0)
    all_features2 = torch.cat(all_features2, dim=0)

    cka = linear_CKA(all_features1, all_features2)
    return cka

def main():
    """
    Main pipeline for CKA evolution analysis.

    Pipeline:
        1. Load data and create dataloader
        2. Find and sort checkpoint files for both model types
        3. Load reference models (epoch 0) for both ESM-initialized and random-initialized
        4. For each ESM-initialized checkpoint:
           - Compute CKA: ref_embedder vs current_embedder
           - Compute CKA: ref_embedder vs current_last_layer
        5. For each random-initialized checkpoint:
           - Compute CKA: ref_esm_embedder vs random_last_layer
           - Compute CKA: ref_random_embedder vs random_last_layer
        6. Save results and generate visualization plot

    Command-line Arguments:
        Required:
            --pretrained_dir: Directory with ESM-initialized checkpoints
            --random_dir: Directory with random-initialized checkpoints
            --fcorpus: Path to corpus file
            --fvocab: Path to vocabulary file

        Optional:
            --d_embedding: Embedding dimension (default: 1280)
            --d_model: Model dimension (default: 256)
            --n_layers: Number of layers (default: 32)
            --max_checkpoints: Max checkpoints to analyze (default: 20)
            --n_batches: Batches for CKA computation (default: 8)
            --batch_size: Batch size (default: 32)
            --gpu: GPU device ID (default: 0)

    Output Files:
        - cka_evolution_results.npy: Dict with epochs and CKA scores
        - cka_evolution_plot.pdf: Visualization of CKA evolution
        - results_summary.txt: Final similarity values
    """
    parser = get_parser()
    args = parser.parse_args()

    run_output_dir = os.path.join(args.output_dir, "cka_evolution")
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

    print("Finding checkpoint files...")
    pretrained_checkpoints = get_checkpoint_files(args.pretrained_dir, args.max_checkpoints)
    random_checkpoints = get_checkpoint_files(args.random_dir, args.max_checkpoints)

    print(f"Found {len(pretrained_checkpoints)} pretrained checkpoints")
    print(f"Found {len(random_checkpoints)} random checkpoints")

    if not pretrained_checkpoints:
        raise ValueError(f"No checkpoints found in {args.pretrained_dir}")
    if not random_checkpoints:
        raise ValueError(f"No checkpoints found in {args.random_dir}")

    ref_epoch, ref_checkpoint_path = pretrained_checkpoints[0]
    print(f"Loading pretrained reference model from epoch {ref_epoch}: {ref_checkpoint_path}")
    ref_model = load_model(ref_checkpoint_path, n_tokens, mask_idx, domains, args.pretrained_frozen, args)
    ref_model.to(device)

    ref_random_epoch, ref_random_checkpoint_path = random_checkpoints[0]
    print(f"Loading random reference model from epoch {ref_random_epoch}: {ref_random_checkpoint_path}")
    ref_random_model = load_model(ref_random_checkpoint_path, n_tokens, mask_idx, domains, args.random_frozen, args)
    ref_random_model.to(device)
    
    epochs = []
    pretrained_embedder_similarities = []
    pretrained_last_similarities = []
    random_last_similarities = []
    random_embedder_random_last_similarities = []

    print("Computing similarities for pretrained models...")
    for epoch, checkpoint_path in tqdm(pretrained_checkpoints, desc="Pretrained checkpoints"):
        print(f"Processing pretrained checkpoint epoch {epoch}")

        current_model = load_model(checkpoint_path, n_tokens, mask_idx, domains, args.pretrained_frozen, args)
        current_model.to(device)

        cka_embedder = compute_layer_cka(ref_model, 0, current_model, 0, dl_test, device, padding_idx, args.n_batches)

        last_layer_idx = args.n_layers - 1
        cka_last = compute_layer_cka(ref_model, 0, current_model, last_layer_idx, dl_test, device, padding_idx, args.n_batches)

        epochs.append(epoch)
        pretrained_embedder_similarities.append(cka_embedder)
        pretrained_last_similarities.append(cka_last)

        del current_model
        torch.cuda.empty_cache()

    print("Computing similarities for random models...")
    random_epochs = []
    for epoch, checkpoint_path in tqdm(random_checkpoints, desc="Random checkpoints"):
        print(f"Processing random checkpoint epoch {epoch}")

        current_model = load_model(checkpoint_path, n_tokens, mask_idx, domains, args.random_frozen, args)
        current_model.to(device)

        last_layer_idx = args.n_layers - 1
        cka_random_last = compute_layer_cka(ref_model, 0, current_model, last_layer_idx, dl_test, device, args.n_batches)

        cka_random_embedder_random_last = compute_layer_cka(ref_random_model, 0, current_model, last_layer_idx, dl_test, device, args.n_batches)

        random_epochs.append(epoch)
        random_last_similarities.append(cka_random_last)
        random_embedder_random_last_similarities.append(cka_random_embedder_random_last)

        del current_model
        torch.cuda.empty_cache()
    
    results = {
        'epochs': epochs,
        'pretrained_embedder_similarities': pretrained_embedder_similarities,
        'pretrained_last_similarities': pretrained_last_similarities,
        'random_epochs': random_epochs,
        'random_last_similarities': random_last_similarities,
        'random_embedder_random_last_similarities': random_embedder_random_last_similarities
    }
    np.save(os.path.join(run_output_dir, 'cka_evolution_results.npy'), results)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#2E86C1', '#E74C3C', '#28B463', '#8E44AD']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '-', '-', '-']

    ax.plot(epochs, pretrained_embedder_similarities, color=colors[0], marker=markers[0],
            linestyle=linestyles[0], label='ESM-initialised Embedder vs ESM-initialised Embedder',
            linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=colors[0])
    ax.plot(epochs, pretrained_last_similarities, color=colors[1], marker=markers[1],
            linestyle=linestyles[1], label='ESM-initialised Embedder vs ESM-initialised Last Layer',
            linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=colors[1])
    ax.plot(random_epochs, random_last_similarities, color=colors[2], marker=markers[2],
            linestyle=linestyles[2], label='ESM-initialised Embedder vs Random Last Layer',
            linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=colors[2])
    ax.plot(random_epochs, random_embedder_random_last_similarities, color=colors[3], marker=markers[3],
            linestyle=linestyles[3], label='Random Embedder vs Random Last Layer',
            linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=colors[3])

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('CKA Similarity')

    legend = ax.legend(fontsize=16, loc='best', frameon=True, fancybox=False, shadow=False,
                      edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(1)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, 'cka_evolution_plot.pdf'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    with open(os.path.join(run_output_dir, 'results_summary.txt'), 'w') as f:
        f.write("CKA Evolution Analysis Results\n")
        f.write("==============================\n\n")
        f.write(f"Reference model: Epoch {ref_epoch}\n")
        f.write(f"Number of ESM-initialised checkpoints analyzed: {len(epochs)}\n")
        f.write(f"Number of random checkpoints analyzed: {len(random_epochs)}\n\n")

        f.write("Final similarities:\n")
        if pretrained_embedder_similarities:
            f.write(f"ESM-initialised Embedder vs ESM-initialised Embedder (final): {pretrained_embedder_similarities[-1]:.4f}\n")
        if pretrained_last_similarities:
            f.write(f"ESM-initialised Embedder vs ESM-initialised Last Layer (final): {pretrained_last_similarities[-1]:.4f}\n")
        if random_last_similarities:
            f.write(f"ESM-initialised Embedder vs Random Last Layer (final): {random_last_similarities[-1]:.4f}\n")
        if random_embedder_random_last_similarities:
            f.write(f"Random Embedder vs Random Last Layer (final): {random_embedder_random_last_similarities[-1]:.4f}\n")

    del ref_model
    del ref_random_model
    torch.cuda.empty_cache()

    print(f"\nAnalysis complete. Results saved to {run_output_dir}")

if __name__ == "__main__":
    main()
