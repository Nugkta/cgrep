"""
CKA Evolution Analysis for BigCARP Models

This script tracks how different layers evolve during training by computing CKA similarity
between reference models at checkpoint 0 and layers at different checkpoints:
1. ESM-initialised embedder (ckp 0) vs ESM-initialised embedder at different checkpoints
2. ESM-initialised embedder (ckp 0) vs ESM-initialised last layer at different checkpoints
3. ESM-initialised embedder (ckp 0) vs random last layer at different checkpoints
4. Random embedder (ckp 0) vs random last layer at different checkpoints

The script generates a plot with four curves showing how similarity changes across training.

Usage:
    python scripts/cka/cka_evolution.py \
        --pretrained_dir path/to/esm_initialised/checkpoints \
        --random_dir path/to/random/checkpoints \
        --fcorpus data/corpus.csv \
        --fvocab data/vocab.json \
        --d_embedding 1280 \
        --d_model 256 \
        --n_layers 32 \
        --output_dir results/cka_evolution

        conda activate cgrep && srun --gpus=1 --time=01:00:00 python scripts/cka/cka_evolution.py \
        --pretrained_dir artifacts/bigcarp/bigcarp_models/run_esm_init \
        --random_dir artifacts/bigcarp/bigcarp_models/run_random_init \
        --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
        --fvocab data/processed/vocabularies/pfam_vocab_present.json \
        --d_embedding 1280 \
        --d_model 256 \
        --n_layers 32 \
        --kernel_size 3 \
        --r 128 \
        --batch_size 64 \
        --gpu 0 \
        --output_dir results/cka_evolution \
        --max_checkpoints 99 \
        --n_batches 16 \
        --unconditional
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm 
from datetime import datetime
import glob
import re

from sequence_models.convolutional import ByteNetLM
from sequence_models.collaters import _pad

# Import shared BigCARP functions from the cgrep package
from cgrep.bigcarp_functions import load_data, ListDataset, mlm_collate_fn, prepare_dataloaders

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
    parser = argparse.ArgumentParser(description='CKA evolution analysis of BIGCARP models')
    
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
    Compute linear CKA between feature matrices X and Y
    
    Args:
        X: Feature matrix of shape [n_samples, n_features_1]
        Y: Feature matrix of shape [n_samples, n_features_2]
        
    Returns:
        float: CKA value
    """
    # Center features
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    
    # Calculate Gram matrices
    XXT = torch.matmul(X, X.T)
    YYT = torch.matmul(Y, Y.T)
    
    # Calculate HSIC
    n = X.size(0)
    # Optional: add small epsilon for numerical stability
    epsilon = 1e-10
    
    # Simplified calculation with better numerical stability
    HSIC = torch.sum(XXT * YYT)
    var1 = torch.sqrt(torch.sum(XXT * XXT) + epsilon)
    var2 = torch.sqrt(torch.sum(YYT * YYT) + epsilon)
    
    return (HSIC / (var1 * var2)).item()

def load_model(checkpoint_path, n_tokens, mask_idx, domains, is_frozen, args):
    """Load a ByteNetLM model from checkpoint"""
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
    """Extract epoch number from checkpoint filename"""
    # Handle different checkpoint naming patterns
    patterns = [
        r'checkpoint_epoch(\d+)\.tar',
        r'checkpoint(\d+)\.tar',
        r'epoch(\d+)\.tar'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    # If no pattern matches, try to extract any number
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Take the last number found
    
    return None

def get_checkpoint_files(checkpoint_dir, max_checkpoints=None):
    """Get list of checkpoint files sorted by epoch number"""
    # Look for checkpoint files
    checkpoint_patterns = ['checkpoint*.tar', 'checkpoint_epoch*.tar']
    all_checkpoints = []
    
    for pattern in checkpoint_patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        all_checkpoints.extend(files)
    
    # Extract epochs and sort
    checkpoint_info = []
    for checkpoint_path in all_checkpoints:
        filename = os.path.basename(checkpoint_path)
        epoch = extract_epoch_from_filename(filename)
        if epoch is not None:
            checkpoint_info.append((epoch, checkpoint_path))
    
    # Sort by epoch and limit if requested
    checkpoint_info.sort(key=lambda x: x[0])
    if max_checkpoints:
        checkpoint_info = checkpoint_info[:max_checkpoints]
    
    return checkpoint_info

def compute_layer_cka(model1, layer1_idx, model2, layer2_idx, dataloader, device, n_batches=8):
    """Compute CKA between specific layers of two models"""
    model1.eval()
    model2.eval()
    
    all_features1 = []
    all_features2 = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
                
            src, _, _ = [b.to(device) for b in batch]
            input_mask = (src != 0).float().unsqueeze(-1)
            mask_flat = input_mask.squeeze(-1).reshape(-1).bool()
            
            # Get representations from model1
            _, h1 = model1.embedder(src, input_mask=input_mask, return_all_hidden_states=True)
            features1 = h1[layer1_idx].reshape(-1, h1[layer1_idx].size(-1))[mask_flat]
            
            # Get representations from model2  
            _, h2 = model2.embedder(src, input_mask=input_mask, return_all_hidden_states=True)
            features2 = h2[layer2_idx].reshape(-1, h2[layer2_idx].size(-1))[mask_flat]
            
            all_features1.append(features1.cpu())
            all_features2.append(features2.cpu())
    
    # Concatenate all features
    all_features1 = torch.cat(all_features1, dim=0)
    all_features2 = torch.cat(all_features2, dim=0)
    
    # Compute CKA
    cka = linear_CKA(all_features1, all_features2)
    return cka

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"cka_evolution_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output will be saved to: {run_output_dir}")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    
    # Load vocabulary and data
    print("Loading data...")
    args.fdata = None  # Not needed for this script
    (train_tokens, test_tokens,
     specials, domains, domain_tokens,
     n_tokens, padding_idx, mask_idx,
     _, _) = load_data(args)
    
    # Prepare DataLoader
    print("Preparing dataloader...")
    dataloaders = prepare_dataloaders(
        test_tokens, test_tokens, domain_tokens, mask_idx,
        padding_idx, batch_size=args.batch_size, num_workers=4
    )
    dl_test = dataloaders[0]
    
    # Get checkpoint files for both pretrained and random models
    print("Finding checkpoint files...")
    pretrained_checkpoints = get_checkpoint_files(args.pretrained_dir, args.max_checkpoints)
    random_checkpoints = get_checkpoint_files(args.random_dir, args.max_checkpoints)
    
    print(f"Found {len(pretrained_checkpoints)} pretrained checkpoints")
    print(f"Found {len(random_checkpoints)} random checkpoints")
    
    if not pretrained_checkpoints:
        raise ValueError(f"No checkpoints found in {args.pretrained_dir}")
    if not random_checkpoints:
        raise ValueError(f"No checkpoints found in {args.random_dir}")
    
    # Load checkpoint 0 (reference model) - should be the first pretrained checkpoint
    ref_epoch, ref_checkpoint_path = pretrained_checkpoints[0]
    print(f"Loading pretrained reference model from epoch {ref_epoch}: {ref_checkpoint_path}")
    ref_model = load_model(ref_checkpoint_path, n_tokens, mask_idx, domains, args.pretrained_frozen, args)
    ref_model.to(device)
    
    # Load checkpoint 0 for random model as well
    ref_random_epoch, ref_random_checkpoint_path = random_checkpoints[0]
    print(f"Loading random reference model from epoch {ref_random_epoch}: {ref_random_checkpoint_path}")
    ref_random_model = load_model(ref_random_checkpoint_path, n_tokens, mask_idx, domains, args.random_frozen, args)
    ref_random_model.to(device)
    
    # Initialize results storage
    epochs = []
    pretrained_embedder_similarities = []  # Curve 1: ref embedder vs pretrained embedder
    pretrained_last_similarities = []     # Curve 2: ref embedder vs pretrained last layer
    random_last_similarities = []         # Curve 3: ref embedder vs random last layer
    random_embedder_random_last_similarities = []  # Curve 4: ref random embedder vs random last layer
    
    # Compute similarities for pretrained models
    print("Computing similarities for pretrained models...")
    for epoch, checkpoint_path in tqdm(pretrained_checkpoints, desc="Pretrained checkpoints"):
        print(f"Processing pretrained checkpoint epoch {epoch}")
        
        # Load current checkpoint
        current_model = load_model(checkpoint_path, n_tokens, mask_idx, domains, args.pretrained_frozen, args)
        current_model.to(device)
        
        # Compute CKA between ref embedder (layer 0) and current embedder (layer 0)
        cka_embedder = compute_layer_cka(ref_model, 0, current_model, 0, dl_test, device, args.n_batches)
        
        # Compute CKA between ref embedder (layer 0) and current last layer
        last_layer_idx = args.n_layers - 1
        cka_last = compute_layer_cka(ref_model, 0, current_model, last_layer_idx, dl_test, device, args.n_batches)
        
        epochs.append(epoch)
        pretrained_embedder_similarities.append(cka_embedder)
        pretrained_last_similarities.append(cka_last)
        
        # Clean up memory
        del current_model
        torch.cuda.empty_cache()
    
    # Compute similarities for random models
    print("Computing similarities for random models...")
    random_epochs = []
    for epoch, checkpoint_path in tqdm(random_checkpoints, desc="Random checkpoints"):
        print(f"Processing random checkpoint epoch {epoch}")
        
        # Load current random checkpoint
        current_model = load_model(checkpoint_path, n_tokens, mask_idx, domains, args.random_frozen, args)
        current_model.to(device)
        
        # Compute CKA between ref embedder (layer 0) and current random last layer
        last_layer_idx = args.n_layers - 1
        cka_random_last = compute_layer_cka(ref_model, 0, current_model, last_layer_idx, dl_test, device, args.n_batches)
        
        # Compute CKA between ref random embedder (layer 0) and current random last layer
        cka_random_embedder_random_last = compute_layer_cka(ref_random_model, 0, current_model, last_layer_idx, dl_test, device, args.n_batches)
        
        random_epochs.append(epoch)
        random_last_similarities.append(cka_random_last)
        random_embedder_random_last_similarities.append(cka_random_embedder_random_last)
        
        # Clean up memory
        del current_model
        torch.cuda.empty_cache()
    
    # Save results
    results = {
        'epochs': epochs,
        'pretrained_embedder_similarities': pretrained_embedder_similarities,
        'pretrained_last_similarities': pretrained_last_similarities,
        'random_epochs': random_epochs, 
        'random_last_similarities': random_last_similarities,
        'random_embedder_random_last_similarities': random_embedder_random_last_similarities
    }
    np.save(os.path.join(run_output_dir, 'cka_evolution_results.npy'), results)
    
    # Create the evolution plot with publication-quality styling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Enhanced colors and styling for publication quality
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#8E44AD']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '-', '-', '-']

    # Plot the four curves with enhanced styling
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

    ax.set_xlabel('Training Epoch', fontweight='semibold')
    ax.set_ylabel('CKA Similarity', fontweight='semibold')
    ax.set_title('Evolution of Layer Similarities During Training\n(References: ESM-initialised & Random Embedders at Checkpoint 0)',
                 fontweight='bold', pad=20)

    # Enhanced legend styling
    legend = ax.legend(fontsize=11, loc='best', frameon=True, fancybox=False, shadow=False,
                      edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(1)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, 'cka_evolution_plot.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Also save the command line arguments for reproducibility
    with open(os.path.join(run_output_dir, 'run_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Save a summary of results
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
    
    # Clean up reference models
    del ref_model
    del ref_random_model
    torch.cuda.empty_cache()
    
    print(f"\nAnalysis complete. Results saved to {run_output_dir}")
    print("Generated files:")
    print(f"  - cka_evolution_plot.png: Main visualization with four curves")
    print(f"  - cka_evolution_results.npy: Raw results data")
    print(f"  - results_summary.txt: Summary of findings")
    print(f"  - run_config.txt: Configuration used for this run")

if __name__ == "__main__":
    main()
