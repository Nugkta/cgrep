"""
CKA Difference Heatmap Analysis for BigCARP Models

This script compares the internal layer coordination between ESM-initialised and random models
at a specific checkpoint by:
1. Computing CKA matrix for ESM-initialised model vs itself at checkpoint N
2. Computing CKA matrix for random model vs itself at checkpoint N
3. Creating a difference heatmap to show convergence of embedding space coordination

The difference plot helps visualize whether the inner coordination between different layers
converges to be similar between ESM-initialised and random initialized models.

Usage:
    python scripts/cka/cka_difference_heatmap.py \
        --pretrained_checkpoint path/to/esm_initialised/checkpoint50.tar \
        --random_checkpoint path/to/random/checkpoint50.tar \
        --fcorpus data/corpus.csv \
        --fvocab data/vocab.json \
        --d_embedding 1280 \
        --d_model 256 \
        --n_layers 32 \
        --output_dir results/cka_difference

    conda activate cgrep && srun --gpus=1 --time=01:00:00 python scripts/cka/cka_difference_heatmap.py \
    --pretrained_checkpoint artifacts/bigcarp/bigcarp_models/run_esm_init/checkpoint_latest.tar \
    --random_checkpoint artifacts/bigcarp/bigcarp_models/run_random_init/checkpoint_best.tar \
    --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \
    --fvocab data/processed/vocabularies/pfam_vocab_present.json \
    --d_embedding 1280 \
    --d_model 256 \
    --n_layers 32 \
    --kernel_size 3 \
    --r 128 \
    --batch_size 128 \
    --gpu 0 \
    --output_dir results/cka \
    --n_batches 32 \
    --unconditional
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

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
    parser = argparse.ArgumentParser(description='CKA difference heatmap analysis of BIGCARP models')
    
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

def compute_cka_matrix_self(model, dataloader, device, n_batches=8):
    """
    Compute CKA matrix for a model compared to itself (all layers vs all layers)
    """
    model.eval()

    hsic_sum = var1_sum = var2_sum = None

    for _ in range(n_batches):
        src, _, _ = [b.to(device) for b in next(iter(dataloader))]
        input_mask = (src != 0).float().unsqueeze(-1)
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

def extract_checkpoint_info(checkpoint_path):
    """Extract checkpoint number and model type from path"""
    filename = os.path.basename(checkpoint_path)
    
    # Try to extract checkpoint number
    import re
    patterns = [
        r'checkpoint_epoch(\d+)\.tar',
        r'checkpoint(\d+)\.tar',
        r'epoch(\d+)\.tar'
    ]
    
    checkpoint_num = None
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            checkpoint_num = int(match.group(1))
            break
    
    # Try to determine model type from path
    model_type = "unknown"
    if "pretrain" in checkpoint_path.lower() or "_pt_" in checkpoint_path.lower():
        model_type = "pretrained"
    elif "random" in checkpoint_path.lower() or "_rd_" in checkpoint_path.lower():
        model_type = "random"
    
    return checkpoint_num, model_type

def create_heatmap_plot(cka_matrix, title, output_path, model_name=""):
    """Create and save a single CKA heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))

    num_layers = len(cka_matrix)
    step = max(1, num_layers // 8)  # Show about 8 ticks for readability
    tick_positions = np.arange(0, num_layers, step)
    if num_layers - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_layers - 1)
        tick_positions = np.unique(tick_positions)

    tick_labels = [f"Layer {int(i)}" for i in tick_positions]

    # Create heatmap
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

    # Set explicit tick positions and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0)

    # Invert Y axis so Layer 0 is at the bottom
    ax.invert_yaxis()

    ax.set_xlabel(f'Layers ({model_name})')
    ax.set_ylabel(f'Layers ({model_name})')
    # ax.set_title(title, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_difference_plot(pretrained_cka, random_cka, output_path, checkpoint_info=""):
    """Create and save the difference heatmap"""
    # Compute difference (pretrained - random)
    difference = pretrained_cka - random_cka

    fig, ax = plt.subplots(figsize=(12, 10))

    num_layers = len(difference)
    step = max(1, num_layers // 8)
    tick_positions = np.arange(0, num_layers, step)
    if num_layers - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_layers - 1)
        tick_positions = np.unique(tick_positions)

    tick_labels = [f"Layer {int(i)}" for i in tick_positions]

    # Use a diverging colormap centered at 0
    max_abs_diff = np.max(np.abs(difference))

    sns.heatmap(
        difference,
        annot=False,
        cmap='RdBu_r',  # Red-Blue diverging colormap
        center=0,
        cbar_kws={'label': 'CKA Difference (ESM-initialised - Random)', 'shrink': 0.8},
        xticklabels=False,
        yticklabels=False,
        square=True,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        ax=ax
    )

    # Set explicit tick positions and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0)

    # Invert Y axis so Layer 0 is at the bottom
    ax.invert_yaxis()

    ax.set_xlabel('Layers')
    ax.set_ylabel('Layers')
    # ax.set_title(f'CKA Difference: ESM-initialised vs Random \n{checkpoint_info}',
    #              fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    return difference

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create timestamped directory for this run
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_output_dir = os.path.join(args.output_dir, f"cka_difference_{timestamp}")
    run_output_dir = os.path.join(args.output_dir, f"cka_difference")
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
    
    # Extract checkpoint information
    pretrained_ckp_num, pretrained_type = extract_checkpoint_info(args.pretrained_checkpoint)
    random_ckp_num, random_type = extract_checkpoint_info(args.random_checkpoint)
    
    print(f"Analyzing ESM-initialised checkpoint: {pretrained_ckp_num} ({pretrained_type})")
    print(f"Analyzing random checkpoint: {random_ckp_num} ({random_type})")

    # Load ESM-initialised model
    print(f"Loading ESM-initialised model from {args.pretrained_checkpoint}")
    pretrained_model = load_model(args.pretrained_checkpoint, n_tokens, mask_idx, domains, args.pretrained_frozen, args)
    pretrained_model.to(device)

    # Load random model
    print(f"Loading random model from {args.random_checkpoint}")
    random_model = load_model(args.random_checkpoint, n_tokens, mask_idx, domains, args.random_frozen, args)
    random_model.to(device)

    # Compute CKA matrices
    print("Computing CKA matrix for ESM-initialised model vs itself...")
    pretrained_cka = compute_cka_matrix_self(pretrained_model, dl_test, device, args.n_batches)

    print("Computing CKA matrix for random model vs itself...")
    random_cka = compute_cka_matrix_self(random_model, dl_test, device, args.n_batches)
    
    # Save raw CKA matrices
    np.save(os.path.join(run_output_dir, 'pretrained_cka_matrix.npy'), pretrained_cka)
    np.save(os.path.join(run_output_dir, 'random_cka_matrix.npy'), random_cka)
    
    # Create individual heatmaps
    pretrained_name = f"ESM-initialised (ckp {pretrained_ckp_num})" if pretrained_ckp_num else "ESM-initialised"
    random_name = f"Random (ckp {random_ckp_num})" if random_ckp_num else "Random"

    create_heatmap_plot(
        pretrained_cka,
        f'CKA Self-Similarity: {pretrained_name}',
        os.path.join(run_output_dir, 'pretrained_cka_heatmap.pdf'),
        pretrained_name
    )
    
    create_heatmap_plot(
        random_cka,
        f'CKA Self-Similarity: {random_name}',
        os.path.join(run_output_dir, 'random_cka_heatmap.pdf'),
        random_name
    )
    
    # Create difference plot
    checkpoint_info = ""
    if pretrained_ckp_num and random_ckp_num:
        checkpoint_info = f"(Checkpoints: ESM-initialised {pretrained_ckp_num}, Random {random_ckp_num})"
    
    difference_matrix = create_difference_plot(
        pretrained_cka, random_cka,
        os.path.join(run_output_dir, 'cka_difference_heatmap.pdf'),
        checkpoint_info
    )
    
    # Save difference matrix
    np.save(os.path.join(run_output_dir, 'cka_difference_matrix.npy'), difference_matrix)
    
    # Calculate and save statistics
    stats = {
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'random_checkpoint': args.random_checkpoint,
        'pretrained_ckp_num': pretrained_ckp_num,
        'random_ckp_num': random_ckp_num,
        'difference_mean': float(np.mean(difference_matrix)),
        'difference_std': float(np.std(difference_matrix)),
        'difference_max': float(np.max(difference_matrix)),
        'difference_min': float(np.min(difference_matrix)),
        'difference_abs_mean': float(np.mean(np.abs(difference_matrix))),
        'frobenius_norm_pretrained': float(np.linalg.norm(pretrained_cka, 'fro')),
        'frobenius_norm_random': float(np.linalg.norm(random_cka, 'fro')),
        'frobenius_norm_difference': float(np.linalg.norm(difference_matrix, 'fro'))
    }
    
    # Save configuration and results
    with open(os.path.join(run_output_dir, 'run_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    with open(os.path.join(run_output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("CKA Difference Analysis Results\n")
        f.write("===============================\n\n")
        f.write(f"ESM-initialised checkpoint: {args.pretrained_checkpoint}\n")
        f.write(f"Random checkpoint: {args.random_checkpoint}\n")
        f.write(f"ESM-initialised checkpoint number: {pretrained_ckp_num}\n")
        f.write(f"Random checkpoint number: {random_ckp_num}\n\n")

        f.write("Difference Statistics (ESM-initialised - Random):\n")
        f.write(f"Mean difference: {stats['difference_mean']:.6f}\n")
        f.write(f"Std difference: {stats['difference_std']:.6f}\n")
        f.write(f"Max difference: {stats['difference_max']:.6f}\n")
        f.write(f"Min difference: {stats['difference_min']:.6f}\n")
        f.write(f"Mean absolute difference: {stats['difference_abs_mean']:.6f}\n\n")
        
        f.write("Frobenius Norms:\n")
        f.write(f"ESM-initialised CKA matrix: {stats['frobenius_norm_pretrained']:.6f}\n")
        f.write(f"Random CKA matrix: {stats['frobenius_norm_random']:.6f}\n")
        f.write(f"Difference matrix: {stats['frobenius_norm_difference']:.6f}\n\n")
        
        f.write("Interpretation:\n")
        f.write("- Small differences suggest convergent layer coordination\n")
        f.write("- Large differences suggest distinct coordination patterns\n")
        f.write("- Red areas: ESM-initialised has higher CKA than Random\n")
        f.write("- Blue areas: Random has higher CKA than ESM-initialised\n")
    
    # Clean up models
    del pretrained_model
    del random_model
    torch.cuda.empty_cache()
    
    print(f"\nAnalysis complete. Results saved to {run_output_dir}")
    print("Generated files:")
    print(f"  - pretrained_cka_heatmap.pdf: ESM-initialised model self-similarity")
    print(f"  - random_cka_heatmap.pdf: Random model self-similarity")
    print(f"  - cka_difference_heatmap.pdf: Difference plot (main result)")
    print(f"  - pretrained_cka_matrix.npy: Raw ESM-initialised CKA matrix")
    print(f"  - random_cka_matrix.npy: Raw random CKA matrix")
    print(f"  - cka_difference_matrix.npy: Raw difference matrix")
    print(f"  - analysis_summary.txt: Statistical analysis")
    print(f"  - run_config.txt: Configuration used")

if __name__ == "__main__":
    main()
