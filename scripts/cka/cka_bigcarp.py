"""
CKA Analysis for bigcarp Models

This script performs Centered Kernel Alignment (CKA) analysis to compare the internal representations
of two bigcarp (ByteNet Language Model) checkpoints. CKA is a method for measuring the similarity
between neural network representations across different layers.

The script:
1. Loads two bigcarp model checkpoints from specified paths
2. Extracts intermediate layer representations from both models on the same input data
3. Computes CKA similarity matrices between corresponding layers of the two models
4. Generates visualizations including:
   - Heatmap showing layer-to-layer CKA similarities
   - Diagonal plot showing self-similarity across layers
5. Saves results and configuration for reproducibility

This is useful for:
- Analyzing how model representations evolve during training
- Comparing different training strategies (e.g., frozen vs unfrozen embeddings)
- Understanding which layers change most between checkpoints
- Validating that models learn similar representations

Usage:
    python scripts/cka/cka_bigcarp.py \
        --model1_path path/to/checkpoint1.tar \
        --model2_path path/to/checkpoint2.tar \
        --fcorpus data/corpus.csv \
        --fvocab data/vocab.json \
        --d_embedding 1280 \
        --d_model 256 \
        --n_layers 32 \
        --output_dir results/cka_analysis
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

# Import shared bigcarp functions from the cgrep package
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
    parser = argparse.ArgumentParser(description='CKA analysis of bigcarp models')
    
    # Paths to model checkpoints
    parser.add_argument('--model1_path', type=str, required=True, 
                        help='Path to the first model checkpoint')
    parser.add_argument('--model2_path', type=str, required=True,
                        help='Path to the second model checkpoint')
    
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
    parser.add_argument('--output_dir', type=str, default='results/bigcarp/cka',
                        help='Directory to save CKA analysis results')
    
    # Replace single frozen embeddings flag with model-specific flags
    parser.add_argument('--model1_frozen', action='store_true',
                        help='If set, load model1 with frozen embeddings')
    parser.add_argument('--model2_frozen', action='store_true',
                        help='If set, load model2 with frozen embeddings')
    
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

def compute_cka_between_models(model1, model2, dataloader, device, n_batches=8):
    model1.eval(); model2.eval()

    hsic_sum = var1_sum = var2_sum = None

    for _ in range(n_batches):
        src, _, _ = [b.to(device) for b in next(iter(dataloader))]
        input_mask = (src != 0).float().unsqueeze(-1)
        mask_flat  = input_mask.squeeze(-1).reshape(-1).bool()   # <-- define it here

        with torch.no_grad():
            _, h1 = model1.embedder(src, input_mask=input_mask, return_all_hidden_states=True)
            _, h2 = model2.embedder(src, input_mask=input_mask, return_all_hidden_states=True)

        if hsic_sum is None:                    # allocate on first pass
            L1, L2 = len(h1), len(h2)
            hsic_sum = torch.zeros(L1, L2, device=device)
            var1_sum = torch.zeros(L1, device=device)
            var2_sum = torch.zeros(L2, device=device)

        # ---- outer loop over layers of model 1 ----
        for i, l1 in enumerate(h1):
            f1 = l1.reshape(-1, l1.size(-1))[mask_flat]
            f1 -= f1.mean(0, keepdim=True)
            gram1 = f1 @ f1.T
            var1  = (gram1 * gram1).sum()
            var1_sum[i] += var1                        # <-- accumulate

            # ---- inner loop over layers of model 2 ----
            for j, l2 in enumerate(h2):
                f2 = l2.reshape(-1, l2.size(-1))[mask_flat]
                f2 -= f2.mean(0, keepdim=True)
                gram2 = f2 @ f2.T

                if i == 0:                             # add each var2 only once per-batch
                    var2_sum[j] += (gram2 * gram2).sum()

                hsic_sum[i, j] += (gram1 * gram2).sum()

    # final normalisation
    denom = torch.outer(torch.sqrt(var1_sum), torch.sqrt(var2_sum))
    cka   = (hsic_sum / denom).cpu().numpy()
    return cka

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"cka_run_{timestamp}")
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
    # Fix: Store all returned values and only use the first one (test dataloader)
    dataloaders = prepare_dataloaders(
        test_tokens, test_tokens, domain_tokens, mask_idx,  # Using test data twice to match function signature
        padding_idx, batch_size=args.batch_size, num_workers=4
    )
    dl_test = dataloaders[0]  # Only use the first dataloader
    
    # Load models with their specific frozen configuration
    print(f"Loading model 1 from {args.model1_path}")
    model1 = load_model(args.model1_path, n_tokens, mask_idx, domains, args.model1_frozen, args)
    model1.to(device)
    
    print(f"Loading model 2 from {args.model2_path}")
    model2 = load_model(args.model2_path, n_tokens, mask_idx, domains, args.model2_frozen, args)
    model2.to(device)
    
    # Compute CKA
    print("Computing CKA between models...")
    cka_matrix = compute_cka_between_models(model1, model2, dl_test, device)
    
    # Save results to timestamped directory
    np.save(os.path.join(run_output_dir, 'cka_matrix.npy'), cka_matrix)
    
    # Extract model names from paths for better labels
    model1_name = os.path.basename(args.model1_path).replace('.tar', '')
    model2_name = os.path.basename(args.model2_path).replace('.tar', '')
    
    # Plot and save figure with improved axes
    fig, ax = plt.subplots(figsize=(12, 10))

    # Set better ticks for axes
    num_layers = len(cka_matrix)
    step = max(1, num_layers // 8)  # Show about 8 ticks for readability
    # tick_positions will be like [0, 4, 8, ..., 32] (if num_layers is 33 and step is 4)
    tick_positions = np.arange(0, num_layers, step)
    # Ensure the last layer is included if not covered by step
    if num_layers - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_layers - 1)
        tick_positions = np.unique(tick_positions) # ensure sorted and unique if num_layers-1 was already there or close

    tick_labels = [f"Layer {int(i)}" for i in tick_positions]

    # Create heatmap with improved formatting
    sns.heatmap(
        cka_matrix,
        annot=False,
        cmap='viridis',
        cbar_kws={'label': 'CKA Similarity', 'shrink': 0.8},
        # Let plt.xticks/yticks handle the tick labels for precision
        xticklabels=False,
        yticklabels=False,
        square=True,  # Make cells square for better visualization
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

    ax.set_xlabel(f'Layers ({model2_name})', fontweight='semibold')
    ax.set_ylabel(f'Layers ({model1_name})', fontweight='semibold')
    ax.set_title('CKA Similarity Between Model Layers', fontweight='bold', pad=20)

    plt.tight_layout()

    plt.savefig(os.path.join(run_output_dir, 'cka_heatmap.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Also update the diagonal plot with consistent labeling
    fig, ax = plt.subplots(figsize=(10, 6))
    # The diagonal plot should use the same tick logic for consistency
    diag_tick_positions = tick_positions # Use the same positions as the heatmap
    diag_tick_labels = [f"Layer {int(i)}" for i in diag_tick_positions]

    ax.plot(np.arange(num_layers), np.diag(cka_matrix), marker='o', linewidth=2.5,
            markersize=8, color='#2E86C1', markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#2E86C1')

    # Disable offset/scientific notation on both axes
    ax.ticklabel_format(useOffset=False, style='plain')
    # Apply consistent ticks to the diagonal plot
    ax.set_xticks(diag_tick_positions)
    ax.set_xticklabels(diag_tick_labels, rotation=45, ha='right')

    ax.set_xlabel('Layer Index', fontweight='semibold')
    ax.set_ylabel('CKA Similarity', fontweight='semibold')
    ax.set_title(f'Layer-wise CKA Similarity\n{model1_name} vs {model2_name}', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, 'diagonal_cka.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # Also save the command line arguments for reproducibility
    with open(os.path.join(run_output_dir, 'run_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f"Analysis complete. Results saved to {run_output_dir}")

if __name__ == "__main__":
    main()



'''
python scripts/cka_bigcarp.py \
    --model1_path artifacts/bigcarp/bigcarp_models/run_20250404_020145_pt_pfam_present/checkpoint80.tar \
    --model2_path artifacts/bigcarp/bigcarp_models/run_20250404_020145_pt_pfam_present/checkpoint80.tar \
    --fcorpus data/processed/bgc_corpus/antidb_pfam_BC.csv \
    --fvocab data/processed/vocabularies/pfam_vocab_present.json \
    --d_embedding 1280 \
    --d_model 256 \
    --n_layers 32 \
    --kernel_size 3 \
    --r 128 \
    --batch_size 32 \
    --gpu 0 \
    --output_dir results/bigcarp/cka \
    --unconditional
    '''