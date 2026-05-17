"""
Extract Layer-Specific Embeddings from BigCARP for MIBiG Classification
========================================================================

This script extracts embeddings from different layers of a trained BigCARP model 
for BGC classification tasks. Similar to BERT layer analysis, it supports:
  - Individual layers (e.g., layer 4, 8, 16, 31)
  - Last layer (layer 31 - current baseline)
  - Embedder layer (raw, non-contextualized embeddings)
  - Averaged layers (e.g., average of last 4 layers)
  - Concatenated layers (e.g., concatenate layers 16 and 31)

Layer Configuration Strategies (inspired by BERT analysis):
  1. last: Final layer (layer 31) - baseline
  2. embedder: Raw embedding layer (no context)
  3. early: Layer 4 (early contextualization)
  4. early_mid: Layer 8
  5. middle: Layer 16 (middle of network)
  6. late_mid: Layer 24
  7. second_last: Layer 30
  8. avg_last_4: Average of layers 28-31
  9. avg_all: Average of all 32 layers

Usage:
    python scripts/classification/extract_mibig_layer_embeddings.py \
        --checkpoint artifacts/bigcarp/bigcarp_models/run_random_200epochs/checkpoint_epoch99.tar \
        --vocab-path data/processed/vocabularies/pfam_vocab_present.json \
        --mibig-data data/processed/bgc_product_classification/processed_mibig3/mibig3_preprocessed.pkl \
        --output-dir artifacts/classification/mibig3_layer_analysis \
        --model-name random_init \
        --layer-configs last middle avg_last_4

Output:
    Saves pickle files to output_dir/model_name/:
        - mibig3_bigcarp_{layer_config}.pkl
    Each file contains a DataFrame with columns:
        - bgc_id: BGC identifier
        - embeddings: Layer-specific embeddings
        - product_class: Product class labels
"""

import os
import argparse
import json
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sequence_models.convolutional import ByteNetLM
from cgrep.bigcarp_functions import load_data
import sys

# Layer configuration definitions (for 32-layer ByteNetLM)
LAYER_CONFIGS = {
    "last": {"type": "single", "layers": [31], "description": "Final layer (baseline)"},
    "embedder": {"type": "embedder", "layers": None, "description": "Raw embedding layer"},
    "early": {"type": "single", "layers": [4], "description": "Early layer (4/32)"},
    "early_mid": {"type": "single", "layers": [8], "description": "Early-middle layer (8/32)"},
    "middle": {"type": "single", "layers": [16], "description": "Middle layer (16/32)"},
    "late_mid": {"type": "single", "layers": [24], "description": "Late-middle layer (24/32)"},
    "second_last": {"type": "single", "layers": [30], "description": "Second-to-last layer"},
    "avg_last_4": {"type": "average", "layers": [28, 29, 30, 31], "description": "Average of last 4 layers"},
    "avg_middle_4": {"type": "average", "layers": [14, 15, 16, 17], "description": "Average of middle 4 layers"},
    "avg_all": {"type": "average", "layers": list(range(32)), "description": "Average of all layers"},
}


def load_mibig_preprocessed_data(mibig_pkl_path):
    """
    Load preprocessed MIBiG data from pickle file.
    
    Args:
        mibig_pkl_path (str): Path to MIBiG pickle file (preprocessed or existing embedding file)
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['bgc_id', 'domains', 'product_class', ...]
    """
    print(f"Loading MIBiG data from: {mibig_pkl_path}")
    
    # Try loading with standard pickle first, fall back to pickle5 for compatibility
    try:
        df = pd.read_pickle(mibig_pkl_path)
    except (ValueError, AttributeError) as e:
        if "unsupported pickle protocol" in str(e) or "_unpickle_block" in str(e):
            print(f"   Using pickle5 for compatibility...")
            try:
                import pickle5 as pickle
            except ImportError:
                print(f"   ERROR: pickle5 not available. Please install: pip install pickle5")
                print(f"   Or use an existing embedding file that's compatible with current environment")
                raise
            with open(mibig_pkl_path, 'rb') as f:
                df = pickle.load(f)
        else:
            raise
    
    # Check if this is an existing embedding file or preprocessed data
    if 'embeddings' in df.columns and 'domains' not in df.columns:
        print(f"   Detected existing embedding file - will extract bgc_id and product_class only")
        # For existing embedding files, we need domains column - this won't work
        print(f"   ERROR: Cannot use existing embedding file as input - need original domain sequences")
        print(f"   Please provide preprocessed data with 'domains' column or fix pickle compatibility")
        raise ValueError("Invalid input file: need 'domains' column for re-extraction")
    
    print(f"   Loaded {len(df)} BGC sequences")
    return df


def load_vocab_and_model(vocab_path, checkpoint_path, frozen_embeddings=False, device='cuda'):
    """
    Load vocabulary and initialize BigCARP model from checkpoint.
    
    Args:
        vocab_path (str): Path to vocabulary JSON file
        checkpoint_path (str): Path to model checkpoint (.tar file)
        frozen_embeddings (bool): Whether model uses frozen embeddings
        device (str): Device to load model on
        
    Returns:
        tuple: (model, vocab_info, specials, domains, n_tokens, padding_idx, mask_idx)
    """
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        vocab_info = json.load(f)
    
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    n_tokens = vocab_info['size']
    padding_idx = specials['-']
    mask_idx = specials['#']
    
    print(f"   Vocabulary size: {n_tokens}")
    print(f"   Number of domains: {len(domains)}")
    
    # Determine frozen embeddings
    n_frozen_embs = None
    if frozen_embeddings:
        n_frozen_embs = len(domains) - 1
        print(f"   Using frozen embeddings: {n_frozen_embs}")
    
    # Create model architecture
    print(f"Initializing model...")
    model = ByteNetLM(
        n_tokens=n_tokens,
        d_embedding=1280,
        d_model=256,
        n_layers=32,
        kernel_size=3,
        r=128,
        slim=True,
        padding_idx=padding_idx,
        causal=False,
        final_ln=True,
        activation="gelu",
        n_frozen_embs=n_frozen_embs,
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model, vocab_info, specials, domains, n_tokens, padding_idx, mask_idx


def tokenize_bgc_sequences(df, specials, domains, unconditional=True):
    """
    Convert BGC domain sequences to token IDs.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'domain_sequence' column
        specials (dict): Special token mapping
        domains (dict): Domain token mapping
        unconditional (bool): If False, prepend function token
        
    Returns:
        list: List of torch.Tensor token sequences
    """
    print(f"Tokenizing {len(df)} sequences...")
    tokens_list = []
    
    # Check which column name is used
    domain_col = 'domain_sequence' if 'domain_sequence' in df.columns else 'domains'
    
    for _, row in df.iterrows():
        if unconditional:
            t = []
        else:
            # Prepend function token
            func_token = row.get('function', '*')
            t = [specials.get(func_token, specials['*'])]
        
        # Get domain sequence - handle both list and string formats
        domain_seq = row[domain_col]
        if isinstance(domain_seq, str):
            domain_list = domain_seq.split(';')
        elif isinstance(domain_seq, list):
            domain_list = domain_seq
        else:
            domain_list = []
        
        # Convert domain sequence to tokens
        for d in domain_list:
            d = str(d).strip()
            if d and d in domains:
                t.append(domains[d])
            elif d:
                t.append(domains['UNK'])
        
        tokens_list.append(torch.tensor(t, dtype=torch.long))
    
    print(f"   Tokenization complete")
    return tokens_list


def collate_fn_simple(batch, padding_idx):
    """Simple collate function for padding sequences."""
    from sequence_models.collaters import _pad
    src = _pad(batch, padding_idx)
    return src


def extract_layer_embeddings_advanced(model, src, layer_config, padding_idx, mask_idx, device):
    """
    Extract embeddings from specific layers according to configuration.
    
    Args:
        model: ByteNetLM model
        src: Input tensor (batch_size, seq_len)
        layer_config (dict): Configuration with 'type' and 'layers'
        padding_idx (int): Padding token ID
        mask_idx (int): Mask token ID
        device: torch device
        
    Returns:
        torch.Tensor: Extracted embeddings (batch_size, seq_len, hidden_dim)
    """
    config_type = layer_config['type']
    layers = layer_config['layers']
    
    with torch.no_grad():
        src = src.to(device)
        input_mask = (src != mask_idx).float().unsqueeze(-1)
        
        # Special case: embedder (raw embeddings)
        if config_type == 'embedder':
            hidden = model.embedder.embedder(src)
            return hidden
        
        # Use built-in return_all_hidden_states feature
        _, all_hidden_states = model.embedder(src, input_mask=input_mask, return_all_hidden_states=True)
        
        # all_hidden_states[0] = raw embeddings after up_embedder
        # all_hidden_states[1] = after layer 0
        # all_hidden_states[2] = after layer 1
        # ...
        # all_hidden_states[32] = after layer 31 (final layer)
        
        if config_type == 'single':
            # Extract from a single specified layer
            layer_idx = layers[0]
            # Add 1 because all_hidden_states[0] is the embedding, layers start at index 1
            hidden = all_hidden_states[layer_idx + 1]
        elif config_type == 'average':
            # Average over multiple specified layers
            layer_outputs = [all_hidden_states[layer_idx + 1] for layer_idx in layers]
            hidden = torch.stack(layer_outputs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
        
        # Apply final layer normalization (like the model does)
        return model.last_norm(hidden)


def extract_bgc_embeddings(model, tokens_list, padding_idx, mask_idx, layer_config, 
                          device='cuda', batch_size=32):
    """
    Extract BGC-level embeddings (mean-pooled) for all sequences.
    
    Args:
        model: Trained BigCARP model
        tokens_list (list): List of tokenized sequences
        padding_idx (int): Padding token ID
        mask_idx (int): Mask token ID
        layer_config (dict): Layer configuration
        device (str): Device for computation
        batch_size (int): Batch size for processing
        
    Returns:
        list: List of numpy arrays (one embedding vector per BGC)
    """
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, tokens):
            self.tokens = tokens
        def __len__(self):
            return len(self.tokens)
        def __getitem__(self, idx):
            return self.tokens[idx]
    
    dataset = SimpleDataset(tokens_list)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda b: collate_fn_simple(b, padding_idx)
    )
    
    all_embeddings = []
    
    print(f"Extracting embeddings from layer config: {layer_config['description']}")
    
    with torch.no_grad():
        for batch_src in tqdm(dataloader, desc="Processing batches"):
            # Extract layer-specific embeddings
            hidden = extract_layer_embeddings_advanced(
                model, batch_src, layer_config, padding_idx, mask_idx, device
            )
            
            # Mean-pool over sequence dimension (ignoring padding)
            batch_src_device = batch_src.to(device)
            mask = (batch_src_device != padding_idx).float().unsqueeze(-1)  # (bs, seq_len, 1)
            
            # Masked mean pooling
            masked_hidden = hidden * mask
            seq_lengths = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
            bgc_embeddings = masked_hidden.sum(dim=1) / seq_lengths  # (bs, hidden_dim)
            
            # Move to CPU and convert to numpy
            bgc_embeddings_np = bgc_embeddings.cpu().numpy()
            
            for i in range(bgc_embeddings_np.shape[0]):
                all_embeddings.append(bgc_embeddings_np[i])
    
    return all_embeddings


def save_embeddings_to_pickle(df, embeddings, output_path):
    """
    Save embeddings to pickle file in MIBiG format.
    
    Args:
        df (pandas.DataFrame): Original MIBiG DataFrame
        embeddings (list): List of embedding vectors
        output_path (str): Output pickle file path
    """
    # Create output dataframe
    output_df = pd.DataFrame({
        'bgc_id': df['bgc_id'].values,
        'embeddings': embeddings,
        'product_class': df['product_class'].values
    })
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_pickle(output_path)
    
    print(f"   Saved embeddings to: {output_path}")
    print(f"   Shape: {len(embeddings)} BGCs x {embeddings[0].shape[0]} dims")


def main():
    parser = argparse.ArgumentParser(
        description="Extract layer-specific embeddings from BigCARP for MIBiG classification"
    )
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.tar)')
    parser.add_argument('--vocab-path', required=True, help='Path to vocabulary JSON')
    parser.add_argument('--mibig-data', required=True, help='Path to MIBiG data (preprocessed pkl or existing embedding pkl)')
    parser.add_argument('--output-dir', required=True, help='Output directory for embeddings')
    parser.add_argument('--model-name', required=True, help='Model name (e.g., esm_init, random_init)')
    parser.add_argument('--layer-configs', nargs='+', 
                       choices=list(LAYER_CONFIGS.keys()),
                       default=list(LAYER_CONFIGS.keys()),
                       help='Layer configurations to extract')
    parser.add_argument('--frozen-embeddings', action='store_true',
                       help='Model uses frozen pre-trained embeddings')
    parser.add_argument('--unconditional', action='store_true', default=True,
                       help='Do not prepend function token (default: True)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("="*70)
    print("BigCARP Layer Embedding Extraction")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model name: {args.model_name}")
    print(f"MIBiG data: {args.mibig_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Layer configs: {args.layer_configs}")
    print(f"Device: {args.device}")
    print("="*70)
    print()
    
    # Load data and model
    mibig_df = load_mibig_preprocessed_data(args.mibig_data)
    model, vocab_info, specials, domains, n_tokens, padding_idx, mask_idx = load_vocab_and_model(
        args.vocab_path, args.checkpoint, args.frozen_embeddings, args.device
    )
    
    # Tokenize sequences
    tokens_list = tokenize_bgc_sequences(mibig_df, specials, domains, args.unconditional)
    
    # Extract embeddings for each layer configuration
    for config_name in args.layer_configs:
        if config_name not in LAYER_CONFIGS:
            print(f"WARNING: Unknown layer config '{config_name}', skipping")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {config_name}")
        print(f"{'='*70}")
        
        layer_config = LAYER_CONFIGS[config_name]
        print(f"Description: {layer_config['description']}")
        print(f"Type: {layer_config['type']}")
        if layer_config['layers']:
            print(f"Layers: {layer_config['layers']}")
        
        # Extract embeddings
        embeddings = extract_bgc_embeddings(
            model, tokens_list, padding_idx, mask_idx, layer_config,
            device=args.device, batch_size=args.batch_size
        )
        
        # Save to pickle
        output_filename = f"mibig3_bigcarp_{config_name}.pkl"
        output_path = os.path.join(args.output_dir, args.model_name, output_filename)
        save_embeddings_to_pickle(mibig_df, embeddings, output_path)
    
    print(f"\n{'='*70}")
    print("Extraction complete!")
    print("="*70)
    print(f"\nGenerated {len(args.layer_configs)} embedding files in:")
    print(f"  {os.path.join(args.output_dir, args.model_name)}/")


if __name__ == '__main__':
    main()
