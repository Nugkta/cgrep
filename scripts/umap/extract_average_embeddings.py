"""
This script extracts average embeddings from a trained ByteNetLM model for a given corpus of sequences.
It supports two modes:
1. Bulk processing: Process all checkpoint files in a directory
2. Single processing: Process a single checkpoint file

Usage:
  # Bulk processing mode:
  python scripts/umap/extract_average_embeddings.py bulk \
    --checkpoint-dir /path/to/checkpoints \
    --vocab-path vocab.json \
    --corpus-path corpus.csv \
    --save-dir /path/to/save

  # Single processing mode:

"""

import torch
import pandas as pd
import numpy as np
import json
import argparse
import glob
from pathlib import Path
from torch.utils.data import DataLoader
from sequence_models.convolutional import ByteNetLM
from cgrep import utils
from tqdm import tqdm 
import os


def find_checkpoint_files(checkpoint_dir):
    """Find all .tar checkpoint files in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    return list(checkpoint_dir.glob("*.tar"))

def load_vocab_info(vocab_path):
    """Load vocabulary information from JSON file."""
    with open(vocab_path, "r") as f:
        vocab_info = json.load(f)
    
    specials = vocab_info["specials"]
    domains = vocab_info["domains"]
    padding_idx = specials["-"]
    mask_idx = specials["#"]
    n_tokens = vocab_info["size"]
    
    return vocab_info, specials, domains, padding_idx, mask_idx, n_tokens

def load_corpus_data(corpus_path, specials, domains, mask_idx, padding_idx):
    """Load and preprocess corpus data."""
    df = pd.read_csv(corpus_path).fillna("")
    tokens_list = []
    
    for _, row in df.iterrows():
        func_token = row["function"]
        func_id = specials.get(func_token, specials["*"])
        sequence_ids = [func_id]
        
        for d in row["domains"].split(";"):
            d = d.strip()
            sequence_ids.append(domains.get(d, domains["UNK"]))
        tokens_list.append(torch.tensor(sequence_ids))

    ds = utils.ListDataset_extraction(tokens_list)
    dl = DataLoader(ds, batch_size=256, shuffle=False,
                    collate_fn=lambda b: utils.mlm_collate_fn_extraction(b, mask_idx, padding_idx, mask_frac=0))
    
    return dl

def create_model(n_tokens, mask_idx, n_frozen_embs, device):
    """Create and return the ByteNetLM model."""
    model = ByteNetLM(
        n_tokens=n_tokens,
        d_embedding=1280,
        d_model=256,
        n_layers=32,
        kernel_size=3,
        r=128,
        slim=True,
        padding_idx=mask_idx,
        causal=False,
        final_ln=True,
        activation="gelu",
        n_frozen_embs=n_frozen_embs,
    ).to(device)
    
    return model

def extract_and_save_embeddings(model, checkpoint_path, dataloader, n_tokens, min_special_id, 
                               padding_idx, mask_idx, save_path, layer_indices, device):
    """Extract embeddings from a checkpoint and save to disk."""
    # Load checkpoint and update model weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Extract average embeddings
    avg_embeddings = utils.generate_average_embeddings(
        model, dataloader, n_tokens, min_special_id, padding_idx, mask_idx, 
        layer_indices=layer_indices, device=device
    )
    
    # Convert to numpy arrays and then to tensor
    avg_emb_np = {t_id: avg_embeddings[t_id].numpy() for t_id in avg_embeddings}
    avg_emb_np_array = np.array(list(avg_emb_np.values()))
    avg_emb_tensor = torch.tensor(avg_emb_np_array)
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the embeddings tensor
    torch.save(avg_emb_tensor, save_path)
    return save_path

def process_single_checkpoint(args):
    """Process a single checkpoint file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary info
    vocab_info, specials, domains, padding_idx, mask_idx, n_tokens = load_vocab_info(args.vocab_path)
    
    # Handle frozen embeddings
    n_frozen_embs = None
    if args.frozen_embeddings:
        n_frozen_embs = len(domains) - 1
    
    # Load corpus data
    dataloader = load_corpus_data(args.corpus_path, specials, domains, mask_idx, padding_idx)
    
    # Create model
    model = create_model(n_tokens, mask_idx, n_frozen_embs, device)
    
    # Extract embeddings for each layer separately
    checkpoint_name = Path(args.checkpoint_path).stem
    
    for layer_idx in args.layer_indices:
        save_path = Path(args.save_dir) / f"embeddings_{checkpoint_name}_{layer_idx}.pt"
        
        extract_and_save_embeddings(
            model, args.checkpoint_path, dataloader, n_tokens, args.min_special_id,
            padding_idx, mask_idx, save_path, [layer_idx], device
        )
        
        print(f"Embeddings saved to {save_path}")

def process_bulk_checkpoints(args):
    """Process all checkpoint files in a directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all checkpoint files
    checkpoint_files = find_checkpoint_files(args.checkpoint_dir)
    
    if not checkpoint_files:
        print(f"No .tar checkpoint files found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Load vocabulary info
    vocab_info, specials, domains, padding_idx, mask_idx, n_tokens = load_vocab_info(args.vocab_path)
    
    # Handle frozen embeddings
    n_frozen_embs = None
    if args.frozen_embeddings:
        n_frozen_embs = len(domains) - 1
    
    # Load corpus data
    dataloader = load_corpus_data(args.corpus_path, specials, domains, mask_idx, padding_idx)
    
    # Create model
    model = create_model(n_tokens, mask_idx, n_frozen_embs, device)
    
    # Process each checkpoint
    for checkpoint_path in tqdm(checkpoint_files, desc="Processing Checkpoints"):
        checkpoint_name = checkpoint_path.stem
        
        # Extract embeddings for each layer separately
        for layer_idx in args.layer_indices:
            save_path = Path(args.save_dir) / f"embeddings_{checkpoint_name}_{layer_idx}.pt"
            
            try:
                extract_and_save_embeddings(
                    model, checkpoint_path, dataloader, n_tokens, args.min_special_id,
                    padding_idx, mask_idx, save_path, [layer_idx], device
                )
                tqdm.write(f"Embeddings saved to {save_path}")
            except Exception as e:
                tqdm.write(f"Error processing {checkpoint_path} layer {layer_idx}: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract average embeddings from ByteNetLM checkpoints")
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--vocab-path", required=True, help="Path to vocabulary JSON file")
    common_parser.add_argument("--corpus-path", required=True, help="Path to corpus CSV file")
    common_parser.add_argument("--save-dir", required=True, help="Directory to save embeddings")
    common_parser.add_argument("--min-special-id", type=int, default=4, help="Special tokens to skip")
    common_parser.add_argument("--layer-indices", nargs="+", default=["last"], help="Layer indices to extract (separate file per layer)")
    common_parser.add_argument("--frozen-embeddings", action="store_true", help="Use frozen embeddings")
    
    # Single checkpoint mode
    single_parser = subparsers.add_parser("single", parents=[common_parser], help="Process a single checkpoint")
    single_parser.add_argument("--checkpoint-path", required=True, help="Path to checkpoint file")
    
    # Bulk processing mode
    bulk_parser = subparsers.add_parser("bulk", parents=[common_parser], help="Process all checkpoints in a directory")
    bulk_parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoint files")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.mode == "single":
        process_single_checkpoint(args)
    elif args.mode == "bulk":
        process_bulk_checkpoints(args)
    else:
        print("Please specify a processing mode: 'single' or 'bulk'")

if __name__ == "__main__":
    main()