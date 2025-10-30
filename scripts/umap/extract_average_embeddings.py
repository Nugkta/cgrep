"""
This script extracts average embeddings from a trained bigcarp model for a given corpus of sequences.
It supports two modes:
1. Bulk processing: Process all checkpoint files in a directory
2. Single processing: Process a single checkpoint file

Arguments:
  mode: Either 'single' or 'bulk'
  --checkpoint-path: Path to a single checkpoint file (.tar) [single mode only]
  --checkpoint-dir: Directory containing checkpoint files (.tar) [bulk mode only]
  --vocab-path: Path to vocabulary JSON file containing 'specials', 'domains', and 'size'
  --corpus-path: Path to corpus CSV file with columns ['domains', 'function', 'split']
  --save-dir: Directory to save extracted embedding files (.pt)
  --min-special-id: Number of special tokens to skip when extracting embeddings (default: 4)
  --layer-indices: Layer indices to extract embeddings from. Use 'last' for final layer or space-separated integers (default: last)
  --frozen-embeddings: If set, model uses frozen pre-trained embeddings (n_frozen_embs parameter)
  --conditional: If set, prepend functional token at the start of sequences (default: False, unconditional)

Usage:
  # Bulk processing mode:
  python scripts/umap/extract_average_embeddings.py bulk \
    --checkpoint-dir /path/to/checkpoints \
    --vocab-path vocab.json \
    --corpus-path corpus.csv \
    --save-dir /path/to/save

  # Single processing mode:
  python scripts/umap/extract_average_embeddings.py single \
    --checkpoint-path artifacts/bigcarp/bigcarp_models/run_esm_init_frozen/checkpoint_best.tar \
    --vocab-path data/processed/vocabularies/pfam_vocab_present.json \
    --corpus-path data/processed/bgc_corpus/antidb_pfam_corpus.csv \
    --save-dir artifacts/bigcarp/average_embeddings/esm_init_frozen \
    --frozen-embeddings \
    --layer-indices last

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
    """Find all .tar checkpoint files in a directory.

    Args:
        checkpoint_dir (str or Path): Directory path to search for checkpoint files

    Returns:
        list of Path: List of Path objects for all .tar files found in the directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    return list(checkpoint_dir.glob("*.tar"))

def load_vocab_info(vocab_path):
    """Load vocabulary information from JSON file.

    Args:
        vocab_path (str): Path to vocabulary JSON file containing 'specials', 'domains', and 'size' keys

    Returns:
        tuple: (vocab_info, specials, domains, padding_idx, mask_idx, n_tokens) where:
            - vocab_info (dict): Full vocabulary dictionary from JSON
            - specials (dict): Mapping of special token names to IDs
            - domains (dict): Mapping of domain names to IDs
            - padding_idx (int): Token ID for padding ('-')
            - mask_idx (int): Token ID for masking ('#')
            - n_tokens (int): Total vocabulary size
    """
    with open(vocab_path, "r") as f:
        vocab_info = json.load(f)
    
    specials = vocab_info["specials"]
    domains = vocab_info["domains"]
    padding_idx = specials["-"]
    mask_idx = specials["#"]
    n_tokens = vocab_info["size"]
    
    return vocab_info, specials, domains, padding_idx, mask_idx, n_tokens

def load_corpus_data(corpus_path, specials, domains, mask_idx, padding_idx, conditional=False, subcluster=False):
    """Load and preprocess corpus data into a DataLoader.

    Args:
        corpus_path (str): Path to corpus CSV file with columns ['domains', 'function', 'split']
                           or subcluster format (first column BGC ID, remaining columns domains)
        specials (dict): Mapping of special token names to IDs
        domains (dict): Mapping of domain names to IDs
        mask_idx (int): Token ID for masking
        padding_idx (int): Token ID for padding
        conditional (bool): If True, prepend functional token at the start of sequences (default: False)
        subcluster (bool): If True, read corpus in subcluster format (default: False)

    Returns:
        DataLoader: PyTorch DataLoader with batched and tokenized sequences (batch_size=256)
    """
    tokens_list = []

    if subcluster:
        # Subcluster format: first column is BGC ID, remaining columns are subcluster domains
        # Read line-by-line to handle variable number of columns
        with open(corpus_path, 'r') as f:
            for line in f:
                # Split by comma to get all fields
                fields = line.strip().split(',')
                if len(fields) < 2:  # Skip lines with only BGC ID or empty lines
                    continue

                # Skip first field (BGC ID), remaining fields are domains
                sequence_ids = []
                for domain_val in fields[1:]:
                    # Skip empty/NaN values (represented as '-' or empty)
                    if not domain_val or domain_val == '-':
                        continue

                    # Handle semicolon-separated domains within a cell
                    for d in domain_val.split(';'):
                        d = d.strip()
                        if d and d != '-':
                            sequence_ids.append(domains.get(d, domains["UNK"]))

                if len(sequence_ids) > 0:  # Only add sequences with at least one domain
                    tokens_list.append(torch.tensor(sequence_ids))

    else:
        # Original format: columns ['domains', 'function', 'split']
        df = pd.read_csv(corpus_path).fillna("")

        for _, row in df.iterrows():
            if conditional:
                # Include functional token at the start
                func_token = row["function"]
                func_id = specials.get(func_token, specials["*"])
                sequence_ids = [func_id]
            else:
                # Unconditional: start with empty sequence
                sequence_ids = []

            for d in row["domains"].split(";"):
                d = d.strip()
                sequence_ids.append(domains.get(d, domains["UNK"]))
            tokens_list.append(torch.tensor(sequence_ids))

    ds = utils.ListDataset_extraction(tokens_list)
    dl = DataLoader(ds, batch_size=256, shuffle=False,
                    collate_fn=lambda b: utils.mlm_collate_fn_extraction(b, mask_idx, padding_idx, mask_frac=0))

    return dl

def create_model(n_tokens, mask_idx, n_frozen_embs, device):
    """Create and return the ByteNetLM model with specified architecture.

    Args:
        n_tokens (int): Total vocabulary size
        mask_idx (int): Token ID for padding
        n_frozen_embs (int or None): Number of frozen pre-trained embeddings, or None if not using frozen embeddings
        device (torch.device): Device to place the model on ('cuda' or 'cpu')

    Returns:
        ByteNetLM: Initialized model with d_embedding=1280, d_model=256, n_layers=32, kernel_size=3
    """
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
    """Extract embeddings from a checkpoint and save to disk.

    Args:
        model (ByteNetLM): Model architecture to load checkpoint weights into
        checkpoint_path (str or Path): Path to checkpoint .tar file
        dataloader (DataLoader): DataLoader with tokenized sequences
        n_tokens (int): Total vocabulary size
        min_special_id (int): Number of special tokens to skip when extracting embeddings
        padding_idx (int): Token ID for padding
        mask_idx (int): Token ID for masking
        save_path (str or Path): Output path for saving embeddings .pt file
        layer_indices (list of int): Layer indices to extract embeddings from
        device (torch.device): Device for computation

    Returns:
        str or Path: Path where embeddings were saved

    Output:
        Saves embeddings tensor to save_path as .pt file
    """
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
    """Process a single checkpoint file and extract embeddings for specified layers.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - checkpoint_path (str): Path to checkpoint .tar file
            - vocab_path (str): Path to vocabulary JSON
            - corpus_path (str): Path to corpus CSV
            - save_dir (str): Output directory for embeddings
            - min_special_id (int): Number of special tokens to skip
            - layer_indices (list): Layer indices to extract
            - frozen_embeddings (bool): Whether to use frozen embeddings
            - conditional (bool): Whether to prepend functional tokens

    Output:
        Saves embeddings to: save_dir/embeddings_{checkpoint_name}_{layer_idx}.pt
        Prints confirmation message for each saved file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary info
    vocab_info, specials, domains, padding_idx, mask_idx, n_tokens = load_vocab_info(args.vocab_path)

    # Handle frozen embeddings
    n_frozen_embs = None
    if args.frozen_embeddings:
        n_frozen_embs = len(domains) - 1

    # Load corpus data
    dataloader = load_corpus_data(args.corpus_path, specials, domains, mask_idx, padding_idx,
                                   conditional=args.conditional, subcluster=args.subcluster)

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
    """Process all checkpoint files in a directory and extract embeddings.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - checkpoint_dir (str): Directory containing checkpoint .tar files
            - vocab_path (str): Path to vocabulary JSON
            - corpus_path (str): Path to corpus CSV
            - save_dir (str): Output directory for embeddings
            - min_special_id (int): Number of special tokens to skip
            - layer_indices (list): Layer indices to extract
            - frozen_embeddings (bool): Whether to use frozen embeddings
            - conditional (bool): Whether to prepend functional tokens

    Output:
        Saves embeddings to: save_dir/embeddings_{checkpoint_name}_{layer_idx}.pt for each checkpoint
        Prints progress with tqdm and handles errors gracefully
    """
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
    dataloader = load_corpus_data(args.corpus_path, specials, domains, mask_idx, padding_idx,
                                   conditional=args.conditional, subcluster=args.subcluster)

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
    """Parse command line arguments for embedding extraction script.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - mode (str): Processing mode ('single' or 'bulk')
            - vocab_path (str): Path to vocabulary JSON file
            - corpus_path (str): Path to corpus CSV file
            - save_dir (str): Directory to save embeddings
            - min_special_id (int): Special tokens to skip (default: 4)
            - layer_indices (list): Layer indices to extract (default: ['last'])
            - frozen_embeddings (bool): Whether to use frozen embeddings
            - conditional (bool): Whether to include functional tokens
            Single mode only:
            - checkpoint_path (str): Path to checkpoint file
            Bulk mode only:
            - checkpoint_dir (str): Directory containing checkpoint files
    """
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
    common_parser.add_argument("--conditional", action="store_true", help="Include functional token at the start of sequences")
    common_parser.add_argument("--subcluster", action="store_true", help="Read corpus in subcluster format: first column is BGC ID, remaining columns are subcluster domains")
    
    # Single checkpoint mode
    single_parser = subparsers.add_parser("single", parents=[common_parser], help="Process a single checkpoint")
    single_parser.add_argument("--checkpoint-path", required=True, help="Path to checkpoint file")
    
    # Bulk processing mode
    bulk_parser = subparsers.add_parser("bulk", parents=[common_parser], help="Process all checkpoints in a directory")
    bulk_parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoint files")
    
    return parser.parse_args()

def main():
    """Main execution function that orchestrates embedding extraction workflow.

    Workflow:
        1. Parse command line arguments to determine processing mode
        2. Route to appropriate processing function:
            - Single mode: Process one checkpoint file
            - Bulk mode: Process all checkpoints in directory
        3. Each process loads vocabulary, corpus, creates model, and extracts embeddings

    Output:
        Saves embedding tensors to save_dir with naming: embeddings_{checkpoint_name}_{layer_idx}.pt
    """
    args = parse_arguments()
    
    if args.mode == "single":
        process_single_checkpoint(args)
    elif args.mode == "bulk":
        process_bulk_checkpoints(args)
    else:
        print("Please specify a processing mode: 'single' or 'bulk'")

if __name__ == "__main__":
    main()