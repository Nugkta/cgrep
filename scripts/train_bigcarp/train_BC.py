"""
BiGCARP: ByteNet-based Masked Language Model Training for Protein Domain Sequences
===================================================================================

This script implements training for BiGCARP (Big Generative Context-Aware Representation
of Proteins), a ByteNet-based language model designed for protein domain sequence modeling.
The model uses masked language modeling (MLM) to learn contextual representations of
Pfam protein domains in biosynthetic gene cluster sequences.

Key Features:
    - Masked Language Modeling on protein domain tokens
    - ByteNet architecture with dilated convolutions for long-range dependencies
    - Pre-trained ESM embedding integration with freeze/fine-tune options
    - Conditional modeling with functional annotation tokens
    - Comprehensive checkpoint management and training resumption
    - Multi-GPU support and memory-efficient data loading

Model Architecture:
    - Input: Tokenized protein domain sequences (Pfam domains)
    - Embedding: Learnable or pre-trained (ESM) domain embeddings
    - Encoder: Multi-layer ByteNet with dilated convolutions
    - Output: Vocabulary-sized logits for masked token prediction
    - Training: Cross-entropy loss with masking strategy (80/10/10 rule)

Training Data Requirements:
    - Corpus CSV: columns [domains, function, split] where:
      * domains: semicolon-separated Pfam domain names
      * function: functional annotation for conditional modeling
      * split: 'train' or 'test' designation
    - Vocabulary JSON: token mappings for domains and special tokens
    - Optional ESM embeddings: pre-trained domain representations

Usage Examples:
    # Basic training with random initialization
    python scripts/train_bigcarp/train_BC.py \\
        --fcorpus data/processed/bgc_corpus/antidb_pfam_corpus.csv \\
        --fvocab data/processed/vocabularies/pfam_vocab_present.json \\
        --out_fpath artifacts/bigcarp/models \\
        --epochs 100 \\
        --batch_size 256


Output Files:
    - checkpoint_best.tar: Best validation performance model
    - checkpoint_latest.tar: Most recent training state for resumption  
    - checkpoint_epoch{N}.tar: Periodic backups every 5 epochs
    - metrics.csv: Training and validation metrics per epoch
    - loss_plot.png: Training and validation loss visualization
    - final_embedder.pt: Final learned embedding matrix weights

Model Hyperparameters:
    - d_embedding: Embedding dimension (default: 1280, ESM-compatible)
    - d_model: Hidden dimension in ByteNet layers (default: 256)
    - n_layers: Number of ByteNet layers (default: 32)
    - kernel_size: Convolution kernel width (default: 3)
    - r: Dilation factor parameter (default: 128)
    - batch_size: Training batch size (default: 64)
    - lr: Learning rate (default: 1e-4)
    - epochs: Number of training epochs (default: 50)

Dependencies:
    - PyTorch with CUDA support
    - sequence_models package for ByteNet implementation
    - Standard scientific Python stack (numpy, pandas, matplotlib)

Author: [Author information]
Version: 1.0
Date: 2024
"""

import argparse
import json
import os
import pathlib
import random
import time
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ----------- import the seqeuence_models packages -----------
from sequence_models.convolutional import ByteNetLM
from sequence_models.metrics import MaskedAccuracy
from sequence_models.losses import MaskedCrossEntropyLoss
from sequence_models.collaters import _pad
from sequence_models.utils import parse_fasta
# -------------------------------------------------------------

def get_parser():
    """
    Creates and configures an argument parser for BiGCARP training hyperparameters and I/O settings.

    This function defines all command-line arguments needed for training the BiGCARP model,
    including required paths, optional hyperparameters, and training configuration options.

    Returns:
        argparse.ArgumentParser: Configured argument parser with the following argument groups:
            
            Required/Key Arguments:
                --out_fpath (str): Output directory for model checkpoints and metrics
                --gpu (int): GPU index for training (default: 0)
                --restart (flag): Resume training from latest checkpoint if available
                --freeze (flag): Freeze pre-trained embeddings during training
                --pretrain (flag): Load pre-trained embeddings but allow updates
                --fcorpus (str): Path to training corpus CSV file
                --fvocab (str): Path to vocabulary JSON file
                --fdata (str): Path to pre-trained data
                --esm_emb_fpath (str): Path to ESM embeddings file
                --fpfams_fpath (str): Path to final Pfam domains file

            Model Hyperparameters:
                --batch_size (int): Training batch size (default: 64)
                --d_embedding (int): Embedding dimension (default: 1280)
                --d_model (int): Model hidden dimension (default: 256)
                --n_layers (int): Number of ByteNet layers (default: 32)
                --kernel_size (int): Convolution kernel width (default: 3)
                --r (int): Dilation factor parameter (default: 128)
                --wide (flag): Use wide ByteNet variant instead of slim
                --lr (float): Learning rate (default: 1e-4)
                --epochs (int): Number of training epochs (default: 50)

            Training Configuration:
                --conditional (flag): Prepend function tokens to sequences
                --ar (flag): Use autoregressive (causal) modeling
                --cp_fpath (str): Specific checkpoint path to load from
    """
    parser = argparse.ArgumentParser(description='Process hyperparameters')

    # Required / key arguments
    parser.add_argument('--out_fpath', type=str, required=False,
                        default='outputs/BIGCARP_output/',
                        help='Output path for model checkpoints and metrics.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for training.')
    parser.add_argument('--restart', action='store_true',
                        help='If set, attempt to restart training from the last checkpoint in out_fpath.')
    parser.add_argument('--freeze', action='store_true',
                        help='If set, freeze the pre-trained embeddings in the model.')
    parser.add_argument('--pretrain', action='store_true',
                        help='If set, load pre-trained embeddings but allow them to be updated.')
    parser.add_argument('--fcorpus', type=str, default=None,
                        help='Path to the corpus file for training.')
    parser.add_argument('--fvocab', type=str, default=None,
                        help='Path to the vocabulary file for training.')
    parser.add_argument('--fdata', type=str, default=None,
                        help='Path to the pretrained data.')
    parser.add_argument('--esm_emb_fpath', type=str, default=None,
                        help='Path to the ESM embeddings file.')
    parser.add_argument('--fpfams_fpath', type=str, default=None,
                        help='Path to the final pfams file.')

    # Optional arguments for hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--d_embedding', type=int, default=1280, help='Dimension of embedding.')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension within ByteNet model.')
    parser.add_argument('--n_layers', type=int, default=32, help='Number of ByteNet layers.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel width.')
    parser.add_argument('--r', type=int, default=128,
                        help='Used to calculate the dilation factor in the ByteNet model.')
    parser.add_argument('--wide', action='store_true',
                        help='If set, use the "wide" version instead of the "slim" version of ByteNet.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--conditional', action='store_true',
                        help='If set, prepend a special function token.')
    parser.add_argument('--ar', action='store_true',
                        help='If set, use autoregressive model (causal).')
    parser.add_argument('--cp_fpath', type=str, default=None,
                        help='Path to the checkpoint to load for training.')
    
    
    return parser


def load_data(args):
    """
    Loads training corpus, vocabulary, and prepares tokenized sequences for model training.

    This function reads the CSV corpus file and JSON vocabulary file, converts domain sequences
    to token IDs, and splits data into training and test sets. Optionally prepends function
    tokens for conditional modeling.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - fcorpus (str): Path to CSV file with columns [domains, function, split]
                domains: Semicolon-separated domain names (e.g., "PF00001;PF00002")
                function: Functional annotation string
                split: 'train' or 'test' designation
            - fvocab (str): Path to JSON file with vocabulary structure:
                {
                    'specials': {token_name: token_id, ...},
                    'domains': {domain_name: token_id, ...},
                    'size': total_vocabulary_size
                }
            - conditional (bool): If True, prepend function tokens to sequences
            - fdata (str): Path to pre-trained data (passed through)

    Returns:
        tuple: (train_tokens, test_tokens, specials, domains, domain_tokens, 
                n_tokens, padding_idx, mask_idx, data_fpath, df) where:
            - train_tokens (List[torch.Tensor]): Training sequences as token ID tensors
            - test_tokens (List[torch.Tensor]): Test sequences as token ID tensors  
            - specials (Dict[str, int]): Special token name to ID mapping
            - domains (Dict[str, int]): Domain name to token ID mapping
            - domain_tokens (np.ndarray): Array of all domain token IDs (excluding specials)
            - n_tokens (int): Total vocabulary size
            - padding_idx (int): Token ID used for sequence padding
            - mask_idx (int): Token ID used for masked language modeling
            - data_fpath (str): Path to pre-trained data (from args.fdata)
            - df (pd.DataFrame): Original corpus dataframe for reference

    Processing Details:
        - Unknown domains are mapped to 'UNK' token
        - Conditional mode: sequences start with function token
        - Unconditional mode: sequences contain only domain tokens
        - Train/test split determined by 'split' column in CSV
    """
    # Read CSV
    df = pd.read_csv(args.fcorpus)
    # Load token definitions
    vocab_info = json.load(open(args.fvocab))
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    domain_tokens = np.array([domains[d] for d in domains])
    n_tokens = vocab_info['size']
    padding_idx = specials['-']
    mask_idx = specials['#']

    # Prepare the token sequences
    tokens_list = []
    for _, row in df.iterrows():
        if args.conditional:
            # Prepend the function token if conditional
            t = [specials[row['function']]]
        else:
            # Unconditional: start with empty sequence
            t = []

        for d in row['domains'].split(';'):
            if d in domains:
                t.append(domains[d])
            else:
                # Use 'UNK' domain if the domain token is not found
                t.append(domains['UNK'])
        tokens_list.append(torch.tensor(t))

    # Split the data into train, test (the split is already done in the csv file)
    train_tokens = [tokens_list[i] for i in df[df['split'] == 'train'].index]
    test_tokens  = [tokens_list[i] for i in df[df['split'] == 'test'].index]

    data_fpath = args.fdata

    return (train_tokens, test_tokens,
            specials, domains, domain_tokens,
            n_tokens, padding_idx, mask_idx, data_fpath, df)


def mlm_collate_fn(batch, domain_tokens, mask_idx, padding_idx):
    """
    Collate function implementing masked language modeling data preparation for batch processing.

    This function takes a batch of tokenized sequences and applies random masking following
    the BERT masking strategy: 80% mask token, 10% random token, 10% original token.
    Approximately 15% of tokens in each sequence are selected for masking.

    Args:
        batch (List[Tuple[torch.Tensor]]): Batch of sequences, where each element is a tuple
            containing a single tensor of token IDs
        domain_tokens (np.ndarray): Array of valid domain token IDs for random replacement
        mask_idx (int): Special token ID used for masking (typically '#')
        padding_idx (int): Special token ID used for padding sequences (typically '-')

    Returns:
        tuple: (src, tgt, mask) where:
            - src (torch.Tensor): Padded input sequences with applied masking, shape (batch_size, max_seq_len)
            - tgt (torch.Tensor): Padded original target sequences, shape (batch_size, max_seq_len)
            - mask (torch.Tensor): Padded binary mask indicating masked positions, shape (batch_size, max_seq_len)
                1.0 for masked tokens, 0.0 for unmasked tokens

    Masking Strategy:
        For each selected position (15% of sequence):
        - 80% probability: Replace with mask_idx
        - 10% probability: Replace with random domain token (different from original)
        - 10% probability: Keep original token (no change)

    Note:
        - Empty sequences are skipped during processing
        - At least 1 token per sequence is always masked
        - All tensors are padded to the same length within the batch
        - Random replacement excludes the original token to avoid identity mapping
    """
    data = tuple(zip(*batch))
    tgt = list(data[0])  # Each element is a torch tensor of tokens

    src_list, mask_list = [], []
    for t in tgt:
        # Make a separate copy of target tokens for input
        s = t.clone().detach()
        if len(s) == 0:
            continue

        # Randomly select ~15% of positions to mask
        n_mask = max(1, int(len(s) * 0.15))
        mod_idx = random.sample(range(len(s)), n_mask)

        for idx in mod_idx:
            p = np.random.uniform()
            # 10% chance to do nothing (leave original token)
            if p <= 0.10:
                mod = t[idx]
            # 10% chance to replace with random domain token (not the same as original)
            elif 0.10 < p <= 0.20:
                # Sample from domain tokens excluding the original
                possible_tokens = domain_tokens[domain_tokens != t[idx].item()]
                mod = np.random.choice(possible_tokens) if len(possible_tokens) > 0 else t[idx]
            # 80% chance to mask
            else:
                mod = mask_idx

            s[idx] = mod

        # Prepare the mask vector
        m = torch.zeros(len(s))
        m[mod_idx] = 1.0

        src_list.append(s)
        mask_list.append(m)

    # Pad everything
    src = _pad(src_list, padding_idx)
    tgt = _pad(tgt, padding_idx)
    mask = _pad(mask_list, 0)

    return src, tgt, mask


class ListDataset(Dataset):
    """
    Simple PyTorch Dataset wrapper for list of tensors with MLM-compatible interface.

    This dataset class wraps a list of pre-tokenized sequences and provides the interface
    required by PyTorch DataLoader. Each sequence is returned as a single-element tuple
    to maintain compatibility with the mlm_collate_fn function.

    Args:
        data (List[torch.Tensor]): List of tokenized sequences, where each tensor
            contains token IDs for one sequence

    Methods:
        __getitem__(idx): Returns tuple containing the sequence at index idx
        __len__(): Returns total number of sequences in the dataset

    Note:
        The tuple wrapping in __getitem__ is specifically designed to work with
        mlm_collate_fn which expects batch elements to be tuples.
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        # Return a tuple to align with mlm_collate_fn usage
        return (self.data[idx], )

    def __len__(self):
        return len(self.data)


def prepare_dataloaders(train_tokens, test_tokens, domain_tokens, mask_idx,
                        padding_idx, batch_size, num_workers=4):
    """
    Create PyTorch DataLoaders for training and validation with MLM collation.

    This function wraps the tokenized sequences in Dataset objects and creates DataLoaders
    with the appropriate collate function for masked language modeling. The collate function
    handles batching, padding, and masking operations.

    Args:
        train_tokens (List[torch.Tensor]): Training sequences as token ID tensors
        test_tokens (List[torch.Tensor]): Test/validation sequences as token ID tensors
        domain_tokens (np.ndarray): Array of domain token IDs for random masking
        mask_idx (int): Token ID used for masking positions
        padding_idx (int): Token ID used for sequence padding
        batch_size (int): Number of sequences per batch
        num_workers (int, optional): Number of worker processes for data loading (default: 4)

    Returns:
        tuple: (dl_train, dl_test, ds_train, ds_test) where:
            - dl_train (DataLoader): Training data loader with shuffling enabled
            - dl_test (DataLoader): Test data loader without shuffling
            - ds_train (ListDataset): Training dataset object
            - ds_test (ListDataset): Test dataset object

    DataLoader Configuration:
        - Training: shuffle=True for randomized batch ordering
        - Test: shuffle=False for deterministic evaluation
        - Both use custom collate_wrapper for MLM processing
        - Parallel loading with configurable num_workers

    Note:
        The collate_wrapper function encapsulates the MLM-specific parameters
        (domain_tokens, mask_idx, padding_idx) for use by the DataLoader.
    """
    ds_train = ListDataset(train_tokens)
    ds_test = ListDataset(test_tokens)

    def collate_wrapper(batch):
        return mlm_collate_fn(batch, domain_tokens, mask_idx, padding_idx)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=collate_wrapper)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_wrapper)
    return dl_train, dl_test, ds_train, ds_test


def maybe_load_pretrained_embeddings(args, model, data_fpath, specials, domains, domain_tokens):
    """
    Load and initialize pre-trained ESM embeddings into the model embedding layer.

    This function handles loading ESM (Evolutionary Scale Modeling) embeddings and integrating
    them into the BiGCARP model. Supports both frozen (non-trainable) and fine-tunable
    embedding initialization strategies.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - pretrain (bool): If True, load embeddings but allow training updates
            - freeze (bool): If True, load embeddings and freeze them during training
            - esm_emb_fpath (str): Path to ESM embeddings tensor file (.pt format)
            - fpfams_fpath (str): Path to Pfam domain FASTA file for alignment (optional)
        model (ByteNetLM): BiGCARP model with embedding layer to initialize
        data_fpath (str): Path to pre-trained data (currently unused)
        specials (Dict[str, int]): Special token mappings
        domains (Dict[str, int]): Domain name to token ID mappings
        domain_tokens (np.ndarray): Array of domain token IDs

    Returns:
        int or None: Number of frozen embeddings if freeze mode is used, None otherwise

    Embedding Integration Strategies:
        1. Freeze Mode (--freeze):
           - Embeddings are loaded into a separate frozen parameter
           - No gradient updates during training
           - Preserves original ESM representations

        2. Pretrain Mode (--pretrain):
           - Embeddings initialize the main embedding layer
           - Allows fine-tuning during training
           - Offset by special tokens + UNK token count

    Note:
        - ESM embeddings are expected as a PyTorch tensor
        - Domain alignment via fpfams_fpath is currently commented out
        - Special tokens and UNK are excluded from pre-trained initialization
    """
    if args.pretrain or args.freeze:
        # Load the pre-trained ESM embeddings
        esm_embeddings = torch.load(args.esm_emb_fpath)
        # Parse final_pfams from args.fpfams_fpath

        # !! unquote this for not clean dataset, emb is not aligned

        # make sure the embedding follows the the order of parse_fasta
        # assume the model follows the order of the parse_fasta generated domains
        # in practice these wont change the order.
        # pfams, names = parse_fasta(args.fpfams_fpath, return_names=True)
        # names = [name.split(';')[-2] for name in names]
        # new_idx = np.array([domains[name] for name in names]) - min(domain_tokens) - 1


        # esm_embeddings = esm_embeddings[new_idx]
        n_frozen_embs = len(esm_embeddings)
    else:
        esm_embeddings = None
        n_frozen_embs = None

    # If freeze, set the model to contain an extra embedding table for the frozen portion
    if args.freeze and esm_embeddings is not None:
        with torch.no_grad():
            model.embedder.embedder.frozen.weight = torch.nn.Parameter(torch.tensor(esm_embeddings))
    elif args.pretrain and esm_embeddings is not None:
        with torch.no_grad():
            # Use the count of special tokens + 1 as an offset (to exclude UNK as well)
            offset = len(specials) + 1 
            model.embedder.embedder.weight[offset:] = torch.nn.Parameter(esm_embeddings.clone().detach())

    return n_frozen_embs


def create_model(args, n_tokens, mask_idx, n_frozen_embs=None):
    """
    Instantiate and configure a ByteNetLM model for BiGCARP training.

    This function creates the core BiGCARP model using the ByteNet architecture with
    specified hyperparameters. The model supports both standard and frozen embedding
    configurations for transfer learning scenarios.

    Args:
        args (argparse.Namespace): Training configuration containing:
            - d_embedding (int): Embedding layer dimension (e.g., 1280 for ESM compatibility)
            - d_model (int): Hidden dimension within ByteNet layers
            - n_layers (int): Number of ByteNet convolutional layers
            - kernel_size (int): Convolution kernel width
            - r (int): Dilation rate parameter for temporal modeling
            - wide (bool): If True, use wide ByteNet variant; else slim variant
            - ar (bool): If True, use autoregressive (causal) masking
        n_tokens (int): Total vocabulary size including special tokens and domains
        mask_idx (int): Token ID used as padding index in the model
        n_frozen_embs (int, optional): Number of frozen embeddings for transfer learning

    Returns:
        ByteNetLM: Configured BiGCARP model ready for training with the following architecture:
            - Embedding layer (with optional frozen component)
            - Multi-layer ByteNet encoder with dilated convolutions
            - Final layer normalization
            - GELU activation functions
            - Output projection to vocabulary size

    Model Configuration:
        - Padding index set to mask_idx (BiGCARP-specific implementation)
        - Final layer normalization enabled for training stability
        - GELU activation for improved gradient flow
        - Causal masking configurable via args.ar
        - Slim vs wide architecture determined by args.wide

    Note:
        The model uses mask_idx as padding_idx following BiGCARP implementation conventions.
        This differs from standard practice where padding and masking indices are separate.
    """
    model = ByteNetLM(
        n_tokens=n_tokens,
        d_embedding=args.d_embedding,
        d_model=args.d_model,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        r=args.r,
        slim=(not args.wide),
        padding_idx=mask_idx, # in bigcarp implementation, the padding index is the same as the mask index
        causal=args.ar,
        final_ln=True,
        activation='gelu',
        n_frozen_embs=n_frozen_embs
    )
    return model


def load_checkpoint_if_exists(args, model, optimizer):
    """
    Load model and optimizer state from checkpoint for training resumption.

    This function handles checkpoint loading when --restart flag is specified, allowing
    training to resume from the last saved state. It restores model parameters, optimizer
    state, and training progress tracking variables.

    Args:
        args (argparse.Namespace): Training arguments containing:
            - restart (bool): If True, attempt to load checkpoint
            - out_fpath (str): Primary output directory for checkpoints
            - cp_fpath (str, optional): Specific checkpoint directory path
        model (ByteNetLM): Model instance to load state into
        optimizer (torch.optim.Optimizer): Optimizer to restore state

    Returns:
        tuple: (initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch) where:
            - initial_epoch (int): Epoch number to resume training from (0 if no checkpoint)
            - total_steps (int): Total training steps completed (0 if no checkpoint)
            - best_val_loss (float): Best validation loss achieved (inf if no checkpoint)
            - best_val_acc (float): Best validation accuracy achieved (0.0 if no checkpoint)
            - best_epoch (int): Epoch of best model (-1 if no checkpoint)

    Checkpoint Loading Logic:
        1. Skip loading if --restart not specified
        2. Check cp_fpath first, then out_fpath for checkpoint directory
        3. Look specifically for 'checkpoint_latest.tar' file
        4. Load model_state_dict and optimizer_state_dict
        5. Extract training progress metadata
        6. Handle loading failures gracefully

    Error Handling:
        - Missing directories or files: Start training from scratch
        - Corrupted checkpoints: Log error and start fresh
        - State dict mismatches: Allow PyTorch to handle compatibility

    Note:
        Only loads from 'checkpoint_latest.tar' to ensure resuming from most recent state.
        Best model metrics are restored to maintain proper model selection during training.
    """
    initial_epoch = 0
    total_steps = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = -1

    if not args.restart:
        return initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch

    # Use cp_fpath if provided, otherwise out_fpath
    cp_dir = args.cp_fpath if args.cp_fpath else args.out_fpath
    if not os.path.exists(cp_dir):
        print(f"Checkpoint directory {cp_dir} does not exist, starting from scratch.")
        return initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch

    # Check for latest checkpoint only
    latest_ckpt = os.path.join(cp_dir, 'checkpoint_latest.tar')
    if not os.path.exists(latest_ckpt):
        print(f"No checkpoint_latest.tar found in {cp_dir}, starting from scratch.")
        return initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch

    print(f"Loading checkpoint from {latest_ckpt}")

    try:
        sd = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(sd['model_state_dict'])
        optimizer.load_state_dict(sd['optimizer_state_dict'])

        # Extract information from checkpoint
        total_steps = sd.get('step', 0)
        initial_epoch = sd.get('epoch', 0) + 1

        # Load best model metrics if available
        best_val_loss = sd.get('val_loss', float('inf'))
        best_val_acc = sd.get('val_acc', 0.0)
        best_epoch = sd.get('best_epoch', -1)

        print(f"Resuming from epoch {initial_epoch}, step {total_steps}")
        if best_epoch >= 0:
            print(f"Best model so far: loss={best_val_loss:.4f}, acc={best_val_acc:.4f}, epoch={best_epoch+1}")

    except Exception as e:
        print(f"Failed to load checkpoint {latest_ckpt}: {e}")
        print("Starting from scratch.")
        return 0, 0, float('inf'), 0.0, -1

    return initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch


def train_one_epoch(
    model, device, dl_train, optimizer, loss_func, accu_func,
    e, epochs, total_steps, args
):
    """
    Execute one complete training epoch with masked language modeling.

    This function performs a full training pass over the training dataset, including
    forward propagation, loss computation, backpropagation, and parameter updates.
    Progress tracking and metrics logging are included.

    Args:
        model (ByteNetLM): BiGCARP model in training mode
        device (torch.device): CUDA device for tensor operations
        dl_train (DataLoader): Training data loader with MLM collation
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        loss_func (MaskedCrossEntropyLoss): Loss function for MLM training
        accu_func (MaskedAccuracy): Accuracy metric for MLM evaluation
        e (int): Current epoch number (0-indexed)
        epochs (int): Total number of training epochs
        total_steps (int): Cumulative training steps from all previous epochs
        args (argparse.Namespace): Training configuration containing:
            - mask_idx (int): Token ID for masking
            - out_fpath (str): Output directory for metrics logging

    Returns:
        tuple: (total_steps, avg_loss, avg_accu) where:
            - total_steps (int): Updated total step count after this epoch
            - avg_loss (float): Average training loss across all batches
            - avg_accu (float): Average training accuracy across all batches

    Training Process:
        1. Set model to training mode
        2. Process each batch: src (masked input), tgt (original), mask (positions)
        3. Create input_mask for non-masked positions
        4. Forward pass through model
        5. Compute MLM loss and accuracy
        6. Backpropagate gradients and update parameters
        7. Track batch-wise metrics weighted by masked positions
        8. Log progress every 10 batches

    Metrics Calculation:
        - Loss and accuracy weighted by number of masked positions per batch
        - Ensures equal contribution from all masked tokens regardless of sequence length
        - Final averages computed over total masked positions, not batches

    Logging:
        - Progress bar with running loss and accuracy
        - Epoch summary with timing information
        - CSV metrics file with format: "train,loss,accuracy,epoch,total_steps"

    Note:
        Input masking excludes positions with mask_idx from attention computation.
        This differs from target masking used in loss calculation.
    """
    model.train()
    start_time = datetime.now()

    losses, accuracies, counts = [], [], []
    n_total = len(dl_train.dataset)

    progress_bar = tqdm(dl_train, desc=f"Training Epoch {e+1}/{epochs}", leave=True, unit="batch")
    
    # Collect epoch-wide metrics
    epoch_loss = 0.0
    epoch_acc = 0.0
    total_masked_positions = 0
    
    batch_count = 0
    display_interval = 10  # Log stats every 10 batches

    for i, batch in enumerate(progress_bar):
        # Unpack batch
        src, tgt, mask = [b.to(device) for b in batch]
        input_mask = (src != args.mask_idx).float()

        # Forward pass
        outputs = model(src, input_mask=input_mask.unsqueeze(-1))
        loss = loss_func(outputs, tgt, mask)
        accu = accu_func(outputs, tgt, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        masked_positions = mask.sum().item()
        losses.append(loss.item() * masked_positions)
        accuracies.append(accu.item() * masked_positions)
        counts.append(masked_positions)
        total_masked_positions += masked_positions

        total_steps += 1
        avg_loss = sum(losses) / sum(counts)
        avg_accu = sum(accuracies) / sum(counts)
        
        batch_count += 1
        
        # Less frequent progress bar updates
        if batch_count % display_interval == 0 or batch_count == len(dl_train):
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{avg_accu:.4f}"
            })

        epoch_loss += loss.item() * masked_positions
        epoch_acc += accu.item() * masked_positions

    # Final metrics for the epoch
    avg_loss = epoch_loss / sum(counts)
    avg_accu = epoch_acc / sum(counts)
    
    print(f"\nTraining Epoch {e+1}/{epochs} completed in {datetime.now() - start_time}")
    print(f"  Loss: {avg_loss:.4f} | Accuracy: {avg_accu:.4f} | Total Steps: {total_steps}")

    # Write training metrics
    metrics_path = os.path.join(args.out_fpath, 'metrics.csv')
    with open(metrics_path, 'a') as f:
        # store train loss/accu with label "train"
        f.write(f"train,{avg_loss},{avg_accu},{e},{total_steps}\n")

    return total_steps, avg_loss, avg_accu


def test_one_epoch(
    model, device, dl_test, loss_func, accu_func,
    e, epochs, total_steps, args
):
    """
    Execute one complete validation epoch for model evaluation.

    This function performs a full validation pass over the test dataset without
    gradient updates. Used to monitor model performance, detect overfitting,
    and select the best model during training.

    Args:
        model (ByteNetLM): BiGCARP model in evaluation mode
        device (torch.device): CUDA device for tensor operations
        dl_test (DataLoader): Validation data loader with MLM collation
        loss_func (MaskedCrossEntropyLoss): Loss function for MLM evaluation
        accu_func (MaskedAccuracy): Accuracy metric for MLM evaluation
        e (int): Current epoch number (0-indexed)
        epochs (int): Total number of training epochs
        total_steps (int): Cumulative training steps (for logging purposes)
        args (argparse.Namespace): Training configuration containing:
            - mask_idx (int): Token ID for masking
            - out_fpath (str): Output directory for metrics logging

    Returns:
        tuple: (avg_loss, avg_accu) where:
            - avg_loss (float): Average validation loss across all batches
            - avg_accu (float): Average validation accuracy across all batches
            Returns (None, None) if no test data is processed

    Evaluation Process:
        1. Set model to evaluation mode (disables dropout, batch norm updates)
        2. Disable gradient computation for efficiency
        3. Process each batch: src (masked input), tgt (original), mask (positions)
        4. Create input_mask for non-masked positions
        5. Forward pass through model
        6. Compute MLM loss and accuracy
        7. Track batch-wise metrics weighted by masked positions
        8. Log progress every 30 batches

    Metrics Calculation:
        - Loss and accuracy weighted by number of masked positions per batch
        - Ensures equal contribution from all masked tokens regardless of sequence length
        - Final averages computed over total masked positions, not batches

    Logging:
        - Progress bar with running loss and accuracy
        - Epoch summary with timing information
        - CSV metrics file with format: "test,loss,accuracy,epoch,total_steps"

    Note:
        This function does not update model parameters. All computations are
        performed within torch.no_grad() context for memory efficiency.
    """
    model.eval()
    start_time = datetime.now()

    losses, accuracies, counts = [], [], []
    n_total = len(dl_test.dataset)

    progress_bar = tqdm(dl_test, desc=f"Validation Epoch {e+1}/{epochs}", leave=True, unit="batch")
    
    batch_count = 0
    display_interval = 30  # Log stats every 30 batches

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            src, tgt, mask = [b.to(device) for b in batch]
            input_mask = (src != args.mask_idx).float()

            outputs = model(src, input_mask=input_mask.unsqueeze(-1))
            loss = loss_func(outputs, tgt, mask)
            accu = accu_func(outputs, tgt, mask)

            # Track metrics
            masked_positions = mask.sum().item()
            losses.append(loss.item() * masked_positions)
            accuracies.append(accu.item() * masked_positions)
            counts.append(masked_positions)
            
            batch_count += 1

            # Less frequent progress bar updates
            if batch_count % display_interval == 0 or batch_count == len(dl_test):
                avg_loss = sum(losses) / sum(counts) if sum(counts) > 0 else 0
                avg_accu = sum(accuracies) / sum(counts) if sum(counts) > 0 else 0

                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'acc': f"{avg_accu:.4f}"
                })

    if not counts: 
        print("No test data processed, skipping metrics write.")
        return None, None

    avg_loss = sum(losses) / sum(counts)
    avg_accu = sum(accuracies) / sum(counts)
    
    print(f"\nValidation Epoch {e+1}/{epochs} completed in {datetime.now() - start_time}")
    print(f"  Loss: {avg_loss:.4f} | Accuracy: {avg_accu:.4f}")

    metrics_path = os.path.join(args.out_fpath, 'metrics.csv')
    with open(metrics_path, 'a') as f:
        # store test loss/accu with label "test"
        f.write(f"test,{avg_loss},{avg_accu},{e},{total_steps}\n")
    
    return avg_loss, avg_accu


def save_checkpoint(model, optimizer, epoch, total_steps, args, checkpoint_type, val_loss=None, val_acc=None):
    """
    Save model and optimizer state with different naming conventions based on checkpoint type.

    This function handles the creation and saving of training checkpoints, supporting
    multiple checkpoint types for different use cases: best model selection, training
    resumption, and periodic backups.

    Args:
        model (ByteNetLM): BiGCARP model to save state from
        optimizer (torch.optim.Optimizer): Optimizer to save state from
        epoch (int): Current epoch number (0-indexed)
        total_steps (int): Total training steps completed across all epochs
        args (argparse.Namespace): Training configuration containing:
            - out_fpath (str): Output directory for checkpoint files
        checkpoint_type (str): Type of checkpoint to save. Options:
            - 'best': Best validation performance model (checkpoint_best.tar)
            - 'latest': Most recent training state (checkpoint_latest.tar)
            - 'periodic': Regular backup (checkpoint_epoch{N}.tar)
        val_loss (float, optional): Current validation loss for best model tracking
        val_acc (float, optional): Current validation accuracy for best model tracking

    Checkpoint Contents:
        All checkpoints contain:
        - 'step' (int): Total training steps completed
        - 'epoch' (int): Current epoch number
        - 'model_state_dict' (dict): Complete model parameters and buffers
        - 'optimizer_state_dict' (dict): Optimizer state including momentum terms
        - 'val_loss' (float): Validation loss if provided
        - 'val_acc' (float): Validation accuracy if provided

    Checkpoint Types:
        1. Best Checkpoint ('best'):
           - Saved when validation performance improves
           - Used for final model selection and inference
           - Filename: checkpoint_best.tar

        2. Latest Checkpoint ('latest'):
           - Updated every epoch
           - Used for training resumption with --restart
           - Filename: checkpoint_latest.tar

        3. Periodic Checkpoint ('periodic'):
           - Saved at regular intervals (e.g., every 5 epochs)
           - Provides backup points during long training
           - Filename: checkpoint_epoch{epoch}.tar

    Error Handling:
        - Retry logic: Attempts saving up to 10 times with 1-second delays
        - Handles temporary filesystem issues and disk space problems
        - Logs success/failure messages for debugging

    Note:
        Checkpoints are saved in CPU format for cross-device compatibility.
        The retry mechanism handles common filesystem race conditions during
        high-frequency saving operations.
    """
    checkpoint_data = {
        'step': total_steps,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Add validation metrics for best checkpoint tracking
    if val_loss is not None:
        checkpoint_data['val_loss'] = val_loss
    if val_acc is not None:
        checkpoint_data['val_acc'] = val_acc
    
    # Determine filename based on checkpoint type
    if checkpoint_type == 'best':
        filename = "checkpoint_best.tar"
    elif checkpoint_type == 'latest':
        filename = "checkpoint_latest.tar"
    elif checkpoint_type == 'periodic':
        filename = f"checkpoint_epoch{epoch}.tar"
    else:
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")
    
    checkpoint_path = os.path.join(args.out_fpath, filename)
    
    # Save with retry logic
    for attempt in range(10):
        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved {checkpoint_type} checkpoint: {filename}")
            break
        except OSError:
            time.sleep(1)
    else:
        print(f'Failed to save {checkpoint_type} checkpoint after 10 attempts!')


def main():
    """
    Main training pipeline for BiGCARP masked language modeling.

    This function orchestrates the complete training process for the BiGCARP model,
    including data loading, model initialization, training loop execution, and
    results visualization. Supports both fresh training and resumption from checkpoints.

    Training Pipeline:
        1. Parse command-line arguments and create timestamped output directory
        2. Load and tokenize training corpus with vocabulary mapping
        3. Prepare PyTorch DataLoaders with MLM collation
        4. Initialize ByteNetLM model with optional pre-trained embeddings
        5. Setup optimizer and load checkpoints if resuming training
        6. Execute training loop with validation and checkpoint saving
        7. Generate training curves and save final embedding weights

    Key Features:
        - Automatic checkpoint management (best, latest, periodic)
        - Pre-trained embedding integration (ESM embeddings)
        - Comprehensive metrics logging and visualization
        - Training resumption with --restart flag
        - GPU memory optimization and error handling

    Output Files:
        - checkpoint_best.tar: Best validation performance model
        - checkpoint_latest.tar: Most recent training state
        - checkpoint_epoch{N}.tar: Periodic backups every 5 epochs
        - metrics.csv: Training and validation metrics per epoch
        - loss_plot.png: Training and validation loss curves
        - final_embedder.pt: Final learned embedding matrix

    Configuration:
        All training parameters are configurable via command-line arguments.
        See get_parser() for complete list of available options including
        model architecture, training hyperparameters, and I/O paths.

    Error Handling:
        - Graceful handling of missing files and directories
        - Automatic fallback for failed checkpoint operations
        - Comprehensive logging for debugging training issues

    Note:
        Creates timestamped subdirectories to prevent output conflicts
        when running multiple training experiments.
    """
    # -------------------- 1. Parse arguments --------------------
    parser = get_parser()
    args = parser.parse_args()

    # Create a new output folder named with current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.out_fpath = os.path.join(args.out_fpath, f"run_{timestamp}")
    os.makedirs(args.out_fpath, exist_ok=True)

    # -------------------- 2. Load data --------------------------
    (train_tokens, test_tokens,
     specials, domains, domain_tokens,
     n_tokens, padding_idx, mask_idx,
     data_fpath, df) = load_data(args)
    args.mask_idx = mask_idx
    args.padding_idx = padding_idx

    print(f"{len(train_tokens)} training sequences")
    print(f"{len(test_tokens)} test sequences")

    # -------------------- 3. Prepare dataloaders ----------------
    dl_train, dl_test, ds_train, ds_test = prepare_dataloaders(
        train_tokens, test_tokens, domain_tokens, mask_idx,
        padding_idx, batch_size=args.batch_size
    )

    # -------------------- 4. Build and/or load model ------------
    n_frozen_embs = None
    esm_embeddings = None
    if args.pretrain or args.freeze:
        print(f"Loading pretrained embeddings from {args.esm_emb_fpath}")
        esm_embeddings = torch.load(args.esm_emb_fpath)
        if args.freeze:
            n_frozen_embs = len(esm_embeddings)

    model = create_model(args, n_tokens, mask_idx, n_frozen_embs=n_frozen_embs)
    
    if esm_embeddings is not None:
        with torch.no_grad():
            if args.freeze:
                model.embedder.embedder.frozen.weight.copy_(esm_embeddings.clone().detach())
            elif args.pretrain:
                # Offset by the number of special tokens plus one for the UNK token
                offset = len(specials) + 1
                model.embedder.embedder.weight[offset:].copy_(esm_embeddings.clone().detach())

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    model.to(device)

    # -------------------- 5. Optimizer --------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -------------------- 6. Save initial checkpoint ------------
    checkpoint_fname = os.path.join(args.out_fpath, "checkpoint00.tar")
    torch.save({
        'step': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_fname)
    print(f"Initial checkpoint saved at {checkpoint_fname}")

    # -------------------- 7. Load Checkpoints if exists ------------------------
    initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch = load_checkpoint_if_exists(args, model, optimizer)

    # -------------------- 8. Training Loop ----------------------
    loss_func = MaskedCrossEntropyLoss()
    accu_func = MaskedAccuracy()

    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"{n_parameters} total model parameters.")

    for e in range(initial_epoch, args.epochs):
        # --- Train ---
        total_steps, train_loss, train_acc = train_one_epoch(
            model=model,
            device=device,
            dl_train=dl_train,
            optimizer=optimizer,
            loss_func=loss_func,
            accu_func=accu_func,
            e=e,
            epochs=args.epochs,
            total_steps=total_steps,
            args=args
        )

        # --- Test ---
        val_loss, val_acc = test_one_epoch(
            model=model,
            device=device,
            dl_test=dl_test,
            loss_func=loss_func,
            accu_func=accu_func,
            e=e,
            epochs=args.epochs,
            total_steps=total_steps,
            args=args
        )
        
        # --- Checkpoint Saving ---
        # 1. Save latest checkpoint
        save_checkpoint(model, optimizer, e, total_steps, args, 'latest', val_loss, val_acc)
        
        # 2. Save best checkpoint if this is the best validation performance
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = e
            save_checkpoint(model, optimizer, e, total_steps, args, 'best', val_loss, val_acc)
            print(f"New best model at epoch {e+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # 3. Save periodic checkpoint every 5 epochs
        if (e + 1) % 5 == 0:
            save_checkpoint(model, optimizer, e, total_steps, args, 'periodic', val_loss, val_acc)

    # Print best model summary
    if best_epoch >= 0:
        print(f"\nBest model found at epoch {best_epoch+1}:")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")

    # -------------------- 9. Generate loss plot ----------------------
    metrics_csv = os.path.join(args.out_fpath, 'metrics.csv')
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv, header=None)
        df_metrics.columns = ['split', 'loss', 'accu', 'epoch', 'steps']
        # filter and plot each split
        train_data = df_metrics[df_metrics['split'] == 'train']
        test_data  = df_metrics[df_metrics['split'] == 'test']
        plt.figure()
        plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss')
        plt.plot(test_data['epoch'], test_data['loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.out_fpath, 'loss_plot.png'))
        plt.close()

    # -------------------- 10. Save the final embeddings matrix ----------------
    # this is the static embeddings in the embed layer (not contextualized)
    with torch.no_grad():
        final_embs = model.embedder.embedder.weight.detach().cpu()
    torch.save(final_embs, os.path.join(args.out_fpath, 'final_embedder.pt'))
    print("Done")

if __name__ == '__main__':
    main()
