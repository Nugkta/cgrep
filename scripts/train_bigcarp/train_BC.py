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
from tqdm.auto import tqdm  # Add this import for progress bars

# ----------- Custom imports from your own modules -----------
from sequence_models.convolutional import ByteNetLM
from sequence_models.metrics import MaskedAccuracy
from sequence_models.losses import MaskedCrossEntropyLoss
from sequence_models.collaters import _pad
from sequence_models.utils import parse_fasta
# -------------------------------------------------------------

# Helper function to determine if output is going to a file
def is_output_redirected():
    return not sys.stdout.isatty()

# Configure tqdm settings based on output destination
def get_tqdm_config():
    if is_output_redirected():
        # If outputting to a file or pipe, use these settings
        return {
            'disable': False,
            'leave': False,
            'ncols': 80,
            'mininterval': 5.0,      # Update at most every 5 seconds
            'miniters': 20,          # Update at most every 20 iterations
            'bar_format': '{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        }
    else:
        # If outputting to terminal, use these settings
        return {
            'leave': True,
            'unit': "batch",
        }

def get_parser():
    """
    Creates an argument parser for hyperparameters and I/O configurations.
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
    parser.add_argument('--unconditional', action='store_true',
                        help='If set, do not prepend a special function token.')
    parser.add_argument('--ar', action='store_true',
                        help='If set, use autoregressive model (causal).')
    parser.add_argument('--cp_fpath', type=str, default=None,
                        help='Path to the checkpoint to load for training.')
    
    
    return parser


def load_data(args):
    """
    Loads and prepares the CSV data, domain-to-token dictionary, and splits into train/test.

    Returns:
        train_tokens, test_tokens, specials, domains, domain_tokens, n_tokens, padding_idx, mask_idx
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
        if args.unconditional:
            t = []
        else:
            # Prepend the function token if not unconditional
            t = [specials[row['function']]]

        for d in row['domains'].split(';'):
            if d in domains:
                t.append(domains[d])
            else:
                # Use 'UNK' domain if the domain token is not found
                t.append(domains['UNK'])
        tokens_list.append(torch.tensor(t))

    # Split the data into train, test
    train_tokens = [tokens_list[i] for i in df[df['split'] == 'train'].index]
    test_tokens  = [tokens_list[i] for i in df[df['split'] == 'test'].index]

    data_fpath = args.fdata

    return (train_tokens, test_tokens,
            specials, domains, domain_tokens,
            n_tokens, padding_idx, mask_idx, data_fpath, df)


def mlm_collate_fn(batch, domain_tokens, mask_idx, padding_idx):
    """
    Collate function for masked language modeling.

    Args:
        batch: A batch of token sequences.
        domain_tokens: Array of possible domain tokens.
        mask_idx: Special index used for masking in the vocabulary.
        padding_idx: Special index used for padding.

    Returns:
        src: Padded input sequences with random masking (15%).
        tgt: Padded original (target) sequences.
        mask: Padded mask (1 for masked tokens, 0 for not).
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
    Simple PyTorch Dataset that wraps a list of tensors.
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
    Prepare PyTorch DataLoaders for train and test splits.
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

# don't need the pretraining for this purpose

def maybe_load_pretrained_embeddings(args, model, data_fpath, specials, domains, domain_tokens):
    """
    If pretrain or freeze is specified, loads ESM embeddings and places them into the model.
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
    Creates the ByteNetLM model based on arguments and returns it.
    """
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
        n_frozen_embs=n_frozen_embs
    )
    return model


def load_checkpoint_if_exists(args, model, optimizer):
    """
    If --restart is set, attempts to load the latest checkpoint from out_fpath.
    Returns the initial_epoch, total_steps, and best model metrics from the checkpoint.
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

    # Priority order: latest checkpoint, then best checkpoint, then periodic checkpoints
    checkpoint_candidates = []
    
    # Check for latest checkpoint (highest priority)
    latest_ckpt = os.path.join(cp_dir, 'checkpoint_latest.tar')
    if os.path.exists(latest_ckpt):
        checkpoint_candidates.append(('latest', latest_ckpt))
    
    # Check for best checkpoint
    best_ckpt = os.path.join(cp_dir, 'checkpoint_best.tar')
    if os.path.exists(best_ckpt):
        checkpoint_candidates.append(('best', best_ckpt))
    
    # Check for periodic checkpoints (legacy and new format)
    all_files = [f for f in os.listdir(cp_dir) if f.startswith('checkpoint') and f.endswith('.tar')]
    periodic_checkpoints = []
    
    for fname in all_files:
        if fname.startswith('checkpoint_epoch'):
            # New format: checkpoint_epoch{N}.tar
            epoch_str = fname.replace('checkpoint_epoch', '').replace('.tar', '')
            try:
                epoch_num = int(epoch_str)
                periodic_checkpoints.append((epoch_num, os.path.join(cp_dir, fname)))
            except ValueError:
                continue
        elif fname.startswith('checkpoint') and fname not in ['checkpoint_latest.tar', 'checkpoint_best.tar']:
            # Legacy format: checkpoint{N}.tar
            epoch_str = fname.replace('checkpoint', '').replace('.tar', '')
            try:
                epoch_num = int(epoch_str)
                periodic_checkpoints.append((epoch_num, os.path.join(cp_dir, fname)))
            except ValueError:
                continue
    
    # Add the most recent periodic checkpoint as a candidate
    if periodic_checkpoints:
        periodic_checkpoints.sort(key=lambda x: x[0], reverse=True)  # Sort by epoch descending
        checkpoint_candidates.append(('periodic', periodic_checkpoints[0][1]))

    if not checkpoint_candidates:
        print(f"No valid checkpoints found in {cp_dir}, starting from scratch.")
        return initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch

    # Load the highest priority checkpoint
    checkpoint_type, ckpt_path = checkpoint_candidates[0]
    print(f"Loading {checkpoint_type} checkpoint from {ckpt_path}")
    
    try:
        sd = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(sd['model_state_dict'])
        optimizer.load_state_dict(sd['optimizer_state_dict'])
        
        # Extract information from checkpoint
        total_steps = sd.get('step', 0)
        if 'epoch' in sd:
            initial_epoch = sd['epoch'] + 1
        else:
            # Legacy checkpoint format - try to extract from filename
            if 'epoch' in os.path.basename(ckpt_path):
                epoch_str = os.path.basename(ckpt_path).replace('checkpoint_epoch', '').replace('checkpoint', '').replace('.tar', '')
                try:
                    initial_epoch = int(epoch_str) + 1
                except ValueError:
                    initial_epoch = 0
        
        # Try to load best model metrics if available
        if checkpoint_type == 'best' or 'val_loss' in sd:
            best_val_loss = sd.get('val_loss', float('inf'))
            best_val_acc = sd.get('val_acc', 0.0)
            best_epoch = sd.get('epoch', -1)
            print(f"Loaded best model metrics: loss={best_val_loss:.4f}, acc={best_val_acc:.4f}, epoch={best_epoch+1}")
            
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        print("Starting from scratch.")
        return 0, 0, float('inf'), 0.0, -1

    return initial_epoch, total_steps, best_val_loss, best_val_acc, best_epoch


def train_one_epoch(
    model, device, dl_train, optimizer, loss_func, accu_func,
    e, epochs, total_steps, args
):
    """
    Runs one training epoch. Returns the updated total_steps and epoch metrics.
    """
    model.train()
    start_time = datetime.now()

    losses, accuracies, counts = [], [], []
    n_total = len(dl_train.dataset)
    
    # Use optimized progress bar settings
    tqdm_config = get_tqdm_config()
    progress_bar = tqdm(dl_train, desc=f"Training Epoch {e+1}/{epochs}", **tqdm_config)
    
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
            
            # For redirected output, periodically print summary instead of relying on the progress bar
            if is_output_redirected() and (batch_count % (display_interval * 10) == 0):
                print(f"Training Batch {batch_count}/{len(dl_train)} | Loss: {avg_loss:.4f} | Acc: {avg_accu:.4f}")

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
    Runs one validation epoch. Returns validation loss and accuracy.
    """
    model.eval()
    start_time = datetime.now()

    losses, accuracies, counts = [], [], []
    n_total = len(dl_test.dataset)
    
    # Use optimized progress bar settings
    tqdm_config = get_tqdm_config()
    progress_bar = tqdm(dl_test, desc=f"Validation Epoch {e+1}/{epochs}", **tqdm_config)
    
    batch_count = 0
    display_interval = 10  # Log stats every 10 batches

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
                
                # For redirected output, periodically print summary instead of relying on the progress bar
                if is_output_redirected() and (batch_count % (display_interval * 10) == 0):
                    print(f"Validation Batch {batch_count}/{len(dl_test)} | Loss: {avg_loss:.4f} | Acc: {avg_accu:.4f}")

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
    Save checkpoint with different naming conventions based on type.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        total_steps: Total training steps
        args: Arguments object
        checkpoint_type: 'best', 'latest', or 'periodic'
        val_loss: Validation loss (for best model tracking)
        val_acc: Validation accuracy (for best model tracking)
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

    # -------------------- 7. Checkpoints ------------------------
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
        # 1. Always save latest checkpoint (for recovery)
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

    # -------------------- 10. Save the final embeddings ----------------
    # this is the static embeddings in the embed layer (not contextualized)
    with torch.no_grad():
        final_embs = model.embedder.embedder.weight.detach().cpu()
    torch.save(final_embs, os.path.join(args.out_fpath, 'final_embedder.pt'))
    print("All done!")

if __name__ == '__main__':
    main()
