"""
Shared functions for BigCARP model training and analysis.

This module contains functions extracted from the training scripts that are commonly
used across different analysis scripts.
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sequence_models.collaters import _pad


def load_data(args):
    """
    Loads and prepares the CSV data, domain-to-token dictionary, and splits into train/test.

    Args:
        args: Arguments object with fcorpus, fvocab, and unconditional attributes

    Returns:
        tuple: (train_tokens, test_tokens, specials, domains, domain_tokens,
                n_tokens, padding_idx, mask_idx, data_fpath, df)
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
    test_tokens = [tokens_list[i] for i in df[df['split'] == 'test'].index]

    data_fpath = getattr(args, 'fdata', None)

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
        tuple: (src, tgt, mask) - Padded input sequences with random masking (15%),
               padded original (target) sequences, and padded mask (1 for masked tokens, 0 for not).
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

    Args:
        train_tokens: List of training token sequences
        test_tokens: List of test token sequences
        domain_tokens: Array of domain tokens for masking
        mask_idx: Index for mask token
        padding_idx: Index for padding token
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders

    Returns:
        tuple: (dl_train, dl_test, ds_train, ds_test) - Train and test dataloaders and datasets
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