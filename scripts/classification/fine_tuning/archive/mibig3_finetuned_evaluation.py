#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIBiG 3.0 Fine-tuned BigCarp Model Evaluation with Stratified Cross-Validation
==============================================================================
This script evaluates fine-tuned BigCarp models on MIBiG 3.0 data using the exact same
5-fold stratified CV pipeline as mibig3_stratified_evaluation.py for fair comparison.

Models evaluated:
- BigCarp Random (fine-tuned)
- BigCarp ESM1b Fine-tuned

This allows direct comparison with the embedding-based approaches.
"""

import os, sys, json, argparse, pathlib, random, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             roc_auc_score, hamming_loss)
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Add paths for BigCarp imports
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

from sequence_models.convolutional import ByteNetLM
from sequence_models.collaters import _pad

# Import evaluation functions from the main script for consistency
from scripts.classification.mibig3_stratified_evaluation import (
    convert_product_classes_to_binary,
    create_stratified_splits,
    compute_comprehensive_metrics,
    create_comparison_table,
    exact_match_accuracy
)

# Try stratified multilabel CV
try:
    from skmultilearn.model_selection import IterativeStratification
    STRATIFIED_AVAILABLE = True
    print("✅ scikit-multilearn available for stratified CV")
except ImportError:
    STRATIFIED_AVAILABLE = False
    print("⚠️  scikit-multilearn not available. Install with: pip install scikit-multilearn")

# -------- Data Loading Functions -------------------------------------------
def load_mibig3_data(artifacts_dir="artifacts/classification/mibig3"):
    """Load MIBiG 3.0 data for fine-tuning evaluation."""
    print("📂 Loading MIBiG 3.0 data...")
    
    # Load preprocessed MIBiG data with sequences
    preprocessed_path = f"{artifacts_dir}/mibig3_preprocessed.pkl"
    
    if os.path.exists(preprocessed_path):
        df = pd.read_pickle(preprocessed_path)
        print(f"   ✅ Loaded preprocessed data: {df.shape}")
        print(f"      Columns: {list(df.columns)}")
        return df
    else:
        print(f"   ❌ File not found: {preprocessed_path}")
        return None

# -------- Model Architecture -----------------------------------------------
class MIBiGClassMapper:
    """Maps BigCarp's function classes to MIBiG's 7 classes."""
    
    def __init__(self, class_mapping_path: str):
        with open(class_mapping_path, 'r') as f:
            self.mapping = json.load(f)
        
        # MIBiG classes
        self.mibig_classes = [
            'Alkaloid', 'NRP', 'Other', 'Polyketide', 'RiPP', 'Saccharide', 'Terpene'
        ]
    
    def map_labels(self, original_labels: List[str]) -> List[str]:
        """Map original function labels to MIBiG classes."""
        mibig_labels = []
        for label in original_labels:
            if label in self.mapping:
                mapped = self.mapping[label]
                if mapped in self.mibig_classes:
                    mibig_labels.append(mapped)
        return list(set(mibig_labels))  # Remove duplicates

class MIBiGDataset(Dataset):
    """Dataset for MIBiG sequences and labels."""
    
    def __init__(self, sequences, labels, specials, domains, max_length=None):
        self.sequences = sequences
        self.labels = labels
        self.specials = specials
        self.domains = domains
        
        # Create vocabulary mapping
        self.vocab = {**specials, **{d: i + len(specials) for i, d in enumerate(domains)}}
        self.vocab_size = len(self.vocab)
        
        # Process sequences
        self.encoded_sequences = []
        for seq in sequences:
            encoded = self._encode_sequence(seq, max_length)
            self.encoded_sequences.append(encoded)
    
    def _encode_sequence(self, sequence, max_length=None):
        """Encode sequence to token IDs."""
        if isinstance(sequence, str):
            domains = sequence.split()
        else:
            domains = sequence
            
        # Add CLS and SEP tokens
        tokens = ['<cls>'] + domains + ['<sep>']
        
        # Truncate if necessary
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            tokens[-1] = '<sep>'
        
        # Convert to IDs
        encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return encoded
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.encoded_sequences[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def collate_fn(batch, padding_idx=0):
    """Collate function for batching."""
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    # Pad sequences
    padded_sequences = _pad(sequences, padding_idx)
    
    return {
        'sequences': padded_sequences,
        'labels': labels,
        'attention_mask': (padded_sequences != padding_idx)
    }

class BigCarpMultiLabelClassifier(nn.Module):
    """BigCarp model with multi-label classification head."""
    
    def __init__(self, encoder, num_classes=7, hidden_dim=256, dropout=0.1, pooling='mean'):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        
        # Get encoder output dimension
        self.encoder_dim = encoder.embed_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, sequences, attention_mask=None):
        # Get encoder embeddings
        embeddings = self.encoder(sequences)  # [batch, seq_len, hidden_dim]
        
        # Pool embeddings
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Mask padding tokens
                mask = attention_mask.unsqueeze(-1).float()
                embeddings = embeddings * mask
                pooled = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = embeddings.mean(dim=1)
        elif self.pooling == 'max':
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            pooled = embeddings.max(dim=1)[0]
        elif self.pooling == 'cls':
            pooled = embeddings[:, 0]  # Use CLS token
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Classify
        logits = self.classifier(pooled)
        return logits

def load_pretrained_bigcarp(checkpoint_path: str, vocab_size: int, mask_idx: int):
    """Load a pre-trained BigCarp model."""
    print(f"Loading BigCarp model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        # Default configuration for paper models
        model_args = {
            'n_tokens': vocab_size,
            'embed_dim': 256,
            'n_layers': 8,
            'kernel_size': 3,
            'r': 128,
            'rank': None,
            'n_frozen_embs': None,
            'padding_idx': 0,
            'causal': True,
            'dropout': 0.1,
            'slim': True,
            'activation': 'relu'
        }
        # Update with actual vocab size
        model_args['n_tokens'] = vocab_size
    
    # Create model
    model = ByteNetLM(**model_args)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the state dict
        model.load_state_dict(checkpoint)
    
    return model

# -------- Training Functions -----------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, mask_idx):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        sequences = batch['sequences'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(sequences, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device, mask_idx):
    """Evaluate model and return predictions."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(sequences, attention_mask)
            loss = criterion(logits, labels)
            
            # Get probabilities
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            total_loss += loss.item()
    
    # Concatenate all results
    predictions = np.vstack(all_predictions)
    probabilities = np.vstack(all_probabilities)
    labels = np.vstack(all_labels)
    
    avg_loss = total_loss / len(dataloader)
    return predictions, probabilities, labels, avg_loss

def train_fold(train_data, test_data, specials, domains, class_columns, 
               checkpoint_path, device, args):
    """Train and evaluate one fold."""
    print(f"   📊 Training fold with {len(train_data)} train, {len(test_data)} test samples")
    
    # Create datasets
    train_sequences = train_data['domain_sequence'].tolist()
    train_labels = train_data[class_columns].values
    
    test_sequences = test_data['domain_sequence'].tolist()
    test_labels = test_data[class_columns].values
    
    train_dataset = MIBiGDataset(train_sequences, train_labels, specials, domains)
    test_dataset = MIBiGDataset(test_sequences, test_labels, specials, domains)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, padding_idx=0)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, padding_idx=0)
    )
    
    # Load pre-trained BigCarp model
    encoder = load_pretrained_bigcarp(checkpoint_path, train_dataset.vocab_size, mask_idx=0)
    
    # Create classifier
    model = BigCarpMultiLabelClassifier(
        encoder=encoder,
        num_classes=len(class_columns),
        hidden_dim=256,
        dropout=0.1,
        pooling=args.pooling
    )
    
    # Optionally freeze encoder
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    
    # Set up training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"     Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, mask_idx=0)
        
        # Evaluate
        predictions, probabilities, labels, val_loss = evaluate_model(
            model, test_loader, criterion, device, mask_idx=0
        )
        
        print(f"       Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best predictions
            best_predictions = predictions.copy()
            best_probabilities = probabilities.copy()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"       Early stopping at epoch {epoch + 1}")
                break
    
    return best_predictions, best_probabilities, labels

# -------- Evaluation Functions ---------------------------------------------
def evaluate_finetuned_model(df, cv_splits, model_path, model_name, class_cols, specials, domains, args):
    """Evaluate a fine-tuned BigCarp model using 5-fold CV."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"   Model: {model_path}")
    print(f"{'='*70}")
    
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n📁 Fold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")
        
        # Prepare fold data
        train_data = df.iloc[train_idx].copy()
        test_data = df.iloc[test_idx].copy()
        
        # Train and evaluate fold
        try:
            y_pred, y_proba, y_true = train_fold(
                train_data, test_data, specials, domains, class_cols,
                model_path, device, args
            )
            
            # Compute fold metrics
            print(f"   📊 Computing metrics...")
            fold_metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, class_cols)
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
            
            # Collect for aggregate
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
            print(f"   📊 Exact Match: {fold_metrics['exact_match_accuracy']:.4f}, "
                  f"Macro F1: {fold_metrics['macro_f1']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error in fold {fold_idx}: {e}")
            continue
    
    if not all_y_true:
        print(f"❌ No successful folds for {model_name}")
        return None
    
    # Aggregate all folds
    print(f"\n   📈 Aggregating results across {len(all_y_true)} folds...")
    aggregate_y_true = np.vstack(all_y_true)
    aggregate_y_pred = np.vstack(all_y_pred)
    aggregate_y_proba = np.vstack(all_y_proba)
    
    print(f"   🧮 Computing final aggregate metrics...")
    aggregate_metrics = compute_comprehensive_metrics(
        aggregate_y_true, aggregate_y_pred, aggregate_y_proba, class_cols
    )
    
    print(f"\n🎯 {model_name} - Final Results:")
    print(f"   Exact Match Accuracy: {aggregate_metrics['exact_match_accuracy']:.4f}")
    print(f"   Macro F1: {aggregate_metrics['macro_f1']:.4f}")
    print(f"   Macro AUC: {aggregate_metrics['macro_auc']:.4f}")
    
    return {
        'model_name': model_name,
        'embedding_column': f"Fine-tuned {model_name}",
        'fold_results': fold_results,
        'aggregate_metrics': aggregate_metrics,
        'class_names': class_cols
    }

# -------- Main Execution ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MIBiG 3.0 Fine-tuned BigCarp Evaluation")
    parser.add_argument('--artifacts_dir', type=str, default="artifacts/classification/mibig3",
                       help="Path to MIBiG 3.0 artifacts directory")
    parser.add_argument('--outdir', type=str, default="results/mibig3_finetuned",
                       help="Output directory for results")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    parser.add_argument('--vocab_path', type=str,
                       default="/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json",
                       help="Path to vocabulary file")
    parser.add_argument('--class_mapping', type=str,
                       default="/home/u5bb/han00.u5bb/workspace/tg_learn/data/raw/bgc_class_mapping.json",
                       help="Path to class mapping file")
    parser.add_argument('--batch_size', type=int, default=32,
                       help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument('--patience', type=int, default=5,
                       help="Early stopping patience")
    parser.add_argument('--freeze_encoder', action='store_true',
                       help="Freeze encoder parameters")
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'cls'],
                       help="Pooling strategy")
    
    args = parser.parse_args()

    print("🚀 MIBiG 3.0 Fine-tuned BigCarp Evaluation with Stratified CV")
    print(f"📁 Artifacts: {args.artifacts_dir}")
    print(f"📁 Output: {args.outdir}")
    
    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    data = load_mibig3_data(args.artifacts_dir)
    if data is None:
        print("❌ Failed to load MIBiG 3.0 data!")
        return
    
    # Load vocabulary
    print("📚 Loading vocabulary...")
    with open(args.vocab_path, 'r') as f:
        vocab_info = json.load(f)
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    
    # Convert product classes to binary
    print("🔄 Converting product classes to binary...")
    data, class_cols = convert_product_classes_to_binary(data)
    print(f"   Found classes: {class_cols}")
    
    # Create stratified CV splits
    print("🔄 Creating stratified CV splits...")
    cv_splits = create_stratified_splits(data, class_cols, n_splits=5, random_state=args.seed)
    print(f"   Created {len(cv_splits)} CV splits")
    
    # Define models to evaluate
    models_to_evaluate = [
        ("artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt", "BigCarp Random (Fine-tuned)"),
        ("artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt", "BigCarp ESM1b Fine-tuned"),
    ]
    
    all_results = []
    
    # Evaluate each model
    for model_path, model_name in models_to_evaluate:
        if os.path.exists(model_path):
            print(f"\n{'🔬' * 3} EVALUATING {model_name.upper()} {'🔬' * 3}")
            
            result = evaluate_finetuned_model(
                data, cv_splits, model_path, model_name, class_cols, 
                specials, domains, args
            )
            
            if result is not None:
                all_results.append(result)
                print(f"✅ {model_name} completed successfully!")
            else:
                print(f"❌ {model_name} failed!")
        else:
            print(f"⚠️  Model not found: {model_path}, skipping {model_name}")
    
    # Create final comparison
    if all_results:
        create_comparison_table(all_results, args.outdir)
        
        # Save complete results
        with open(f"{args.outdir}/complete_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📋 Results saved to: {args.outdir}/")
    else:
        print("❌ No successful evaluations!")

if __name__ == "__main__":
    main()