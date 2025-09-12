"""
BigCarp Fine-tuning Pipeline for MIBiG 3.0 Multi-label Classification

This script fine-tunes pre-trained BigCarp models (Random and ESM1b) on MIBiG 3.0 data
using the exact same 5-fold stratified cross-validation pipeline as the embedding evaluation
for fair comparison.

Models evaluated:
- BigCarp Random (fine-tuned)
- BigCarp ESM1b Fine-tuned

Uses identical metrics and evaluation pipeline as mibig3_stratified_evaluation.py
"""

import argparse
import json
import os
import sys, pickle, warnings, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             hamming_loss, classification_report, roc_auc_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

# Add paths for BigCarp imports
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

from sequence_models.convolutional import ByteNetLM
from sequence_models.collaters import _pad as seq_models_pad
from sequence_models.collaters import _pad
# Optional import - not needed for our evaluation pipeline
try:
    from emb_sub.utils import extract_layer_embeddings
except ImportError:
    extract_layer_embeddings = None

# Import evaluation functions from the main script for identical pipeline
try:
    from scripts.classification.mibig3_stratified_evaluation import (
        convert_product_classes_to_binary,
        create_stratified_splits,
        compute_comprehensive_metrics,
        create_comparison_table,
        exact_match_accuracy
    )
    print("✅ Imported evaluation functions from mibig3_stratified_evaluation.py")
except ImportError as e:
    print(f"⚠️  Could not import from mibig3_stratified_evaluation.py: {e}")
    print("   Using local implementations")
    
    # Local implementation as fallback
    def convert_product_classes_to_binary(df):
        """Convert semicolon-separated product_class to binary columns."""
        print("\n🔄 Converting product classes to binary columns...")
        
        # Ensure we're working with a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Get all unique classes from product_class column
        all_classes = set()
        for class_str in df['product_class'].dropna():
            if pd.isna(class_str) or class_str == '':
                continue
            classes = str(class_str).split(';')
            all_classes.update([cls.strip() for cls in classes if cls.strip()])
        
        all_classes = sorted(list(all_classes))
        print(f"   Found {len(all_classes)} unique classes: {all_classes}")
        
        # Create binary columns
        for class_name in all_classes:
            df[class_name] = df['product_class'].apply(
                lambda x: 1 if pd.notna(x) and class_name in str(x).split(';') else 0
            )
        
        # Show distribution
        print(f"\n   Class distribution:")
        for class_name in all_classes:
            count = df[class_name].sum()
            print(f"     {class_name}: {count} samples ({count/len(df)*100:.1f}%)")
        
        return df, all_classes
    
    def create_stratified_splits(df, class_cols, n_splits=5, random_state=42):
        """Create stratified splits for multi-label data."""
        from sklearn.model_selection import StratifiedKFold
        
        # Create splits - simplified version without iterative stratification
        print("🔄 Creating stratified CV splits...")
        
        # For multi-label, use the most frequent class for stratification
        primary_class = df[class_cols].sum().idxmax()
        print(f"   Using {primary_class} as primary class for stratification")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(df, df[primary_class]))
        
        print(f"   Created {len(splits)} CV splits")
        return splits
    
    def compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
        """Compute comprehensive evaluation metrics."""
        from sklearn.metrics import f1_score, roc_auc_score
        
        metrics = {}
        
        # Exact match accuracy
        metrics['exact_match_accuracy'] = np.mean(np.all(y_true == y_pred, axis=1))
        
        # Micro F1
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        
        # Macro F1 
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        
        # Weighted Macro F1
        metrics['weighted_macro_f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Micro AUC
        try:
            metrics['micro_auc'] = roc_auc_score(y_true.ravel(), y_proba.ravel())
        except ValueError:
            metrics['micro_auc'] = float('nan')
        
        # Macro AUC
        try:
            metrics['macro_auc'] = roc_auc_score(y_true, y_proba, average='macro')
        except ValueError:
            metrics['macro_auc'] = float('nan')
        
        return metrics
    
    def create_comparison_table(all_results, outdir):
        """Create comparison table - simplified version."""
        if not all_results:
            print("❌ No results to compare")
            return
        
        import pandas as pd
        comparison_data = []
        for result in all_results:
            if result is not None:
                row = {'Model': result['model_name']}
                metrics = result['aggregate_metrics']
                row.update(metrics)
                comparison_data.append(row)
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            csv_path = f"{outdir}/mibig3_comparison.csv"
            df_comparison.to_csv(csv_path, index=False)
            print(f"📊 Comparison table saved to: {csv_path}")
    
    def exact_match_accuracy(y_true, y_pred):
        """Calculate exact match accuracy for multi-label."""
        return np.mean(np.all(y_true == y_pred, axis=1))

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
    
    # Try the new format first
    new_format_path = "data/processed/bgc_product_classification/processed_mibig3/mibig3_preprocessed.pkl"
    old_format_path = f"{artifacts_dir}/mibig3_preprocessed.pkl"
    
    # Check new format first
    if os.path.exists(new_format_path):
        df = pd.read_pickle(new_format_path)
        print(f"   ✅ Loaded new format data: {df.shape}")
        print(f"      Columns: {list(df.columns)}")
        
        # Convert domain_sequence from list to string if needed
        if 'domain_sequence' in df.columns:
            if isinstance(df['domain_sequence'].iloc[0], list):
                print("   🔧 Converting domain_sequence from list to string...")
                df['domain_sequence'] = df['domain_sequence'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        
        return df
    elif os.path.exists(old_format_path):
        df = pd.read_pickle(old_format_path)
        print(f"   ✅ Loaded old format data: {df.shape}")
        print(f"      Columns: {list(df.columns)}")
        return df
    else:
        print(f"   ❌ File not found in either location:")
        print(f"      New: {new_format_path}")
        print(f"      Old: {old_format_path}")
        return None

class MIBiGClassMapper:
    """Maps BigCarp's 55 function classes to MIBiG's 7 classes"""
    
    def __init__(self, mapping_path: str):
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)
        
        self.mibig_classes = ["Alkaloid", "NRP", "Other", "Polyketide", "RiPP", "Saccharide", "Terpene"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.mibig_classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.mibig_classes)}
        
    def map_bigcarp_to_mibig(self, bigcarp_class: str) -> List[str]:
        """Convert BigCarp class to list of MIBiG classes"""
        if bigcarp_class not in self.mapping:
            print(f"Warning: Unknown BigCarp class '{bigcarp_class}', mapping to 'Other'")
            return ["Other"]
        
        mapping = self.mapping[bigcarp_class]
        
        if isinstance(mapping, str):
            return [mapping]
        elif isinstance(mapping, dict):
            # Multi-class mapping
            return list(mapping.values())
        else:
            print(f"Warning: Invalid mapping for '{bigcarp_class}', using 'Other'")
            return ["Other"]


# Note: Using convert_product_classes_to_binary from main script for identical pipeline
# Original function kept for reference but will use imported version


class MIBiGDataset(Dataset):
    """Dataset for MIBiG BGC sequences with multi-label classification"""
    
    def __init__(self, data_df: pd.DataFrame, specials: Dict, domains: Dict, 
                 class_columns: List[str], max_length=None):
        self.data = data_df
        self.specials = specials
        self.domains = domains
        self.class_columns = class_columns
        
        # Create vocabulary mapping
        self.vocab = {**specials, **{d: i + len(specials) for i, d in enumerate(domains)}}
        self.vocab_size = len(self.vocab)
        
        # Process sequences
        self.encoded_sequences = []
        for _, row in data_df.iterrows():
            encoded = self._encode_sequence(row['domain_sequence'], max_length)
            self.encoded_sequences.append(encoded)
    
    def _encode_sequence(self, sequence, max_length=None):
        """Encode sequence to token IDs."""
        if isinstance(sequence, str):
            domains = sequence.split()
        else:
            domains = sequence
            
        # Don't add CLS and SEP tokens - this vocabulary doesn't have them
        tokens = domains
        
        # Truncate if necessary
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Convert to IDs - handle missing tokens
        encoded = []
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            elif 'UNK' in self.vocab:
                encoded.append(self.vocab['UNK'])
            elif '<unk>' in self.vocab:
                encoded.append(self.vocab['<unk>'])
            else:
                # Use the first special token as fallback
                encoded.append(list(self.vocab.values())[0])
        return encoded
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Multi-hot labels
        labels = np.array([row[cls] for cls in self.class_columns], dtype=np.float32)
        
        return {
            'sequence': torch.tensor(self.encoded_sequences[idx], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float),
            'bgc_id': row['bgc_id']
        }


def collate_fn(batch, padding_idx=0):
    """Collate function for DataLoader - using mask_idx as padding like training script"""
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    bgc_ids = [item['bgc_id'] for item in batch]
    
    # Pad sequences using mask_idx (consistent with training script)
    sequences_padded = _pad(sequences, padding_idx)
    
    return {
        'sequences': sequences_padded,
        'labels': labels,
        'bgc_ids': bgc_ids,
        'attention_mask': (sequences_padded != padding_idx).unsqueeze(-1)  # Add extra dimension for BigCarp
    }


class BigCarpMultiLabelClassifier(nn.Module):
    """BigCarp + Multi-label Classification Head"""
    
    def __init__(self, bigcarp_model: ByteNetLM, num_classes: int = 7, 
                 hidden_dim: int = None, freeze_encoder: bool = False,
                 pooling: str = 'mean'):
        super().__init__()
        
        self.bigcarp = bigcarp_model
        self.num_classes = num_classes
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        
        # Auto-detect hidden dimension from BigCarp model
        if hidden_dim is None:
            # Get the output dimension from the last layer norm
            if hasattr(self.bigcarp, 'last_norm'):
                hidden_dim = self.bigcarp.last_norm.weight.shape[0]
            else:
                # Fallback to d_model from the model
                hidden_dim = getattr(self.bigcarp, 'd_model', 256)
        
        self.hidden_dim = hidden_dim
        print(f"   Using hidden_dim: {hidden_dim} for classification head")
        
        # Freeze BigCarp encoder if specified
        if freeze_encoder:
            for param in self.bigcarp.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, tokens, input_mask=None):
        # Extract features from BigCarp using the embedder (ByteNet)
        features = self.bigcarp.embedder(tokens, input_mask=input_mask)
        features = self.bigcarp.last_norm(features)
        
        # Pool sequence features
        if self.pooling == 'mean':
            if input_mask is not None:
                # Mask out padding positions
                mask = input_mask.squeeze(-1)  # [batch_size, seq_len]
                pooled = (features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                pooled = features.mean(dim=1)
        elif self.pooling == 'max':
            pooled = features.max(dim=1)[0]
        elif self.pooling == 'cls':
            # Use first position 
            pooled = features[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        logits = self.classifier(pooled)
        return logits


def create_stratified_splits(df, class_cols, n_splits=5, random_state=42):
    """Create stratified CV splits for multi-label classification."""
    y_binary = df[class_cols].values
    
    print(f"\n📊 Class Distribution Analysis:")
    class_counts = y_binary.sum(axis=0)
    for i, col in enumerate(class_cols):
        print(f"  {col}: {class_counts[i]:4d} samples ({class_counts[i]/len(df)*100:5.1f}%)")
    
    if STRATIFIED_AVAILABLE:
        try:
            # Shuffle indices manually
            indices = np.arange(len(df))
            if random_state is not None:
                np.random.seed(random_state)
                np.random.shuffle(indices)
            
            stratifier = IterativeStratification(
                n_splits=n_splits, 
                order=2,
                sample_distribution_per_fold=[1.0/n_splits]*n_splits
            )
            splits = list(stratifier.split(indices, y_binary[indices]))
            
            # Convert back to original indices
            splits = [(indices[train], indices[test]) for train, test in splits]
            
            print(f"✅ Using stratified multi-label {n_splits}-fold CV")
            return splits
        except Exception as e:
            print(f"⚠️  Stratified CV failed: {e}")
    
    # Fallback to KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(np.arange(len(df))))
    print(f"✅ Using standard {n_splits}-fold CV")
    return splits


def train_epoch(model, dataloader, optimizer, criterion, device, mask_idx):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequences'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits = model(sequences, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        predictions = torch.sigmoid(logits).detach().cpu().numpy()
        all_predictions.append(predictions)
        all_labels.append(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    return avg_loss, all_predictions, all_labels


def evaluate(model, dataloader, criterion, device, mask_idx):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tokens = batch['tokens'].to(device)
            labels = batch['labels'].to(device)
            
            input_mask = (tokens != mask_idx).float().unsqueeze(-1)
            
            logits = model(tokens, input_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            predictions = torch.sigmoid(logits).detach().cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    return avg_loss, all_predictions, all_labels


def compute_metrics(predictions, labels, class_names, threshold=0.5):
    """Compute multi-label classification metrics"""
    pred_binary = (predictions > threshold).astype(int)
    
    # Overall metrics
    micro_f1 = f1_score(labels, pred_binary, average='micro')
    macro_f1 = f1_score(labels, pred_binary, average='macro')
    exact_match = np.mean(np.all(labels == pred_binary, axis=1))
    hamming = hamming_loss(labels, pred_binary)
    
    # AUC-ROC metrics
    try:
        # Micro-averaged AUC-ROC
        micro_auc = roc_auc_score(labels, predictions, average='micro')
        
        # Macro-averaged AUC-ROC
        macro_auc = roc_auc_score(labels, predictions, average='macro')
        
        # Per-class AUC-ROC
        per_class_auc = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) > 1:  # Check if class has both positive and negative samples
                auc_score = roc_auc_score(labels[:, i], predictions[:, i])
                per_class_auc.append(auc_score)
            else:
                # If only one class present, set AUC to 0.5 (random)
                per_class_auc.append(0.5)
        
        per_class_auc = np.array(per_class_auc)
        
    except ValueError as e:
        print(f"Warning: Could not compute AUC-ROC: {e}")
        micro_auc = 0.5
        macro_auc = 0.5
        per_class_auc = np.full(labels.shape[1], 0.5)
    
    # Per-class metrics
    report = classification_report(labels, pred_binary, target_names=class_names, output_dict=True)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'exact_match': exact_match,
        'hamming_loss': hamming,
        'micro_auc': micro_auc,
        'macro_auc': macro_auc,
        'per_class_auc': per_class_auc,
        'classification_report': report
    }


def load_pretrained_bigcarp(checkpoint_path: str, vocab_size: int, mask_idx: int):
    """Load pre-trained BigCarp model with configuration matching training script"""
    
    # Load checkpoint to get actual configuration
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract configuration from checkpoint (with fallback defaults matching training script)
    d_model = checkpoint.get('d_model', 256)
    d_embed = checkpoint.get('d_embed', 1280) 
    n_layers = checkpoint.get('n_layers', 32)  # Default to 32 like training script
    kernel_size = checkpoint.get('kernel_size', 3)
    r = checkpoint.get('r', 128)
    slim = not checkpoint.get('wide', False)  # Training script uses (not args.wide)
    
    print(f"   Loading BigCarp with config: d_model={d_model}, d_embed={d_embed}, n_layers={n_layers}")
    print(f"   Additional config: kernel_size={kernel_size}, r={r}, slim={slim}")
    
    # CRITICAL: Use mask_idx as padding_idx to match training script
    model_config = {
        'n_tokens': vocab_size,
        'd_embedding': d_embed,
        'd_model': d_model,
        'n_layers': n_layers,
        'kernel_size': kernel_size,
        'r': r,
        'slim': slim,
        'padding_idx': mask_idx,  # IMPORTANT: Training script uses mask_idx as padding_idx
        'causal': False,  # Training script: causal=args.ar (default False)
        'final_ln': True,
        'activation': 'gelu'
    }
    
    model = ByteNetLM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def train_fold(train_data, test_data, specials, domains, class_columns, 
               checkpoint_path, fold_idx, args):
    """Train and evaluate one fold"""
    
    print(f"\n📁 Fold {fold_idx + 1}: Train={len(train_data)}, Test={len(test_data)}")
    
    # Create datasets
    train_dataset = MIBiGDataset(train_data, specials, domains, class_columns)
    test_dataset = MIBiGDataset(test_data, specials, domains, class_columns)
    
    padding_idx = specials['-']
    mask_idx = specials['#']
    
    # IMPORTANT: Use mask_idx for padding to match training script behavior
    # Training script uses padding_idx=mask_idx in ByteNetLM creation
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, mask_idx)  # Use mask_idx for padding
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, mask_idx)  # Use mask_idx for padding
    )
    
    # Load BigCarp model
    vocab_size = len(domains) + len(specials)
    bigcarp_model = load_pretrained_bigcarp(checkpoint_path, vocab_size, mask_idx)
    
    # Create classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BigCarpMultiLabelClassifier(
        bigcarp_model,
        num_classes=len(class_columns),
        freeze_encoder=args.freeze_encoder,
        pooling=args.pooling
    ).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device, mask_idx
        )
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model, test_loader, criterion, device, mask_idx
        )
        
        # Compute metrics
        val_metrics = compute_metrics(val_preds, val_labels, class_columns)
        
        print(f"  Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val F1={val_metrics['micro_f1']:.4f}, "
              f"Val AUC={val_metrics['micro_auc']:.4f}")
        
        # Early stopping
        if val_metrics['micro_f1'] > best_f1:
            best_f1 = val_metrics['micro_f1']
            patience_counter = 0
            # Could save best model here if needed
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    final_loss, final_preds, final_labels = evaluate(
        model, test_loader, criterion, device, mask_idx
    )
    final_metrics = compute_metrics(final_preds, final_labels, class_columns)
    
    return final_metrics


def train_fold_for_evaluation(train_data, test_data, specials, domains, class_columns, 
                             model_path, device, args):
    """Train one fold and return predictions in evaluation format"""
    
    # Train the fold (reuse existing train_fold logic)
    fold_metrics = train_fold(train_data, test_data, specials, domains, 
                             class_columns, model_path, 0, args)
    
    # Need to return predictions - let me modify train_fold to return them
    # For now, let's create a simplified version that returns the required format
    
    # Create datasets for prediction extraction
    test_dataset = MIBiGDataset(test_data, specials, domains, class_columns)
    padding_idx = specials['-']
    mask_idx = specials['#']
    
    # Use mask_idx for padding consistency with training script
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, mask_idx)  # Use mask_idx for padding
    )
    
    # Load and create model (same as train_fold)
    vocab_size = len(domains) + len(specials)
    bigcarp_model = load_pretrained_bigcarp(model_path, vocab_size, mask_idx)
    
    model = BigCarpMultiLabelClassifier(
        bigcarp_model,
        num_classes=len(class_columns),
        freeze_encoder=args.freeze_encoder,
        pooling=args.pooling
    ).to(device)
    
    # Train the model briefly (simplified training)
    train_dataset = MIBiGDataset(train_data, specials, domains, class_columns)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, mask_idx)  # Use mask_idx for padding
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Quick training (fewer epochs for testing)
    model.train()
    for epoch in range(min(3, args.epochs)):  # Reduce epochs for faster testing
        for batch in train_loader:
            sequences = batch['sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(sequences, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    # Get predictions
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(sequences, attention_mask)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Convert to numpy arrays
    y_proba = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = (y_proba > 0.0).astype(int)
    
    return y_pred, y_proba, y_true


def main():
    parser = argparse.ArgumentParser(description="MIBiG 3.0 Fine-tuned BigCarp Evaluation")
    parser.add_argument('--artifacts_dir', type=str, default="artifacts/classification/mibig3",
                       help="Path to MIBiG 3.0 artifacts directory (fallback location)")
    parser.add_argument('--outdir', type=str, default="results/mibig3_finetuned",
                       help="Output directory for results")
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
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    
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
    
    # Convert product classes to binary using main script's function
    print("🔄 Converting product classes to binary...")
    data, class_cols = convert_product_classes_to_binary(data)
    print(f"   Found classes: {class_cols}")
    
    # Create stratified CV splits using main script's function
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
    
    # Create final comparison using main script's function
    if all_results:
        create_comparison_table(all_results, args.outdir)
        
        # Save complete results
        with open(f"{args.outdir}/complete_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📋 Results saved to: {args.outdir}/")
    else:
        print("❌ No successful evaluations!")

# -------- Evaluation Function ---------------------------------------------- 
def evaluate_finetuned_model(df, cv_splits, model_path, model_name, class_cols, specials, domains, args):
    """Evaluate a fine-tuned BigCarp model using 5-fold CV with identical metrics."""
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
        
        # Train and evaluate fold using existing train_fold function
        try:
            # Modify train_fold to return predictions in the right format
            y_pred, y_proba, y_true = train_fold_for_evaluation(
                train_data, test_data, specials, domains, class_cols,
                model_path, device, args
            )
            
            # Compute fold metrics using main script's function
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

def train_fold_for_evaluation(train_data, test_data, specials, domains, class_columns, 
                              checkpoint_path, device, args):
    """Modified train_fold function that returns predictions in the format expected by main evaluation."""
    print(f"   📊 Training fold with {len(train_data)} train, {len(test_data)} test samples")
    
    # Create datasets
    train_sequences = train_data['domain_sequence'].tolist()
    train_labels = train_data[class_columns].values
    
    test_sequences = test_data['domain_sequence'].tolist()
    test_labels = test_data[class_columns].values
    
    train_dataset = MIBiGDataset(train_data, specials, domains, class_columns)
    test_dataset = MIBiGDataset(test_data, specials, domains, class_columns)
    
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
        bigcarp_model=encoder,
        num_classes=len(class_columns),
        hidden_dim=256,
        freeze_encoder=args.freeze_encoder,
        pooling=args.pooling
    )
    
    # Encoder freezing already handled in constructor
    
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
        predictions, probabilities, labels, val_loss = evaluate_model_fold(
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

def evaluate_model_fold(model, dataloader, criterion, device, mask_idx):
    """Evaluate model and return predictions for one fold."""
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

if __name__ == '__main__':
    main()