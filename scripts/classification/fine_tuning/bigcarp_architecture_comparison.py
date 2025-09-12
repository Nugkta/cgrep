#!/usr/bin/env python
"""
BigCarp Architecture Comparison for Single-Label Classification
===============================================================
Compare different architectures for BGC product class prediction:
1. Mean pooling approach vs Class token approach
2. Single-label classification with class tokens
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Add paths for BigCarp imports
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn')
sys.path.append('/home/u5bb/han00.u5bb/workspace/tg_learn/external/protein-sequence-models')
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')

from sequence_models.convolutional import ByteNetLM
from sequence_models.collaters import _pad

# Import for stratified splits
try:
    from skmultilearn.model_selection import iterative_train_test_split
    STRATIFIED_AVAILABLE = True
except ImportError:
    from sklearn.model_selection import train_test_split
    STRATIFIED_AVAILABLE = False


def load_mibig3_data(artifacts_dir="artifacts/classification/mibig3"):
    """Load MIBiG 3.0 data."""
    print("📂 Loading MIBiG 3.0 data...")
    
    # Try the new format first
    new_format_path = "data/processed/bgc_product_classification/processed_mibig3/mibig3_preprocessed.pkl"
    
    if os.path.exists(new_format_path):
        with open(new_format_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ Loaded new format data: {data.shape}")
        print(f"      Columns: {list(data.columns)}")
        
        # Convert domain_sequence from list to string if needed
        if 'domain_sequence' in data.columns and isinstance(data['domain_sequence'].iloc[0], list):
            print("   🔧 Converting domain_sequence from list to string...")
            data['domain_sequence'] = data['domain_sequence'].apply(lambda x: ';'.join(x) if isinstance(x, list) else x)
        
        return data
    
    # Fallback to old format
    old_format_path = f"{artifacts_dir}/mibig3_preprocessed.pkl"
    if os.path.exists(old_format_path):
        with open(old_format_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ Loaded fallback data: {data.shape}")
        return data
    
    print(f"   ❌ No data found in either location")
    return None


def prepare_single_label_data(data):
    """Convert multi-label data to single-label (primary class only)."""
    print("🔄 Converting to single-label classification...")
    
    # Extract primary class (first class in semicolon-separated list)
    data = data.copy()
    data['primary_class'] = ''
    
    for idx, row in data.iterrows():
        if pd.notna(row['product_class']):
            classes = str(row['product_class']).split(';')
            if classes:
                data.at[idx, 'primary_class'] = classes[0].strip()
    
    # Remove samples without primary class
    data = data[data['primary_class'] != ''].reset_index(drop=True)
    
    # Show class distribution
    class_counts = data['primary_class'].value_counts()
    print(f"   Primary class distribution:")
    for cls, count in class_counts.items():
        print(f"     {cls}: {count} samples ({count/len(data)*100:.1f}%)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['primary_class'])
    
    return data, label_encoder


def create_stratified_splits(data, n_splits=5, random_state=42):
    """Create stratified CV splits for single-label classification."""
    print(f"🔄 Creating {n_splits}-fold stratified CV splits...")
    
    from sklearn.model_selection import StratifiedKFold
    
    X = np.arange(len(data))
    y = data['label_encoded'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))
    
    print(f"   Created {len(splits)} CV splits")
    return splits


class MIBiGSingleLabelDataset(Dataset):
    """Dataset for single-label BGC classification with class tokens."""
    
    def __init__(self, data, specials, domains, include_class_token=True, use_unknown_token=False):
        self.data = data.reset_index(drop=True)
        self.specials = specials
        self.domains = domains
        self.include_class_token = include_class_token
        self.use_unknown_token = use_unknown_token
        
        # Create class token mapping from MIBiG 3.0 classes to BigCarp training tokens
        unique_classes = sorted(data['primary_class'].unique())
        
        # Mapping from our classes to BigCarp training vocabulary tokens
        mibig_to_bigcarp_mapping = {
            'NRP': 'nrps',
            'Polyketide': 't1pks',  # Use t1pks as the main polyketide token
            'Terpene': 'terpene',
            'RiPP': 'bacteriocin',  # Use bacteriocin as representative RiPP
            'Saccharide': 'amglyccycl',  # Use amglyccycl for saccharide
            'Alkaloid': 'hserlactone-nrps',  # Use hybrid token for alkaloid
            'Other': 'otherks'  # Use otherks for other
        }
        
        self.class_to_token = {}
        
        # Map classes to available tokens
        for cls in unique_classes:
            if cls in mibig_to_bigcarp_mapping:
                bigcarp_token = mibig_to_bigcarp_mapping[cls]
                if bigcarp_token in specials:
                    self.class_to_token[cls] = specials[bigcarp_token]
                    print(f"   ✅ Mapped {cls} → {bigcarp_token} (token {specials[bigcarp_token]})")
                else:
                    print(f"   ⚠️  BigCarp token '{bigcarp_token}' for class '{cls}' not found in vocabulary")
            else:
                print(f"   ⚠️  No mapping defined for class '{cls}'")
        
        print(f"   📝 Successfully mapped class tokens: {list(self.class_to_token.keys())}")
        
        # Encode sequences
        self.encoded_sequences = []
        for idx, row in self.data.iterrows():
            sequence = []
            
            # Add class token if requested
            if self.include_class_token:
                if self.use_unknown_token:
                    # Use unknown token for testing
                    if 'UNK' in specials:
                        sequence.append(specials['UNK'])
                    elif '<unk>' in specials:
                        sequence.append(specials['<unk>'])
                    else:
                        sequence.append(0)  # Fallback
                else:
                    # Use ground truth class token for training
                    cls = row['primary_class']
                    if cls in self.class_to_token:
                        sequence.append(self.class_to_token[cls])
                    else:
                        # Fallback to UNK token
                        sequence.append(specials.get('UNK', specials.get('<unk>', 0)))
            
            # Add domain sequence
            domains_seq = row['pfam_sequence']
            unk_idx = specials.get('UNK', specials.get('<unk>', 0))
            
            for domain in domains_seq:
                if domain in domains:
                    sequence.append(domains[domain] + len(specials))
                else:
                    sequence.append(unk_idx)
            
            self.encoded_sequences.append(sequence)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'sequence': torch.tensor(self.encoded_sequences[idx], dtype=torch.long),
            'label': torch.tensor(row['label_encoded'], dtype=torch.long),
            'primary_class': row['primary_class'],
            'bgc_id': row['bgc_id']
        }


def collate_fn(batch, padding_idx=0):
    """Collate function for single-label classification."""
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    primary_classes = [item['primary_class'] for item in batch]
    bgc_ids = [item['bgc_id'] for item in batch]
    
    # Pad sequences
    sequences_padded = _pad(sequences, padding_idx)
    
    return {
        'sequences': sequences_padded,
        'labels': labels,
        'primary_classes': primary_classes,
        'bgc_ids': bgc_ids,
        'attention_mask': (sequences_padded != padding_idx).unsqueeze(-1)
    }


class BigCarpSingleLabelClassifier(nn.Module):
    """BigCarp + Single-label Classification Head with different pooling strategies."""
    
    def __init__(self, bigcarp_model: ByteNetLM, num_classes: int, 
                 hidden_dim: int = None, freeze_encoder: bool = False,
                 pooling: str = 'mean'):
        super().__init__()
        
        self.bigcarp = bigcarp_model
        self.num_classes = num_classes
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        
        # Auto-detect hidden dimension from BigCarp model
        if hidden_dim is None:
            if hasattr(self.bigcarp, 'last_norm'):
                hidden_dim = self.bigcarp.last_norm.weight.shape[0]
            else:
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
        # Extract features from BigCarp
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
            # Use first position (class token if present)
            pooled = features[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        logits = self.classifier(pooled)
        return logits


def load_pretrained_bigcarp(checkpoint_path: str, vocab_size: int, mask_idx: int):
    """Load pre-trained BigCarp model with configuration matching training script."""
    
    # Load checkpoint to get actual configuration
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract configuration from checkpoint
    d_model = checkpoint.get('d_model', 256)
    d_embed = checkpoint.get('d_embed', 1280) 
    n_layers = checkpoint.get('n_layers', 32)
    kernel_size = checkpoint.get('kernel_size', 3)
    r = checkpoint.get('r', 128)
    slim = not checkpoint.get('wide', False)
    
    print(f"   Loading BigCarp with config: d_model={d_model}, d_embed={d_embed}, n_layers={n_layers}")
    
    model_config = {
        'n_tokens': vocab_size,
        'd_embedding': d_embed,
        'd_model': d_model,
        'n_layers': n_layers,
        'kernel_size': kernel_size,
        'r': r,
        'slim': slim,
        'padding_idx': mask_idx,  # Use mask_idx as padding_idx like training script
        'causal': False,
        'final_ln': True,
        'activation': 'gelu'
    }
    
    model = ByteNetLM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def train_and_evaluate_fold(train_data, test_data, specials, domains, label_encoder,
                           model_path, fold_idx, args, pooling_strategy='mean'):
    """Train and evaluate one fold with specified pooling strategy."""
    
    print(f"\n📁 Fold {fold_idx + 1}: Train={len(train_data)}, Test={len(test_data)}")
    print(f"   Pooling strategy: {pooling_strategy}")
    
    # Create datasets
    # Training: use ground truth class tokens
    train_dataset = MIBiGSingleLabelDataset(
        train_data, specials, domains, 
        include_class_token=True, use_unknown_token=False
    )
    
    # Testing: use unknown tokens (simulating real prediction scenario)
    test_dataset = MIBiGSingleLabelDataset(
        test_data, specials, domains, 
        include_class_token=True, use_unknown_token=True
    )
    
    mask_idx = specials['#']
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, mask_idx)
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, mask_idx)
    )
    
    # Load BigCarp model
    vocab_size = len(domains) + len(specials)
    bigcarp_model = load_pretrained_bigcarp(model_path, vocab_size, mask_idx)
    
    # Create classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BigCarpSingleLabelClassifier(
        bigcarp_model,
        num_classes=len(label_encoder.classes_),
        freeze_encoder=args.freeze_encoder,
        pooling=pooling_strategy
    ).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            sequences = batch['sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(sequences, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequences'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(sequences, attention_mask)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"  Epoch {epoch+1:2d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(sequences, attention_mask)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics with zero_division parameter to suppress warnings
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Classification report - handle case where test set doesn't contain all classes
    class_names = label_encoder.classes_
    unique_test_labels = np.unique(all_labels)
    
    if len(unique_test_labels) < len(class_names):
        # Only include classes that appear in the test set
        test_class_names = [class_names[i] for i in unique_test_labels]
        report = classification_report(all_labels, all_predictions, 
                                     target_names=test_class_names, output_dict=True,
                                     labels=unique_test_labels, zero_division=0)
    else:
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True,
                                     zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


def evaluate_architecture(data, cv_splits, model_path, model_name, label_encoder,
                         specials, domains, args, pooling_strategy='mean'):
    """Evaluate an architecture using 5-fold CV."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: {model_name} - {pooling_strategy.upper()} pooling")
    print(f"   Model: {model_path}")
    print(f"{'='*70}")
    
    fold_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        # Prepare fold data
        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()
        
        try:
            result = train_and_evaluate_fold(
                train_data, test_data, specials, domains, label_encoder,
                model_path, fold_idx, args, pooling_strategy
            )
            fold_results.append(result)
            
            print(f"   📊 Fold {fold_idx+1} - Accuracy: {result['accuracy']:.4f}, "
                  f"F1 Macro: {result['f1_macro']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error in fold {fold_idx}: {e}")
            continue
    
    if not fold_results:
        print("❌ No successful folds")
        return None
    
    # Aggregate results
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_f1_macro = np.mean([r['f1_macro'] for r in fold_results])
    avg_f1_weighted = np.mean([r['f1_weighted'] for r in fold_results])
    
    print(f"\n📊 Final Results for {model_name} - {pooling_strategy.upper()}:")
    print(f"   Average Accuracy: {avg_accuracy:.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"   Average F1 Macro: {avg_f1_macro:.4f} ± {np.std([r['f1_macro'] for r in fold_results]):.4f}")
    print(f"   Average F1 Weighted: {avg_f1_weighted:.4f} ± {np.std([r['f1_weighted'] for r in fold_results]):.4f}")
    
    return {
        'model_name': f"{model_name} - {pooling_strategy.upper()}",
        'pooling_strategy': pooling_strategy,
        'fold_results': fold_results,
        'aggregate_metrics': {
            'accuracy': avg_accuracy,
            'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
            'f1_macro': avg_f1_macro,
            'f1_macro_std': np.std([r['f1_macro'] for r in fold_results]),
            'f1_weighted': avg_f1_weighted,
            'f1_weighted_std': np.std([r['f1_weighted'] for r in fold_results])
        }
    }


def main():
    parser = argparse.ArgumentParser(description="BigCarp Architecture Comparison for Single-Label Classification")
    parser.add_argument('--artifacts_dir', type=str, default="artifacts/classification/mibig3",
                       help="Path to MIBiG 3.0 artifacts directory")
    parser.add_argument('--outdir', type=str, default="results/bigcarp_architecture_comparison",
                       help="Output directory for results")
    parser.add_argument('--vocab_path', type=str,
                       default="/home/u5bb/han00.u5bb/workspace/tg_learn/data/processed/vocabularies/pfam_vocab.json",
                       help="Path to vocabulary file")
    parser.add_argument('--batch_size', type=int, default=16,
                       help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument('--patience', type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument('--freeze_encoder', action='store_true',
                       help="Freeze encoder parameters")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()

    print("🚀 BigCarp Architecture Comparison for Single-Label Classification")
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
    
    # Convert to single-label format
    data, label_encoder = prepare_single_label_data(data)
    
    # Load vocabulary
    print("📚 Loading vocabulary...")
    with open(args.vocab_path, 'r') as f:
        vocab_info = json.load(f)
    
    specials = vocab_info['specials']
    domains = vocab_info['domains']
    
    # Create stratified CV splits
    cv_splits = create_stratified_splits(data, n_splits=5, random_state=args.seed)
    
    # Define models and pooling strategies to evaluate
    models_to_evaluate = [
        ("artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_random.pt", "BigCarp Random"),
        ("artifacts/bigcarp/bigcarp_models/paper_models/bigcarp_esm1bfinetune.pt", "BigCarp ESM1b"),
    ]
    
    pooling_strategies = ['mean', 'cls']  # mean pooling vs class token
    
    all_results = []
    
    # Evaluate each combination
    for model_path, model_name in models_to_evaluate:
        if os.path.exists(model_path):
            for pooling in pooling_strategies:
                result = evaluate_architecture(
                    data, cv_splits, model_path, model_name, label_encoder,
                    specials, domains, args, pooling
                )
                
                if result is not None:
                    all_results.append(result)
                    print(f"✅ {model_name} - {pooling.upper()} completed successfully!")
                else:
                    print(f"❌ {model_name} - {pooling.upper()} failed!")
        else:
            print(f"⚠️  Model not found: {model_path}, skipping {model_name}")
    
    # Create comparison table
    if all_results:
        comparison_data = []
        for result in all_results:
            row = {'Model': result['model_name']}
            metrics = result['aggregate_metrics']
            row.update(metrics)
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        csv_path = f"{args.outdir}/architecture_comparison.csv"
        df_comparison.to_csv(csv_path, index=False)
        
        print(f"\n📋 ARCHITECTURE COMPARISON RESULTS")
        print("="*80)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Save complete results
        with open(f"{args.outdir}/complete_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n✅ Comparison completed!")
        print(f"📋 Results saved to: {args.outdir}/")
        print(f"📊 CSV: {csv_path}")
    else:
        print("❌ No successful evaluations!")

if __name__ == "__main__":
    main()