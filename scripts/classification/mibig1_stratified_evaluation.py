#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIBiG 1.0 Multi-Label Classification with Stratified Cross-Validation
================================================================
• Uses MIBiG 1.0 embedding files from artifacts/classification/mibig1/
• Stratified 5-fold CV for fair comparison  
• Multiple embedding approaches with BiLSTM models
• Includes pfam2vec with Random Forest and random baseline
"""

import os, json, argparse, pathlib, random, pickle, warnings
import numpy as np, pandas as pd
import sys
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             roc_auc_score, hamming_loss)
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Add path for your models  
sys.path.append('/home/u5bb/han00.u5bb/workspace/cgrep')
from cgrep.models_multiclass import MultiLabelBiLSTMClassifier  # legacy import; not used after switch

# Stratified multilabel CV
from skmultilearn.model_selection import IterativeStratification

# -------- Data Loading Functions -------------------------------------------
def load_mibig1_data(artifacts_dir="artifacts/classification/mibig1"):
    """Load MIBiG 1.0 embedding data."""
    print("📂 Loading MIBiG 1.0 embedding data...")
    
    # File paths
    esm_init_last = f"{artifacts_dir}/mibig1_esm_last.pkl"
    esm_init_embedder = f"{artifacts_dir}/mibig1_esm_embedder.pkl" 
    random_init_last = f"{artifacts_dir}/mibig1_random_last.pkl"
    random_init_embedder = f"{artifacts_dir}/mibig1_random_embedder.pkl"
    esm_embeddings = f"{artifacts_dir}/mibig_esm_embeddings.pkl"
    pfam2vec_embeddings = f"{artifacts_dir}/mibig_embeddings_p2v.pkl"
    
    data = {}
    files = [
        (esm_init_last, "esm_init_last"),
        (esm_init_embedder, "esm_init_embedder"), 
        (random_init_last, "random_init_last"),
        (random_init_embedder, "random_init_embedder"),
        (esm_embeddings, "esm_embeddings"),
        (pfam2vec_embeddings, "pfam2vec")
    ]
    
    for file_path, key in files:
        try:
            if os.path.exists(file_path):
                df = pd.read_pickle(file_path)
                data[key] = df
                print(f"   ✅ Loaded {key}: {df.shape}")
                print(f"      Columns: {list(df.columns)}")
            else:
                print(f"   ❌ File not found: {file_path}")
                data[key] = None
        except Exception as e:
            print(f"   ❌ Error loading {key}: {e}")
            data[key] = None
    
    return data

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

def tensor_to_list(x):
    """Convert tensor to list if needed."""
    if hasattr(x, 'detach'):  # torch tensor
        return x.detach().cpu().numpy().tolist()
    elif hasattr(x, 'tolist'):  # numpy array
        return x.tolist()
    return x

def prepare_embedding_data(data, embedding_type):
    """Prepare specific embedding type for evaluation."""
    print(f"\n🔧 Preparing {embedding_type} data...")
    
    if embedding_type == "esm_init_last":
        df = data["esm_init_last"]
        if df is not None and 'embeddings' in df.columns:
            df['embeddings'] = df['embeddings'].apply(tensor_to_list)
            return df[['bgc_id', 'embeddings', 'product_class']].copy(), 'embeddings'
    
    elif embedding_type == "esm_init_embedder":
        df = data["esm_init_embedder"] 
        if df is not None and 'embeddings' in df.columns:
            df['embeddings'] = df['embeddings'].apply(tensor_to_list)
            return df[['bgc_id', 'embeddings', 'product_class']].copy(), 'embeddings'
    
    elif embedding_type == "random_init_last":
        df = data["random_init_last"]
        if df is not None and 'embeddings' in df.columns:
            df['embeddings'] = df['embeddings'].apply(tensor_to_list)
            return df[['bgc_id', 'embeddings', 'product_class']].copy(), 'embeddings'
    
    elif embedding_type == "random_init_embedder":
        df = data["random_init_embedder"] 
        if df is not None and 'embeddings' in df.columns:
            df['embeddings'] = df['embeddings'].apply(tensor_to_list)
            return df[['bgc_id', 'embeddings', 'product_class']].copy(), 'embeddings'
    
    elif embedding_type == "esm_embeddings":
        df = data["esm_embeddings"]
        if df is not None and 'esm_embeddings' in df.columns:
            # Use sequence embeddings, not pooled
            df['esm_embeddings'] = df['esm_embeddings'].apply(tensor_to_list)
            return df[['bgc_id', 'esm_embeddings', 'product_class']].copy(), 'esm_embeddings'
    
    elif embedding_type == "pfam2vec":
        df = data["pfam2vec"]
        if df is not None and 'pfam2vec_seq' in df.columns:
            return df[['bgc_id', 'pfam2vec_seq', 'product_class']].copy(), 'pfam2vec_seq'
    
    elif embedding_type == "esm_bigcarp_concatenated":
        # Concatenate ESM and BigCarp embeddings
        esm_df = data["esm_embeddings"]
        bigcarp_df = data["random_init_last"]  # Use random as default
        if esm_df is not None and bigcarp_df is not None:
            print("   🔧 Creating concatenated ESM + BigCarp embeddings...")
            concat_df = create_concatenated_embeddings(esm_df, bigcarp_df)
            if concat_df is not None:
                return concat_df.copy(), 'concatenated_embeddings'
    
    print(f"   ❌ Failed to prepare {embedding_type} data")
    return None, None

def create_concatenated_embeddings(esm_df, bigcarp_df):
    """Create concatenated ESM + BigCarp embeddings."""
    # Find common BGC IDs
    common_ids = set(esm_df['bgc_id']) & set(bigcarp_df['bgc_id'])
    print(f"   Found {len(common_ids)} common BGC IDs for concatenation")
    
    if len(common_ids) == 0:
        print("   ❌ No common BGC IDs found for concatenation")
        return None
    
    # Filter to common IDs and align
    esm_filtered = esm_df[esm_df['bgc_id'].isin(common_ids)].set_index('bgc_id')
    bigcarp_filtered = bigcarp_df[bigcarp_df['bgc_id'].isin(common_ids)].set_index('bgc_id')
    
    concatenated_data = []
    for bgc_id in common_ids:
        try:
            esm_emb = esm_filtered.loc[bgc_id, 'esm_embeddings']
            bigcarp_emb = bigcarp_filtered.loc[bgc_id, 'embeddings']
            
            # Convert to list if needed
            esm_seq = tensor_to_list(esm_emb) if not isinstance(esm_emb, list) else esm_emb
            bigcarp_seq = tensor_to_list(bigcarp_emb) if not isinstance(bigcarp_emb, list) else bigcarp_emb
            
            # Ensure both are sequences
            if not (isinstance(esm_seq, list) and len(esm_seq) > 0 and isinstance(esm_seq[0], list)):
                continue
            if not (isinstance(bigcarp_seq, list) and len(bigcarp_seq) > 0 and isinstance(bigcarp_seq[0], list)):
                continue
                
            # Concatenate at each time step (assuming same sequence length)
            min_len = min(len(esm_seq), len(bigcarp_seq))
            concat_seq = []
            for i in range(min_len):
                concat_vec = esm_seq[i] + bigcarp_seq[i]  # Concatenate feature vectors
                concat_seq.append(concat_vec)
            
            concatenated_data.append({
                'bgc_id': bgc_id,
                'concatenated_embeddings': concat_seq,
                'product_class': esm_filtered.loc[bgc_id, 'product_class']
            })
            
        except Exception as e:
            print(f"   ⚠️  Failed to concatenate for {bgc_id}: {e}")
            continue
    
    if not concatenated_data:
        print("   ❌ No successful concatenations")
        return None
        
    concat_df = pd.DataFrame(concatenated_data)
    print(f"   ✅ Created {len(concat_df)} concatenated embeddings")
    
    # Check dimensions
    sample_concat = concat_df['concatenated_embeddings'].iloc[0]
    if isinstance(sample_concat, list) and len(sample_concat) > 0:
        concat_dim = len(sample_concat[0])
        print(f"   📊 Concatenated dimension: {concat_dim}")
    
    return concat_df

# -------- Validation & Coercion Helpers ---------------------------------------
def _coerce_to_seq2d(sample, emb_dim):
    """Coerce an embedding sample into a 2D sequence [T, D]."""
    if sample is None:
        return None

    # torch tensor -> numpy
    if hasattr(sample, 'detach'):
        try:
            sample = sample.detach().cpu().numpy()
        except Exception:
            return None

    # numpy array -> handle shapes
    if hasattr(sample, 'shape'):
        arr = np.asarray(sample)
        if arr.ndim == 2 and arr.shape[1] == emb_dim:
            return arr.tolist()
        if arr.ndim == 1 and arr.shape[0] == emb_dim:
            return arr.reshape(1, -1).tolist()
        return None

    # pure python list cases
    if isinstance(sample, list):
        if len(sample) == 0:
            return None
        first = sample[0]
        # list-of-lists (sequence) -> validate inner dim
        if isinstance(first, list):
            if len(first) == emb_dim:
                return sample
            else:
                return None
        # 1D vector -> treat as single-timestep sequence if length==emb_dim
        if not isinstance(first, list):
            if len(sample) == emb_dim and all(not isinstance(x, list) for x in sample):
                return [sample]
            else:
                return None

    return None

def _validate_and_prepare_XY(X_raw, y_raw, emb_dim, max_bad_report=5, split_name="train"):
    """Validate & coerce X to [T,D] lists; drop invalid pairs."""
    X, y = [], []
    bad = 0
    for i, (xi, yi) in enumerate(zip(X_raw, y_raw)):
        coerced = _coerce_to_seq2d(xi, emb_dim)
        if coerced is None:
            if bad < max_bad_report:
                print(f"   ⚠️  Dropping {split_name} sample #{i}: invalid shape/type for embedding")
            bad += 1
            continue
        X.append(coerced)
        y.append(yi)
    if bad > 0:
        print(f"   🧹 {split_name.capitalize()} cleanup: kept {len(X)}/{len(X_raw)} (dropped {bad})")
    return X, y, len(X), bad

# -------- Stratified CV Functions ------------------------------------------
def create_stratified_splits(df, class_cols, n_splits=5, random_state=42):
    """Create stratified CV splits for multi-label classification."""
    y_binary = df[class_cols].values
    
    print(f"\n📊 Class Distribution Analysis:")
    class_counts = y_binary.sum(axis=0)
    for i, col in enumerate(class_cols):
        print(f"  {col}: {class_counts[i]:4d} samples ({class_counts[i]/len(df)*100:5.1f}%)")
    
    try:
        indices = np.arange(len(df))
        if random_state is not None:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        
        stratifier = IterativeStratification(
            n_splits=n_splits, 
            order=2,
            sample_distribution_per_fold=[1.0/n_splits]*n_splits,
            random_state=random_state
        )
        splits = list(stratifier.split(indices, y_binary[indices]))
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

# -------- Metrics Functions ------------------------------------------------
def exact_match_accuracy(y_true, y_pred):
    """Exact match accuracy - all labels must be predicted correctly."""
    return np.mean(np.all(y_true == y_pred, axis=1))

def compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
    """Compute all requested metrics including per-class AUC-ROC."""
    metrics = {}
    
    # 1. Exact match accuracy
    metrics['exact_match_accuracy'] = exact_match_accuracy(y_true, y_pred)
    
    # 2. Micro F1
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    
    # 3. Macro F1 
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    
    # 4. Weighted Macro F1
    metrics['weighted_macro_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # 5. Micro AUC
    try:
        metrics['micro_auc'] = roc_auc_score(y_true.ravel(), y_proba.ravel())
    except ValueError:
        metrics['micro_auc'] = float('nan')
    
    # 6. Macro AUC
    try:
        metrics['macro_auc'] = roc_auc_score(y_true, y_proba, average='macro')
    except ValueError:
        metrics['macro_auc'] = float('nan')
    
    # 7. Weighted AUC
    try:
        metrics['weighted_auc'] = roc_auc_score(y_true, y_proba, average='weighted')
    except ValueError:
        metrics['weighted_auc'] = float('nan')
    
    # 8. Per-class AUC-ROC
    per_class_auc = {}
    per_class_support = {}
    for i, class_name in enumerate(class_names):
        if i < y_true.shape[1]:
            class_true = y_true[:, i]
            class_proba = y_proba[:, i]
            support = int(np.sum(class_true))
            
            per_class_support[class_name] = support
            
            # Calculate AUC only if both classes present
            if len(np.unique(class_true)) > 1:
                try:
                    auc = roc_auc_score(class_true, class_proba)
                    per_class_auc[class_name] = auc
                except ValueError:
                    per_class_auc[class_name] = float('nan')
            else:
                per_class_auc[class_name] = float('nan')
    
    metrics['per_class_auc'] = per_class_auc
    metrics['per_class_support'] = per_class_support
    
    return metrics

# -------- Model Evaluation -------------------------------------------------
def evaluate_mlp_model(df, cv_splits, emb_col, emb_dim, model_name, class_cols, seed=42):
    """Evaluate mean-pooled embeddings with a shallow MLP (two layers) via One-vs-Rest."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"   Column: {emb_col} (dim={emb_dim})")
    print(f"{'='*70}")
    
    # Prepare multi-label strings  
    label_strings = [";".join([c for c in class_cols if row[c]==1])
                     for _, row in df.iterrows()]
    
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.multiclass import OneVsRestClassifier

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n📁 Fold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")

        # Prepare data (raw)
        X_train_raw = [df.iloc[i][emb_col] for i in train_idx]
        X_test_raw  = [df.iloc[i][emb_col] for i in test_idx]
        y_train_raw = [label_strings[i] for i in train_idx]
        y_test_raw  = [label_strings[i] for i in test_idx]

        # Coerce/validate shapes -> drop bad samples 
        X_train, y_train, kept_tr, drop_tr = _validate_and_prepare_XY(X_train_raw, y_train_raw, emb_dim, split_name="train")
        X_test,  y_test,  kept_te, drop_te = _validate_and_prepare_XY(X_test_raw,  y_test_raw,  emb_dim, split_name="test")

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"   ❌ Not enough samples after validation (train={len(X_train)}, test={len(X_test)})")
            continue

        # Mean-pool to fixed vectors
        Xtr = create_mean_pooled_features(X_train)
        Xte = create_mean_pooled_features(X_test)
        
        # Labels -> multilabel
        mlb = MultiLabelBinarizer()
        Ytr = mlb.fit_transform([s.split(';') if s else [] for s in y_train])
        Yte = mlb.transform([s.split(';') if s else [] for s in y_test])

        # Train MLP
        try:
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            # Optional PCA to help convergence and speed
            pca = None
            target_dim = 256 if Xtr_s.shape[1] > 256 else None
            if target_dim is not None:
                pca = PCA(n_components=target_dim, random_state=seed)
                Xtr_s = pca.fit_transform(Xtr_s)
                Xte_s = pca.transform(Xte_s)

            # Use lbfgs with moderate regularization and relaxed tolerance
            base = MLPClassifier(
                hidden_layer_sizes=(256, 128), activation='relu',
                alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
                early_stopping=True, n_iter_no_change=10, random_state=seed,
                verbose=False
            )
            clf = OneVsRestClassifier(base, n_jobs=-1)
            clf.fit(Xtr_s, Ytr)
            y_proba = clf.predict_proba(Xte_s)
            y_true = Yte
            y_pred = (y_proba > 0.5).astype(int)

            # Compute fold metrics
            print(f"   📊 Computing metrics...")
            fold_metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, mlb.classes_)
            fold_metrics['fold'] = fold_idx
            # Store raw predictions for per-class analysis
            fold_metrics['y_true'] = y_true
            fold_metrics['y_pred'] = y_pred
            fold_metrics['y_proba'] = y_proba
            fold_results.append(fold_metrics)
            
            # Collect for aggregate
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
            print(f"   📊 Exact Match: {fold_metrics['exact_match_accuracy']:.4f}, "
                  f"Macro F1: {fold_metrics['macro_f1']:.4f}, "
                  f"Macro AUC: {fold_metrics['macro_auc']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error in fold {fold_idx}: {e}")
            continue
    
    if not all_y_true:
        print(f"❌ No successful folds for {model_name}")
        return None
    
    # Aggregate all folds
    print(f"\n   📈 Aggregating results across all folds...")
    aggregate_y_true = np.vstack(all_y_true)
    aggregate_y_pred = np.vstack(all_y_pred)
    aggregate_y_proba = np.vstack(all_y_proba)
    
    aggregate_metrics = compute_comprehensive_metrics(
        aggregate_y_true, aggregate_y_pred, aggregate_y_proba, mlb.classes_
    )
    
    print(f"\n🎯 {model_name} - Final Results:")
    print(f"   Exact Match Accuracy: {aggregate_metrics['exact_match_accuracy']:.4f}")
    print(f"   Macro F1: {aggregate_metrics['macro_f1']:.4f}")
    print(f"   Macro AUC: {aggregate_metrics['macro_auc']:.4f}")
    
    return {
        'model_name': model_name,
        'embedding_column': emb_col,
        'fold_results': fold_results,
        'aggregate_metrics': aggregate_metrics,
        'class_names': mlb.classes_.tolist()
    }

def create_mean_pooled_features(embed_sequences):
    """Convert variable-length embedding sequences to fixed-length features by mean pooling."""
    pooled_features = []
    fallback_dim = None
    
    # First pass: determine embedding dimension
    for seq in embed_sequences:
        if seq is not None and len(seq) > 0:
            if isinstance(seq[0], (list, np.ndarray)) and len(seq[0]) > 0:
                fallback_dim = len(seq[0])
                break
    
    # Second pass: create pooled features
    for seq in embed_sequences:
        if not isinstance(seq, (list, np.ndarray)) or len(seq) == 0 or seq is None:
            # Handle empty/invalid sequences
            if fallback_dim is not None:
                pooled_features.append(np.zeros(fallback_dim))
            else:
                continue
        else:
            try:
                current_seq_np = np.array(seq, dtype=float)
                if current_seq_np.ndim == 1:
                    # Single vector
                    pooled = current_seq_np
                elif current_seq_np.ndim == 2:
                    # Sequence of vectors - mean pool
                    pooled = np.mean(current_seq_np, axis=0)
                else:
                    # Unexpected shape
                    if fallback_dim is not None:
                        pooled_features.append(np.zeros(fallback_dim))
                    continue
                
                pooled_features.append(pooled)
                if fallback_dim is None and pooled.size > 0:
                    fallback_dim = pooled.shape[0]
            except Exception as e:
                if fallback_dim is not None:
                    pooled_features.append(np.zeros(fallback_dim))
    
    if not pooled_features:
        return np.empty((0, fallback_dim if fallback_dim is not None else 1))
    
    return np.stack(pooled_features)

def evaluate_pfam2vec_rf(df, cv_splits, class_cols, seed=42):
    """Evaluate P2V embeddings with Random Forest classifier."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: Pfam2vec + Random Forest")
    print(f"{'='*70}")
    
    # Check for pfam2vec_seq column
    if 'pfam2vec_seq' not in df.columns:
        print("❌ pfam2vec_seq column not found!")
        return None
    
    # Create mean-pooled features from pfam2vec_seq
    print(f"   🔄 Creating mean-pooled features from pfam2vec_seq...")
    X_p2v = create_mean_pooled_features(df['pfam2vec_seq'].values)
    print(f"   P2V feature matrix shape: {X_p2v.shape}")
    
    if X_p2v.shape[0] == 0:
        print("❌ No valid P2V features created!")
        return None
    
    # Prepare multi-label strings  
    label_strings = [";".join([c for c in class_cols if row[c]==1])
                     for _, row in df.iterrows()]
    
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform([s.split(';') if s else [] for s in label_strings])
    
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n📁 Fold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")
        
        X_train, X_test = X_p2v[train_idx], X_p2v[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        
        try:
            # Train separate Random Forest for each class
            class_predictions = np.zeros((len(test_idx), len(class_cols)))
            class_probabilities = np.zeros((len(test_idx), len(class_cols)))
            
            for class_idx, class_name in enumerate(class_cols):
                # Skip if only one class present in training
                if len(np.unique(y_train[:, class_idx])) <= 1:
                    continue
                
                rf_classifier = RandomForestClassifier(
                    n_estimators=100, 
                    max_features='sqrt', 
                    random_state=seed, 
                    n_jobs=-1
                )
                rf_classifier.fit(X_train, y_train[:, class_idx])
                prob_pred = rf_classifier.predict_proba(X_test)
                
                if prob_pred.shape[1] == 2:
                    class_probabilities[:, class_idx] = prob_pred[:, 1]
                elif prob_pred.shape[1] == 1:
                    if rf_classifier.classes_[0] == 1:
                        class_probabilities[:, class_idx] = 1.0
                    else:
                        class_probabilities[:, class_idx] = 0.0
                
                class_predictions[:, class_idx] = (class_probabilities[:, class_idx] > 0.5).astype(int)
            
            # Compute fold metrics
            fold_metrics = compute_comprehensive_metrics(y_test, class_predictions, class_probabilities, mlb.classes_)
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
            
            # Collect for aggregate
            all_y_true.append(y_test)
            all_y_pred.append(class_predictions)
            all_y_proba.append(class_probabilities)
            
            print(f"   📊 Exact Match: {fold_metrics['exact_match_accuracy']:.4f}, "
                  f"Macro F1: {fold_metrics['macro_f1']:.4f}, "
                  f"Macro AUC: {fold_metrics['macro_auc']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error in fold {fold_idx}: {e}")
            continue
    
    if not all_y_true:
        return None
    
    # Aggregate all folds
    aggregate_y_true = np.vstack(all_y_true)
    aggregate_y_pred = np.vstack(all_y_pred)
    aggregate_y_proba = np.vstack(all_y_proba)
    
    aggregate_metrics = compute_comprehensive_metrics(
        aggregate_y_true, aggregate_y_pred, aggregate_y_proba, mlb.classes_
    )
    
    print(f"\n🎯 Pfam2vec + Random Forest - Final Results:")
    print(f"   Exact Match Accuracy: {aggregate_metrics['exact_match_accuracy']:.4f}")
    print(f"   Macro F1: {aggregate_metrics['macro_f1']:.4f}")
    print(f"   Macro AUC: {aggregate_metrics['macro_auc']:.4f}")
    
    return {
        'model_name': "Pfam2vec + Random Forest",
        'embedding_column': f"pfam2vec_seq ({X_p2v.shape[1]} features)",
        'fold_results': fold_results,
        'aggregate_metrics': aggregate_metrics,
        'class_names': mlb.classes_.tolist()
    }

def evaluate_pfam2vec_mlp(df, cv_splits, class_cols, seed=42):
    """Evaluate P2V embeddings with shallow MLP classifier."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: Pfam2vec + MLP")
    print(f"{'='*70}")
    
    # Check for pfam2vec_seq column
    if 'pfam2vec_seq' not in df.columns:
        print("❌ pfam2vec_seq column not found!")
        return None
    
    # Create mean-pooled features from pfam2vec_seq
    print(f"   🔄 Creating mean-pooled features from pfam2vec_seq...")
    X_p2v = create_mean_pooled_features(df['pfam2vec_seq'].values)
    print(f"   P2V feature matrix shape: {X_p2v.shape}")
    
    if X_p2v.shape[0] == 0:
        print("❌ No valid P2V features created!")
        return None
    
    # Prepare multi-label strings  
    label_strings = [";".join([c for c in class_cols if row[c]==1])
                     for _, row in df.iterrows()]
    
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.multiclass import OneVsRestClassifier
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n📁 Fold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")
        
        X_train, X_test = X_p2v[train_idx], X_p2v[test_idx]
        y_train_strings = [label_strings[i] for i in train_idx]
        y_test_strings = [label_strings[i] for i in test_idx]
        
        try:
            # Labels -> multilabel
            mlb = MultiLabelBinarizer()
            y_train = mlb.fit_transform([s.split(';') if s else [] for s in y_train_strings])
            y_test = mlb.transform([s.split(';') if s else [] for s in y_test_strings])
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Optional PCA to help convergence and speed
            pca = None
            target_dim = 256 if X_train_scaled.shape[1] > 256 else None
            if target_dim is not None:
                pca = PCA(n_components=target_dim, random_state=seed)
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)
            
            # Train shallow MLP
            base = MLPClassifier(
                hidden_layer_sizes=(256, 128), activation='relu',
                alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
                early_stopping=True, n_iter_no_change=10, random_state=seed,
                verbose=False
            )
            clf = OneVsRestClassifier(base, n_jobs=-1)
            clf.fit(X_train_scaled, y_train)
            y_proba = clf.predict_proba(X_test_scaled)
            y_pred = (y_proba > 0.5).astype(int)
            
            # Compute fold metrics
            fold_metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba, mlb.classes_)
            fold_metrics['fold'] = fold_idx
            # Store raw predictions for per-class analysis
            fold_metrics['y_true'] = y_test
            fold_metrics['y_pred'] = y_pred
            fold_metrics['y_proba'] = y_proba
            fold_results.append(fold_metrics)
            
            # Collect for aggregate
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
            print(f"   📊 Exact Match: {fold_metrics['exact_match_accuracy']:.4f}, "
                  f"Macro F1: {fold_metrics['macro_f1']:.4f}, "
                  f"Macro AUC: {fold_metrics['macro_auc']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error in fold {fold_idx}: {e}")
            continue
    
    if not all_y_true:
        return None
    
    # Aggregate all folds
    aggregate_y_true = np.vstack(all_y_true)
    aggregate_y_pred = np.vstack(all_y_pred)
    aggregate_y_proba = np.vstack(all_y_proba)
    
    aggregate_metrics = compute_comprehensive_metrics(
        aggregate_y_true, aggregate_y_pred, aggregate_y_proba, mlb.classes_
    )
    
    print(f"\n🎯 Pfam2vec + MLP - Final Results:")
    print(f"   Exact Match Accuracy: {aggregate_metrics['exact_match_accuracy']:.4f}")
    print(f"   Macro F1: {aggregate_metrics['macro_f1']:.4f}")
    print(f"   Macro AUC: {aggregate_metrics['macro_auc']:.4f}")
    
    return {
        'model_name': "Pfam2vec + MLP",
        'embedding_column': f"pfam2vec_seq ({X_p2v.shape[1]} features)",
        'fold_results': fold_results,
        'aggregate_metrics': aggregate_metrics,
        'class_names': mlb.classes_.tolist()
    }

def evaluate_random_baseline(df, cv_splits, class_cols, seed=42):
    """Evaluate random 256-dimensional baseline with shallow MLP (mean-pooled)."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: Random 256D Baseline")
    print(f"{'='*70}")
    
    # Create random embeddings for each BGC
    np.random.seed(seed)
    random_embeddings = []
    for i in range(len(df)):
        # Create random sequence of random length (5-50 timesteps)
        seq_len = np.random.randint(5, 51)
        random_seq = np.random.randn(seq_len, 256).tolist()
        random_embeddings.append(random_seq)
    
    df_random = df.copy()
    df_random['random_embeddings'] = random_embeddings
    
    return evaluate_mlp_model(
        df_random, cv_splits, 'random_embeddings', 256,
        "Random 256D Baseline", class_cols, seed
    )

def evaluate_improved_random_baseline(df, cv_splits, class_cols, seed=42):
    """Evaluate improved random baseline with fixed random embeddings per PFM domain."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: Improved Random Baseline (Fixed per PFM Domain)")
    print(f"{'='*70}")
    
    # Load PFM domain sequences from the original MIBiG data
    try:
        # Try to load the original MIBiG data to get PFM domain sequences
        mibig_path = 'data/raw/MiBIG_1406_dataset.txt'
        df_mibig_raw = pd.read_csv(mibig_path, header=None, names=['bgc_id', 'product_class', 'pfm_sequence'])
        
        # Create mapping from BGC ID to PFM sequence
        pfm_sequences = {}
        for _, row in df_mibig_raw.iterrows():
            bgc_id = row['bgc_id']
            pfm_seq = row['pfm_sequence'].split(';') if pd.notna(row['pfm_sequence']) else []
            pfm_sequences[bgc_id] = pfm_seq
        
        print(f"   📊 Loaded PFM sequences for {len(pfm_sequences)} BGCs")
        
    except Exception as e:
        print(f"   ⚠️  Could not load PFM sequences: {e}")
        print(f"   🔄 Falling back to original random baseline...")
        return evaluate_random_baseline(df, cv_splits, class_cols, seed)
    
    # Create fixed random embeddings for each unique PFM domain
    np.random.seed(seed)
    unique_domains = set()
    for pfm_seq in pfm_sequences.values():
        unique_domains.update(pfm_seq)
    
    print(f"   📊 Found {len(unique_domains)} unique PFM domains")
    
    # Create fixed random embedding for each domain
    domain_embeddings = {}
    for domain in unique_domains:
        domain_embeddings[domain] = np.random.randn(256).tolist()
    
    print(f"   🎲 Created fixed random embeddings for {len(domain_embeddings)} domains")
    
    # Create random embeddings for each BGC based on their PFM sequence
    random_embeddings = []
    missing_domains = set()
    
    for _, row in df.iterrows():
        bgc_id = row['bgc_id']
        if bgc_id in pfm_sequences:
            pfm_seq = pfm_sequences[bgc_id]
            # Create sequence of random embeddings based on PFM domains
            bgc_embeddings = []
            for domain in pfm_seq:
                if domain in domain_embeddings:
                    bgc_embeddings.append(domain_embeddings[domain])
                else:
                    missing_domains.add(domain)
                    # Use UNK domain embedding if available, otherwise random
                    if 'UNK' in domain_embeddings:
                        bgc_embeddings.append(domain_embeddings['UNK'])
                    else:
                        bgc_embeddings.append(np.random.randn(256).tolist())
            
            random_embeddings.append(bgc_embeddings)
        else:
            # Fallback: create random sequence if BGC not found
            seq_len = np.random.randint(5, 51)
            random_seq = [np.random.randn(256).tolist() for _ in range(seq_len)]
            random_embeddings.append(random_seq)
    
    if missing_domains:
        print(f"   ⚠️  Missing embeddings for {len(missing_domains)} domains: {list(missing_domains)[:10]}...")
    
    df_random = df.copy()
    df_random['improved_random_embeddings'] = random_embeddings
    
    print(f"   📊 Created improved random embeddings for {len(random_embeddings)} BGCs")
    
    return evaluate_mlp_model(
        df_random, cv_splits, 'improved_random_embeddings', 256,
        "Improved Random Baseline (Fixed per PFM Domain)", class_cols, seed
    )

def save_per_class_results(all_results, outdir):
    """Save per-class AUC-ROC results as CSV files."""
    per_class_data = []
    
    for result in all_results:
        model_name = result['model_name']
        
        # Aggregate per-class metrics from all folds
        if 'aggregate_metrics' in result:
            agg_metrics = result['aggregate_metrics']
            
            if 'per_class_auc' in agg_metrics and 'per_class_support' in agg_metrics:
                per_class_auc = agg_metrics['per_class_auc']
                per_class_support = agg_metrics['per_class_support']
                
                for class_name in per_class_auc:
                    auc_score = per_class_auc[class_name]
                    support = per_class_support.get(class_name, 0)
                    
                    per_class_data.append({
                        'Model': model_name,
                        'Class': class_name,
                        'AUC_ROC': auc_score,
                        'Support': support
                    })
    
    if per_class_data:
        per_class_df = pd.DataFrame(per_class_data)
        
        # Calculate frequency
        total_samples = per_class_df.groupby('Model')['Support'].sum().iloc[0] if not per_class_df.empty else 1
        per_class_df['Frequency'] = per_class_df['Support'] / total_samples
        
        # Identify minor classes (< 5% frequency)
        class_freq = per_class_df.groupby('Class')['Frequency'].first()
        minor_classes = class_freq[class_freq < 0.05].index.tolist()
        per_class_df['Is_Minor_Class'] = per_class_df['Class'].isin(minor_classes)
        
        # Save detailed per-class results
        per_class_df.to_csv(f"{outdir}/per_class_detailed.csv", index=False)
        
        # Save summary by class (average across models)
        class_summary = per_class_df.groupby('Class').agg({
            'AUC_ROC': ['mean', 'std', 'min', 'max'],
            'Support': 'first',
            'Frequency': 'first',
            'Is_Minor_Class': 'first'
        }).round(4)
        class_summary.to_csv(f"{outdir}/per_class_summary.csv")
        
        # Save minor class analysis
        minor_df = per_class_df[per_class_df['Is_Minor_Class']]
        if not minor_df.empty:
            minor_summary = minor_df.groupby('Model')['AUC_ROC'].agg(['mean', 'std', 'count']).round(4)
            minor_summary.to_csv(f"{outdir}/minor_class_performance.csv")
        
        print(f"📊 Per-class analysis saved:")
        print(f"   - per_class_detailed.csv ({len(per_class_df)} entries)")
        print(f"   - per_class_summary.csv ({len(class_summary)} classes)")
        if not minor_df.empty:
            print(f"   - minor_class_performance.csv ({len(minor_classes)} minor classes)")
    else:
        print("⚠️  No per-class data to save")

# -------- Results Analysis -------------------------------------------------
def create_comparison_table(all_results, outdir):
    """Create comparison table."""
    if not all_results:
        print("❌ No results to compare")
        return None
    
    main_metrics = ['macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy']
    
    comparison_data = []
    for result in all_results:
        if result is not None:
            row = {'Model': result['model_name']}
            metrics = result['aggregate_metrics']
            for metric in main_metrics:
                row[metric] = metrics[metric]
            comparison_data.append(row)
    
    if not comparison_data:
        print("❌ No valid comparison data")
        return None
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print(f"\n{'='*80}")
    print("🏆 MODEL COMPARISON - MIBiG 1.0 RESULTS")
    print(f"{'='*80}")
    
    print("\n📊 Performance Summary:")
    display_df = df_comparison.copy()
    display_df.columns = ['Model', 'Macro F1', 'Macro AUC', 'Weighted AUC', 'Exact Accuracy']
    print(display_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n🥇 Best Performance:")
    for i, metric in enumerate(['Macro F1', 'Macro AUC', 'Weighted AUC', 'Exact Accuracy']):
        col_name = display_df.columns[i+1]
        best_idx = display_df[col_name].idxmax()
        best_model = display_df.loc[best_idx, 'Model']
        best_value = display_df.loc[best_idx, col_name]
        print(f"   {metric}: {best_model} ({best_value:.4f})")
    
    # Save results
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(f"{outdir}/mibig1_comparison.csv", index=False)
    
    return df_comparison

# -------- Main Pipeline ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MIBiG 1.0 Multi-Label Classification with Stratified CV")
    parser.add_argument("--artifacts_dir", default="artifacts/classification/mibig1",
                        help="Directory containing MIBiG 1.0 embedding files")
    parser.add_argument("--outdir", default="results/mibig1_classification",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    print(f"🎲 Setting random seed to {args.seed} for reproducibility")
    import random
    import torch
    
    # Set all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch is not None and hasattr(torch, 'manual_seed'):
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # For sklearn reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    print("🚀 MIBiG 1.0 Multi-Label Classification with Stratified CV")
    print(f"📁 Artifacts: {args.artifacts_dir}")
    print(f"📁 Output: {args.outdir}")
    print(f"🌱 Random seed: {args.seed}")
    
    # Load data
    data = load_mibig1_data(args.artifacts_dir)
    
    # Check if we have any data
    available_data = [k for k, v in data.items() if v is not None]
    if not available_data:
        print("❌ No embedding data found!")
        return
    
    print(f"📊 Available embeddings: {available_data}")
    
    # Define models to evaluate
    models_to_evaluate = [
        ("esm_init_last", "ESM Init Last + MLP"),
        ("esm_init_embedder", "ESM Init Embedder + MLP"),
        ("random_init_last", "Random Init Last + MLP"),
        ("random_init_embedder", "Random Init Embedder + MLP"),
        ("esm_embeddings", "ESM Embeddings + MLP"),
        ("esm_bigcarp_concatenated", "ESM + BigCarp Concatenated + MLP"),
    ]
    
    all_results = []
    
    # Evaluate BiLSTM models
    for embedding_type, model_name in models_to_evaluate:
        # Special handling for concatenated embeddings
        if embedding_type == "esm_bigcarp_concatenated":
            # Check if both ESM and BigCarp data are available
            if data["esm_embeddings"] is not None and data["random_init_last"] is not None:
                print(f"\n{'🔬' * 3} EVALUATING {model_name.upper()} {'🔬' * 3}")
            else:
                print(f"⚠️  {embedding_type}: ESM or BigCarp data not available, skipping {model_name}")
                continue
        elif data[embedding_type] is not None:
            print(f"\n{'🔬' * 3} EVALUATING {model_name.upper()} {'🔬' * 3}")
        else:
            print(f"⚠️  {embedding_type} data not available, skipping {model_name}")
            continue
            
        # Prepare data (common for all cases)
        df_prep, emb_col = prepare_embedding_data(data, embedding_type)
        if df_prep is None:
            print(f"❌ Failed to prepare {embedding_type} data")
            continue
        
        # Convert to binary classes
        df_prep, class_cols = convert_product_classes_to_binary(df_prep)
        
        # Create CV splits
        cv_splits = create_stratified_splits(df_prep, class_cols, n_splits=5, random_state=args.seed)
        
        # Determine embedding dimension
        sample_emb = df_prep[emb_col].dropna().iloc[0]
        if isinstance(sample_emb, list) and len(sample_emb) > 0:
            if isinstance(sample_emb[0], list):
                emb_dim = len(sample_emb[0])
            else:
                emb_dim = len(sample_emb)
        else:
            print(f"❌ Cannot determine embedding dimension for {embedding_type}")
            continue
        
        print(f"   Embedding dimension: {emb_dim}")
        
        # Evaluate model
        result = evaluate_mlp_model(
            df_prep, cv_splits, emb_col, emb_dim, model_name, class_cols, args.seed
        )
        
        if result is not None:
            all_results.append(result)
            print(f"✅ {model_name} completed successfully!")
        else:
            print(f"❌ {model_name} failed!")
    
    # Evaluate Pfam2vec models
    if data["pfam2vec"] is not None:
        print(f"\n{'🔬' * 3} EVALUATING PFAM2VEC MODELS {'🔬' * 3}")
        
        df_prep, emb_col = prepare_embedding_data(data, "pfam2vec")
        if df_prep is not None:
            df_prep, class_cols = convert_product_classes_to_binary(df_prep)
            cv_splits = create_stratified_splits(df_prep, class_cols, n_splits=5, random_state=args.seed)
            
            # Pfam2vec + Random Forest
            result = evaluate_pfam2vec_rf(df_prep, cv_splits, class_cols, args.seed)
            if result is not None:
                all_results.append(result)
                print(f"✅ Pfam2vec + Random Forest completed successfully!")
            else:
                print(f"❌ Pfam2vec + Random Forest failed!")
            
            # Pfam2vec + MLP
            result = evaluate_pfam2vec_mlp(df_prep, cv_splits, class_cols, args.seed)
            if result is not None:
                all_results.append(result)
                print(f"✅ Pfam2vec + MLP completed successfully!")
            else:
                print(f"❌ Pfam2vec + MLP failed!")
    
    # Evaluate Random Baselines
    print(f"\n{'🔬' * 3} EVALUATING RANDOM BASELINES {'🔬' * 3}")
    
    # Use any available data for baseline (just need the labels)
    baseline_data = None
    for key in ["esm_init_last", "esm_embeddings", "pfam2vec"]:
        if data[key] is not None:
            baseline_data = data[key]
            break
    
    if baseline_data is not None:
        df_baseline = baseline_data[['bgc_id', 'product_class']].copy()
        df_baseline, class_cols = convert_product_classes_to_binary(df_baseline)
        cv_splits = create_stratified_splits(df_baseline, class_cols, n_splits=5, random_state=args.seed)
        
        # Original Random Baseline
        print(f"\n{'='*50}")
        print(f"🔬 Original Random Baseline")
        print(f"{'='*50}")
        result = evaluate_random_baseline(df_baseline, cv_splits, class_cols, args.seed)
        if result is not None:
            all_results.append(result)
            print(f"✅ Original Random Baseline completed successfully!")
        else:
            print(f"❌ Original Random Baseline failed!")
        
        # Improved Random Baseline
        print(f"\n{'='*50}")
        print(f"🔬 Improved Random Baseline")
        print(f"{'='*50}")
        result = evaluate_improved_random_baseline(df_baseline, cv_splits, class_cols, args.seed)
        if result is not None:
            all_results.append(result)
            print(f"✅ Improved Random Baseline completed successfully!")
        else:
            print(f"❌ Improved Random Baseline failed!")
    
    # Create final comparison
    if all_results:
        create_comparison_table(all_results, args.outdir)
        
        # Save complete results
        with open(f"{args.outdir}/complete_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save per-class results as CSV for easy analysis
        save_per_class_results(all_results, args.outdir)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📋 Results saved to: {args.outdir}/")
    else:
        print("❌ No successful evaluations!")

if __name__ == "__main__":
    main()
