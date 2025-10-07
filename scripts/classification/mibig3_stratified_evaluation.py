#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIBiG 3.0 Multi-Label Classification with Stratified Cross-Validation
======================================================================

This script evaluates multiple embedding approaches for BGC product class prediction
using MIBiG 3.0 data with stratified 5-fold cross-validation.

Embedding Types Evaluated:
    - ESM-initialized BigCarp (last layer & embedder)
    - Random-initialized BigCarp (last layer & embedder)
    - ESM embeddings (pretrained)
    - ESM + BigCarp concatenated
    - Pfam2vec embeddings
    - Random baselines (simple and domain-consistent)

Models:
    - Multi-layer perceptron (MLP) with One-vs-Rest strategy
    - Random Forest (for Pfam2vec)

Metrics:
    - Exact match accuracy
    - Macro/Micro/Weighted F1 scores
    - Macro/Micro/Weighted AUC-ROC
    - Per-class AUC-ROC

Usage:
    python mibig3_stratified_evaluation.py --artifacts_dir <path> --outdir <path> --seed <int>

Output Files:
    - mibig3_comparison.csv: Aggregate performance across all models
    - complete_results.pkl: Full results including fold-level metrics
    - per_class_detailed.csv: Per-class AUC-ROC for each model
"""

import os, argparse, pathlib, random, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.model_selection import IterativeStratification

# -------- Data Loading Functions -------------------------------------------

def load_mibig3_data(artifacts_dir="artifacts/classification/mibig3"):
    """
    Load MIBiG 3.0 embedding data from pickle files.

    Args:
        artifacts_dir (str): Directory containing embedding pickle files.
            Expected files:
                - esm_init/mibig3_bigcarp_last.pkl
                - esm_init/mibig3_bigcarp_embedder.pkl
                - random_init/mibig3_bigcarp_last.pkl
                - random_init/mibig3_bigcarp_embedder.pkl
                - mibig3_esm_embeddings.pkl
                - mibig3_pfam2vec_embeddings.pkl

    Returns:
        dict: Dictionary mapping embedding type keys to DataFrames.
            Keys: 'esm_init_last', 'esm_init_embedder', 'random_init_last',
                  'random_init_embedder', 'esm_embeddings', 'pfam2vec'
            Values: pandas.DataFrame with columns ['bgc_id', embedding_col, 'product_class']
                   or None if file not found
    """
    print("Loading MIBiG 3.0 embedding data...")

    data = {}
    files = [
        (f"{artifacts_dir}/esm_init/mibig3_bigcarp_last.pkl", "esm_init_last"),
        (f"{artifacts_dir}/esm_init/mibig3_bigcarp_embedder.pkl", "esm_init_embedder"),
        (f"{artifacts_dir}/random_init/mibig3_bigcarp_last.pkl", "random_init_last"),
        (f"{artifacts_dir}/random_init/mibig3_bigcarp_embedder.pkl", "random_init_embedder"),
        (f"{artifacts_dir}/mibig3_esm_embeddings.pkl", "esm_embeddings"),
        (f"{artifacts_dir}/mibig3_pfam2vec_embeddings.pkl", "pfam2vec")
    ]

    for file_path, key in files:
        if os.path.exists(file_path):
            data[key] = pd.read_pickle(file_path)
            print(f"   [OK] Loaded {key}: {data[key].shape}")
        else:
            data[key] = None

    return data

def convert_product_classes_to_binary(df):
    """
    Convert semicolon-separated product class strings to binary indicator columns.

    Args:
        df (pandas.DataFrame): DataFrame with 'product_class' column containing
            semicolon-separated class labels (e.g., "Polyketide;Terpene")

    Returns:
        tuple: (df, class_cols) where:
            - df (pandas.DataFrame): Input DataFrame with added binary columns,
              one column per unique class with 1/0 values
            - class_cols (list): Sorted list of unique class names
    """
    df = df.copy()
    all_classes = set()
    for class_str in df['product_class'].dropna():
        if pd.notna(class_str) and class_str != '':
            all_classes.update(cls.strip() for cls in str(class_str).split(';') if cls.strip())

    all_classes = sorted(list(all_classes))

    for class_name in all_classes:
        df[class_name] = df['product_class'].apply(
            lambda x: 1 if pd.notna(x) and class_name in str(x).split(';') else 0
        )

    return df, all_classes

def tensor_to_list(x):
    """
    Convert PyTorch tensor or numpy array to Python list.

    Args:
        x: PyTorch tensor, numpy array, or any object with .tolist() method

    Returns:
        list or original type: Converted list if input is tensor/array, otherwise returns input unchanged
    """
    if hasattr(x, 'detach'):  # torch tensor
        return x.detach().cpu().numpy().tolist()
    elif hasattr(x, 'tolist'):  # numpy array
        return x.tolist()
    return x

def prepare_embedding_data(data, embedding_type):
    """
    Extract and prepare specific embedding type for model evaluation.

    Args:
        data (dict): Dictionary from load_mibig3_data() containing all embeddings
        embedding_type (str): Type of embedding to prepare. Options:
            - 'esm_init_last', 'esm_init_embedder'
            - 'random_init_last', 'random_init_embedder'
            - 'esm_embeddings', 'pfam2vec'
            - 'esm_bigcarp_concatenated' (special case)

    Returns:
        tuple: (df, emb_col) where:
            - df (pandas.DataFrame): DataFrame with ['bgc_id', embedding_col, 'product_class']
            - emb_col (str): Name of the embedding column in df
            Returns (None, None) if preparation fails
    """
    embedding_configs = {
        "esm_init_last": ("esm_init_last", "embeddings"),
        "esm_init_embedder": ("esm_init_embedder", "embeddings"),
        "random_init_last": ("random_init_last", "embeddings"),
        "random_init_embedder": ("random_init_embedder", "embeddings"),
        "esm_embeddings": ("esm_embeddings", "esm_embeddings"),
        "pfam2vec": ("pfam2vec", "pfam2vec_seq")
    }

    if embedding_type in embedding_configs:
        key, col = embedding_configs[embedding_type]
        df = data[key]
        if df is not None and col in df.columns:
            if col != 'pfam2vec_seq':
                df[col] = df[col].apply(tensor_to_list)
            return df[['bgc_id', col, 'product_class']].copy(), col

    elif embedding_type == "esm_bigcarp_concatenated":
        esm_df = data["esm_embeddings"]
        bigcarp_df = data["random_init_last"]
        if esm_df is not None and bigcarp_df is not None:
            concat_df = create_concatenated_embeddings(esm_df, bigcarp_df)
            if concat_df is not None:
                return concat_df.copy(), 'concatenated_embeddings'

    return None, None

def create_concatenated_embeddings(esm_df, bigcarp_df):
    """
    Concatenate ESM and BigCarp embeddings for each BGC.

    Args:
        esm_df (pandas.DataFrame): DataFrame with ESM embeddings
            Must contain: 'bgc_id', 'esm_embeddings', 'product_class'
        bigcarp_df (pandas.DataFrame): DataFrame with BigCarp embeddings
            Must contain: 'bgc_id', 'embeddings', 'product_class'

    Returns:
        pandas.DataFrame or None: DataFrame with columns:
            ['bgc_id', 'concatenated_embeddings', 'product_class']
            where concatenated_embeddings is list of concatenated vectors.
            Returns None if no common BGC IDs or concatenation fails.
    """
    common_ids = set(esm_df['bgc_id']) & set(bigcarp_df['bgc_id'])
    if len(common_ids) == 0:
        return None

    esm_filtered = esm_df[esm_df['bgc_id'].isin(common_ids)].set_index('bgc_id')
    bigcarp_filtered = bigcarp_df[bigcarp_df['bgc_id'].isin(common_ids)].set_index('bgc_id')

    concatenated_data = []
    for bgc_id in common_ids:
        try:
            esm_seq = tensor_to_list(esm_filtered.loc[bgc_id, 'esm_embeddings'])
            bigcarp_seq = tensor_to_list(bigcarp_filtered.loc[bgc_id, 'embeddings'])

            if not (isinstance(esm_seq, list) and len(esm_seq) > 0 and isinstance(esm_seq[0], list)):
                continue
            if not (isinstance(bigcarp_seq, list) and len(bigcarp_seq) > 0 and isinstance(bigcarp_seq[0], list)):
                continue

            min_len = min(len(esm_seq), len(bigcarp_seq))
            concat_seq = [esm_seq[i] + bigcarp_seq[i] for i in range(min_len)]

            concatenated_data.append({
                'bgc_id': bgc_id,
                'concatenated_embeddings': concat_seq,
                'product_class': esm_filtered.loc[bgc_id, 'product_class']
            })
        except:
            continue

    if not concatenated_data:
        return None

    return pd.DataFrame(concatenated_data)

def _validate_and_prepare_XY(X_raw, y_raw, emb_dim):
    """
    Validate and filter embeddings to ensure correct dimensionality.

    Args:
        X_raw (list): List of embedding sequences (2D arrays or lists)
        y_raw (list): Corresponding labels (same length as X_raw)
        emb_dim (int): Expected embedding dimension

    Returns:
        tuple: (X, y) where:
            - X (list): Filtered valid embeddings
            - y (list): Corresponding filtered labels
            Invalid samples (wrong dimensions or None) are excluded
    """
    X, y = [], []
    for xi, yi in zip(X_raw, y_raw):
        if xi is None:
            continue
        arr = np.asarray(xi)
        if arr.ndim == 2 and arr.shape[1] == emb_dim:
            X.append(xi)
            y.append(yi)
        elif arr.ndim == 1 and arr.shape[0] == emb_dim:
            X.append([xi])
            y.append(yi)
    return X, y

def create_stratified_splits(df, class_cols, n_splits=5, random_state=42):
    """
    Create stratified cross-validation splits for multi-label data.

    Uses iterative stratification algorithm to maintain label distribution across folds.
    Falls back to standard KFold if stratification fails.

    Args:
        df (pandas.DataFrame): DataFrame with binary class columns
        class_cols (list): List of class column names to stratify on
        n_splits (int): Number of CV folds (default: 5)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        list of tuples: List of (train_indices, test_indices) for each fold
    """
    y_binary = df[class_cols].values
    indices = np.arange(len(df))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    try:
        stratifier = IterativeStratification(
            n_splits=n_splits,
            order=2,
            sample_distribution_per_fold=[1.0/n_splits]*n_splits,
            random_state=random_state
        )
        return [(indices[train], indices[test]) for train, test in stratifier.split(indices, y_binary[indices])]
    except:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(np.arange(len(df))))

def compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        y_true (numpy.ndarray): True binary labels, shape (n_samples, n_classes)
        y_pred (numpy.ndarray): Predicted binary labels, shape (n_samples, n_classes)
        y_proba (numpy.ndarray): Predicted probabilities, shape (n_samples, n_classes)
        class_names (array-like): List of class names corresponding to columns

    Returns:
        dict: Dictionary containing:
            - 'exact_match_accuracy' (float): Fraction of samples with all labels correct
            - 'micro_f1' (float): Micro-averaged F1 score
            - 'macro_f1' (float): Macro-averaged F1 score
            - 'weighted_macro_f1' (float): Weighted macro F1 score
            - 'micro_auc' (float): Micro-averaged AUC-ROC
            - 'macro_auc' (float): Macro-averaged AUC-ROC
            - 'weighted_auc' (float): Weighted AUC-ROC
            - 'per_class_auc' (dict): AUC-ROC for each class
            - 'per_class_support' (dict): Number of positive samples per class
    """
    metrics = {
        'exact_match_accuracy': np.mean(np.all(y_true == y_pred, axis=1)),
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_macro_f1': f1_score(y_true, y_pred, average='weighted')
    }

    try:
        metrics['micro_auc'] = roc_auc_score(y_true.ravel(), y_proba.ravel())
    except:
        metrics['micro_auc'] = float('nan')

    try:
        metrics['macro_auc'] = roc_auc_score(y_true, y_proba, average='macro')
    except:
        metrics['macro_auc'] = float('nan')

    try:
        metrics['weighted_auc'] = roc_auc_score(y_true, y_proba, average='weighted')
    except:
        metrics['weighted_auc'] = float('nan')

    per_class_auc = {}
    per_class_support = {}
    for i, class_name in enumerate(class_names):
        if i < y_true.shape[1]:
            class_true = y_true[:, i]
            per_class_support[class_name] = int(np.sum(class_true))
            if len(np.unique(class_true)) > 1:
                try:
                    per_class_auc[class_name] = roc_auc_score(class_true, y_proba[:, i])
                except:
                    per_class_auc[class_name] = float('nan')
            else:
                per_class_auc[class_name] = float('nan')

    metrics['per_class_auc'] = per_class_auc
    metrics['per_class_support'] = per_class_support
    return metrics

def evaluate_mlp_model(df, cv_splits, emb_col, emb_dim, model_name, class_cols, seed=42):
    """
    Evaluate embeddings using mean-pooling + MLP classifier with stratified CV.

    Pipeline:
        1. Mean-pool variable-length embeddings to fixed vectors
        2. Standardize features
        3. Optional PCA if dimension > 256
        4. Train shallow MLP (256->128 units) with One-vs-Rest strategy
        5. Predict with 0.5 threshold

    Args:
        df (pandas.DataFrame): DataFrame with embeddings and class columns
        cv_splits (list): List of (train_idx, test_idx) tuples from create_stratified_splits()
        emb_col (str): Name of embedding column in df
        emb_dim (int): Expected dimension of embeddings
        model_name (str): Display name for this model
        class_cols (list): List of class column names
        seed (int): Random seed for MLP and PCA (default: 42)

    Returns:
        dict or None: Results dictionary containing:
            - 'model_name' (str): Model display name
            - 'embedding_column' (str): Name of embedding column used
            - 'fold_results' (list): Per-fold metrics dictionaries
            - 'aggregate_metrics' (dict): Metrics computed on all folds combined
            - 'class_names' (list): List of class names
            Returns None if all folds fail
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    label_strings = [";".join([c for c in class_cols if row[c]==1]) for _, row in df.iterrows()]
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train_raw = [df.iloc[i][emb_col] for i in train_idx]
        X_test_raw  = [df.iloc[i][emb_col] for i in test_idx]
        y_train_raw = [label_strings[i] for i in train_idx]
        y_test_raw  = [label_strings[i] for i in test_idx]

        X_train, y_train = _validate_and_prepare_XY(X_train_raw, y_train_raw, emb_dim)
        X_test, y_test = _validate_and_prepare_XY(X_test_raw, y_test_raw, emb_dim)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        Xtr = create_mean_pooled_features(X_train)
        Xte = create_mean_pooled_features(X_test)

        mlb = MultiLabelBinarizer()
        Ytr = mlb.fit_transform([s.split(';') if s else [] for s in y_train])
        Yte = mlb.transform([s.split(';') if s else [] for s in y_test])

        try:
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            if Xtr_s.shape[1] > 256:
                pca = PCA(n_components=256, random_state=seed)
                Xtr_s = pca.fit_transform(Xtr_s)
                Xte_s = pca.transform(Xte_s)

            base = MLPClassifier(
                hidden_layer_sizes=(256, 128), activation='relu',
                alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
                early_stopping=True, n_iter_no_change=10, random_state=seed,
                verbose=False
            )
            clf = OneVsRestClassifier(base, n_jobs=-1)
            clf.fit(Xtr_s, Ytr)
            y_proba = clf.predict_proba(Xte_s)
            y_pred = (y_proba > 0.5).astype(int)

            fold_metrics = compute_comprehensive_metrics(Yte, y_pred, y_proba, mlb.classes_)
            fold_metrics['fold'] = fold_idx
            fold_metrics['y_true'] = Yte
            fold_metrics['y_pred'] = y_pred
            fold_metrics['y_proba'] = y_proba
            fold_results.append(fold_metrics)

            all_y_true.append(Yte)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)

            print(f"   Fold {fold_idx+1}: Macro F1={fold_metrics['macro_f1']:.4f}, Macro AUC={fold_metrics['macro_auc']:.4f}")
        except Exception as e:
            print(f"   Fold {fold_idx} failed: {e}")
            continue

    if not all_y_true:
        return None

    aggregate_y_true = np.vstack(all_y_true)
    aggregate_y_pred = np.vstack(all_y_pred)
    aggregate_y_proba = np.vstack(all_y_proba)

    aggregate_metrics = compute_comprehensive_metrics(aggregate_y_true, aggregate_y_pred, aggregate_y_proba, mlb.classes_)

    print(f"\n{model_name} Results:")
    print(f"   Exact Match: {aggregate_metrics['exact_match_accuracy']:.4f}")
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
    """
    Convert variable-length embedding sequences to fixed-length by mean pooling.

    Args:
        embed_sequences (list): List of embedding sequences, where each sequence is:
            - 2D array/list of shape (seq_len, emb_dim), or
            - 1D array/list of shape (emb_dim,)

    Returns:
        numpy.ndarray: Mean-pooled features of shape (n_samples, emb_dim)
            Returns empty array if no valid sequences
    """
    pooled_features = []
    for seq in embed_sequences:
        if seq is None or len(seq) == 0:
            continue
        arr = np.array(seq, dtype=float)
        pooled_features.append(np.mean(arr, axis=0) if arr.ndim == 2 else arr)

    return np.stack(pooled_features) if pooled_features else np.empty((0, 1))

def evaluate_pfam2vec_rf(df, cv_splits, class_cols, seed=42):
    """
    Evaluate Pfam2vec embeddings using Random Forest with stratified CV.

    Trains separate Random Forest classifiers for each class (One-vs-Rest approach).

    Args:
        df (pandas.DataFrame): DataFrame with 'pfam2vec_seq' column and class columns
        cv_splits (list): List of (train_idx, test_idx) tuples
        class_cols (list): List of class column names
        seed (int): Random seed for Random Forest (default: 42)

    Returns:
        dict or None: Results dictionary (same structure as evaluate_mlp_model).
            Returns None if pfam2vec_seq column missing or all folds fail
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: Pfam2vec + Random Forest")
    print(f"{'='*70}")

    # Check for pfam2vec_seq column
    if 'pfam2vec_seq' not in df.columns:
        print("[ERROR] pfam2vec_seq column not found!")
        return None

    # Create mean-pooled features from pfam2vec_seq
    print(f"   Creating mean-pooled features from pfam2vec_seq...")
    X_p2v = create_mean_pooled_features(df['pfam2vec_seq'].values)
    print(f"   P2V feature matrix shape: {X_p2v.shape}")

    if X_p2v.shape[0] == 0:
        print("[ERROR] No valid P2V features created!")
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
        print(f"\nFold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")
        
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
            # Store raw predictions for per-class analysis
            fold_metrics['y_true'] = y_test
            fold_metrics['y_pred'] = class_predictions
            fold_metrics['y_proba'] = class_probabilities
            fold_results.append(fold_metrics)
            
            # Collect for aggregate
            all_y_true.append(y_test)
            all_y_pred.append(class_predictions)
            all_y_proba.append(class_probabilities)

            print(f"   Exact Match: {fold_metrics['exact_match_accuracy']:.4f}, "
                  f"Macro F1: {fold_metrics['macro_f1']:.4f}, "
                  f"Macro AUC: {fold_metrics['macro_auc']:.4f}")

        except Exception as e:
            print(f"   [ERROR] Error in fold {fold_idx}: {e}")
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

    print(f"\nPfam2vec + Random Forest - Final Results:")
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


def evaluate_random_baseline(df, cv_splits, class_cols, seed=42):
    """
    Evaluate simple random baseline with Gaussian embeddings.

    Generates random 256-dimensional embeddings (Gaussian noise) with random sequence lengths.

    Args:
        df (pandas.DataFrame): DataFrame with class columns (embeddings generated randomly)
        cv_splits (list): List of (train_idx, test_idx) tuples
        class_cols (list): List of class column names
        seed (int): Random seed (default: 42)

    Returns:
        dict: Results from evaluate_mlp_model() on random embeddings
    """
    np.random.seed(seed)
    random_embeddings = [np.random.randn(np.random.randint(5, 51), 256).tolist() for _ in range(len(df))]
    df_random = df.copy()
    df_random['random_embeddings'] = random_embeddings
    return evaluate_mlp_model(df_random, cv_splits, 'random_embeddings', 256, "Random 256D Baseline", class_cols, seed)

def evaluate_improved_random_baseline(df, cv_splits, class_cols, seed=42):
    """
    Evaluate domain-consistent random baseline.

    Each unique Pfam domain gets a fixed random 256-dimensional vector. BGC embeddings
    are constructed by stacking domain vectors according to the domain sequence.
    This controls for sequence length and domain composition effects.

    Args:
        df (pandas.DataFrame): DataFrame with 'bgc_id' and class columns
        cv_splits (list): List of (train_idx, test_idx) tuples
        class_cols (list): List of class column names
        seed (int): Random seed (default: 42)

    Returns:
        dict: Results from evaluate_mlp_model() on domain-consistent random embeddings.
            Falls back to simple random baseline if domain sequences unavailable.
    """
    try:
        mibig_pkl = 'data/processed/bgc_product_classification/processed_mibig3/mibig3_preprocessed.pkl'
        df_mibig = pd.read_pickle(mibig_pkl)
        domain_map = dict(zip(df_mibig['bgc_id'], df_mibig['domain_sequence']))
    except:
        return evaluate_random_baseline(df, cv_splits, class_cols, seed)

    rng = np.random.default_rng(seed)
    dim = 256
    unique_domains = set(d for seq in domain_map.values() for d in (seq or []))
    scale = 1.0 / np.sqrt(dim)
    domain_vec = {d: (rng.standard_normal(dim).astype(np.float32) * scale) for d in unique_domains}
    domain_vec['UNK'] = rng.standard_normal(dim).astype(np.float32) * scale

    rand_seqs = []
    for _, row in df.iterrows():
        seq = domain_map.get(row['bgc_id'], []) or []
        mats = [domain_vec.get(tok, domain_vec['UNK']) for tok in seq]
        rand_seqs.append(np.stack(mats, axis=0) if mats else np.zeros((0, dim), dtype=np.float32))

    df_rand = df.copy()
    df_rand['improved_random_embeddings'] = rand_seqs
    return evaluate_mlp_model(df_rand, cv_splits, 'improved_random_embeddings', 256,
                            "Improved Random Baseline", class_cols, seed)

def save_per_class_results(all_results, outdir):
    """
    Save per-class AUC-ROC results to CSV.

    Args:
        all_results (list): List of result dictionaries from evaluation functions
        outdir (str): Output directory path

    Output Files:
        - per_class_detailed.csv: Contains Model, Class, AUC_ROC, Support, Frequency columns
    """
    per_class_data = []
    for result in all_results:
        agg_metrics = result.get('aggregate_metrics', {})
        per_class_auc = agg_metrics.get('per_class_auc', {})
        per_class_support = agg_metrics.get('per_class_support', {})
        for class_name, auc_score in per_class_auc.items():
            per_class_data.append({
                'Model': result['model_name'],
                'Class': class_name,
                'AUC_ROC': auc_score,
                'Support': per_class_support.get(class_name, 0)
            })

    if per_class_data:
        per_class_df = pd.DataFrame(per_class_data)
        total_samples = per_class_df.groupby('Model')['Support'].sum().iloc[0] if not per_class_df.empty else 1
        per_class_df['Frequency'] = per_class_df['Support'] / total_samples
        per_class_df.to_csv(f"{outdir}/per_class_detailed.csv", index=False)

def create_comparison_table(all_results, outdir):
    """
    Create and save model comparison table.

    Args:
        all_results (list): List of result dictionaries from evaluation functions
        outdir (str): Output directory path

    Returns:
        pandas.DataFrame or None: Comparison DataFrame with columns:
            ['Model', 'macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy']
            Returns None if no valid results

    Output Files:
        - mibig3_comparison.csv: Model comparison table
    """
    if not all_results:
        return None

    main_metrics = ['macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy']
    comparison_data = [{'Model': r['model_name'], **{m: r['aggregate_metrics'][m] for m in main_metrics}}
                      for r in all_results if r]

    if not comparison_data:
        return None

    df_comparison = pd.DataFrame(comparison_data)
    display_df = df_comparison.copy()
    display_df.columns = ['Model', 'Macro F1', 'Macro AUC', 'Weighted AUC', 'Exact Accuracy']

    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(display_df.to_string(index=False, float_format='%.4f'))

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(f"{outdir}/mibig3_comparison.csv", index=False)
    return df_comparison

def main():
    """
    Main pipeline for MIBiG 3.0 multi-label classification evaluation.

    Command-line Arguments:
        --artifacts_dir: Directory containing embedding pickle files
                        (default: "artifacts/classification/mibig3")
        --outdir: Output directory for results
                 (default: "results/mibig3_classification")
        --seed: Random seed for reproducibility (default: 42)

    Pipeline:
        1. Load all embedding data
        2. Evaluate MLP models on BigCarp/ESM embeddings
        3. Evaluate Random Forest on Pfam2vec
        4. Evaluate random baselines
        5. Save comparison tables and detailed results
    """
    parser = argparse.ArgumentParser(description="MIBiG 3.0 Multi-Label Classification")
    parser.add_argument("--artifacts_dir", default="artifacts/classification/mibig3",
                        help="Directory containing embedding pickle files")
    parser.add_argument("--outdir", default="results/mibig3_classification",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    data = load_mibig3_data(args.artifacts_dir)
    available_data = [k for k, v in data.items() if v is not None and not (isinstance(v, pd.DataFrame) and v.empty)]
    if not available_data:
        return

    models_to_evaluate = [
        ("esm_init_last", "ESM Init Last + MLP"),
        ("esm_init_embedder", "ESM Init Embedder + MLP"),
        ("random_init_last", "Random Init Last + MLP"),
        ("random_init_embedder", "Random Init Embedder + MLP"),
        ("esm_embeddings", "ESM Embeddings + MLP"),
        ("esm_bigcarp_concatenated", "ESM + BigCarp Concatenated + MLP"),
    ]

    all_results = []

    for embedding_type, model_name in models_to_evaluate:
        if embedding_type == "esm_bigcarp_concatenated":
            if data["esm_embeddings"] is None or data["random_init_last"] is None:
                continue
        elif data[embedding_type] is None:
            continue

        df_prep, emb_col = prepare_embedding_data(data, embedding_type)
        if df_prep is None:
            continue

        df_prep, class_cols = convert_product_classes_to_binary(df_prep)
        cv_splits = create_stratified_splits(df_prep, class_cols, n_splits=5, random_state=args.seed)

        sample_emb = df_prep[emb_col].dropna().iloc[0]
        if isinstance(sample_emb, list) and len(sample_emb) > 0:
            emb_dim = len(sample_emb[0]) if isinstance(sample_emb[0], list) else len(sample_emb)
        else:
            continue

        result = evaluate_mlp_model(df_prep, cv_splits, emb_col, emb_dim, model_name, class_cols, args.seed)
        if result:
            all_results.append(result)

    if data["pfam2vec"] is not None:
        df_prep, emb_col = prepare_embedding_data(data, "pfam2vec")
        if df_prep is not None:
            df_prep, class_cols = convert_product_classes_to_binary(df_prep)
            cv_splits = create_stratified_splits(df_prep, class_cols, n_splits=5, random_state=args.seed)

            result = evaluate_pfam2vec_rf(df_prep, cv_splits, class_cols, args.seed)
            if result:
                all_results.append(result)

    baseline_data = next((data[k] for k in ["esm_init_last", "esm_embeddings", "pfam2vec"] if data[k] is not None), None)
    if baseline_data is not None:
        df_baseline = baseline_data[['bgc_id', 'product_class']].copy()
        df_baseline, class_cols = convert_product_classes_to_binary(df_baseline)
        cv_splits = create_stratified_splits(df_baseline, class_cols, n_splits=5, random_state=args.seed)

        result = evaluate_random_baseline(df_baseline, cv_splits, class_cols, args.seed)
        if result:
            all_results.append(result)

        result = evaluate_improved_random_baseline(df_baseline, cv_splits, class_cols, args.seed)
        if result:
            all_results.append(result)

    if all_results:
        create_comparison_table(all_results, args.outdir)
        pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
        with open(f"{args.outdir}/complete_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        save_per_class_results(all_results, args.outdir)
        print(f"\nResults saved to: {args.outdir}/")

if __name__ == "__main__":
    main()
