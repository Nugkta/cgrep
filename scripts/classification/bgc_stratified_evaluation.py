#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BGC Multi-Label Classification with Stratified Cross-Validation
==============================================================
• Uses your actual data files from artifacts/classification/multiclass/
• Stratified 5-fold CV for fair comparison
• Multiple embedding approaches with comprehensive metrics
• Highlighted comparison table
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
sys.path.append('/mnt/iusers01/mace01/j56806hx/scratch/Embedded_Subclusters')
from emb_sub.models_multiclass import MultiLabelBiLSTMClassifier

# Try stratified multilabel CV
try:
    from skmultilearn.model_selection import IterativeStratification
    STRATIFIED_AVAILABLE = True
    print("✅ scikit-multilearn available for stratified CV")
except ImportError:
    STRATIFIED_AVAILABLE = False
    print("⚠️  scikit-multilearn not available. Install with: pip install scikit-multilearn")

# -------- Data Loading Functions -------------------------------------------
def load_embedding_data(artifacts_dir="artifacts/classification/multiclass"):
    """Load and combine embedding data from your actual files."""
    print("📂 Loading embedding data from artifacts...")
    
    # File paths for different BigCarp variants
    bc_random_path = f"{artifacts_dir}/emb_ckpts_random/mibig_embeddings_last_ckpt50.pkl"
    bc_pretrained_path = f"{artifacts_dir}/ckpts_pt/mibig_embeddings_last_ckpt50.pkl"
    bc_frozen_path = f"{artifacts_dir}/emb_ckpts_fr/mibig_embeddings_last_ckpt50.pkl"
    bc_embedder_path = f"{artifacts_dir}/ckpts_pt_embedder/mibig_embeddings_embedder_ckpt50.pkl"
    esm_path = f"{artifacts_dir}/mibig_esm_embeddings.pkl"
    p2v_path = f"{artifacts_dir}/mibig_embeddings_p2v.csv"  # P2V embeddings for Random Forest
    preprocessed_path = f"{artifacts_dir}/mibig_preprocessed.pkl"
    
    print(f"   📄 BigCarp Random: {bc_random_path}")
    print(f"   📄 BigCarp Pretrained: {bc_pretrained_path}")
    print(f"   📄 BigCarp Frozen: {bc_frozen_path}")
    print(f"   📄 BigCarp Embedder: {bc_embedder_path}")
    print(f"   📄 ESM embeddings: {esm_path}")  
    print(f"   📄 P2V embeddings: {p2v_path}")
    print(f"   📄 Preprocessed data: {preprocessed_path}")
    
    # Load BigCarp variants
    bc_data = {}
    bc_files = [
        (bc_random_path, "BigCarp Random", "random"),
        (bc_pretrained_path, "BigCarp Pretrained", "pretrained"),
        (bc_frozen_path, "BigCarp Frozen", "frozen"),
        (bc_embedder_path, "BigCarp Embedder", "embedder")
    ]
    
    for file_path, description, variant in bc_files:
        try:
            df = pd.read_pickle(file_path)
            bc_data[variant] = df
            print(f"   ✅ Loaded {description}: {df.shape}")
            print(f"      Columns: {list(df.columns)}")
        except Exception as e:
            print(f"   ❌ Error loading {description}: {e}")
            bc_data[variant] = None
    
    # Load ESM data
    try:
        df_esm = pd.read_pickle(esm_path)
        print(f"   ✅ Loaded ESM data: {df_esm.shape}")
        print(f"      Columns: {list(df_esm.columns)}")
    except Exception as e:
        print(f"   ❌ Error loading ESM data: {e}")
        df_esm = None
        
    # Load P2V embeddings data
    try:
        df_p2v = pd.read_csv(p2v_path)
        print(f"   ✅ Loaded P2V data: {df_p2v.shape}")
        print(f"      Columns: {list(df_p2v.columns)}")
        
        # Convert pfam2vec_seq from string representation to actual lists
        if 'pfam2vec_seq' in df_p2v.columns:
            print(f"      Converting pfam2vec_seq from string to list format...")
            # Use eval to convert string representation of arrays to actual arrays
            def safe_eval(x):
                try:
                    if pd.isna(x) or x == '':
                        return []
                    # Convert string representation to actual arrays
                    # Need to provide numpy array function in the namespace
                    import numpy as np
                    # Create a comprehensive namespace with numpy functions and types
                    namespace = {"__builtins__": {}}
                    namespace.update(vars(np))  # Add all numpy functions and types
                    result = eval(x, namespace)
                    # If result is a list of numpy arrays, convert to list of lists
                    if isinstance(result, list) and len(result) > 0:
                        if hasattr(result[0], 'tolist'):  # numpy array
                            return [arr.tolist() for arr in result]
                        else:
                            return result
                    return []
                except Exception as e:
                    print(f"      ⚠️  Error converting sequence: {e}")
                    return []
            df_p2v['pfam2vec_seq'] = df_p2v['pfam2vec_seq'].apply(safe_eval)
            print(f"      ✅ pfam2vec_seq converted to list format")
    except Exception as e:
        print(f"   ❌ Error loading P2V data: {e}")
        df_p2v = None
        
    # Load preprocessed data
    try:
        df_preprocessed = pd.read_pickle(preprocessed_path)
        print(f"   ✅ Loaded preprocessed data: {df_preprocessed.shape}")
        print(f"      Columns: {list(df_preprocessed.columns)}")
        
        # Check for potential class columns in preprocessed data
        print(f"      Sample data types:")
        for col in df_preprocessed.columns:
            dtype = df_preprocessed[col].dtype
            try:
                unique_count = df_preprocessed[col].nunique()
                print(f"        {col}: {dtype}, {unique_count} unique values")
                if pd.api.types.is_numeric_dtype(df_preprocessed[col]) and unique_count <= 10:
                    unique_vals = sorted(df_preprocessed[col].dropna().unique())
                    print(f"          Unique values: {unique_vals}")
            except TypeError:
                # Handle columns with unhashable types (like lists)
                print(f"        {col}: {dtype}, contains lists/complex objects")
                if col == 'product_class':
                    # Show sample product_class values
                    sample_vals = df_preprocessed[col].dropna().head(3).tolist()
                    print(f"          Sample values: {sample_vals}")
                
    except Exception as e:
        print(f"   ❌ Error loading preprocessed data: {e}")
        df_preprocessed = None
    
    return bc_data, df_esm, df_p2v, df_preprocessed

def tensor_to_list(x):
    """Convert tensor to list if needed."""
    if hasattr(x, 'detach'):  # torch tensor
        return x.detach().cpu().numpy().tolist()
    elif hasattr(x, 'tolist'):  # numpy array
        return x.tolist()
    return x

def concatenate_embeddings(esm_embedding, bc_embedding, max_bad_report=5):
    """
    Concatenate ESM and BigCarp embeddings along the feature dimension.
    Both should be sequences of shape [seq_len, emb_dim].
    Returns concatenated embedding of shape [seq_len, esm_dim + bc_dim] or None if invalid.
    """
    # Convert to lists if needed
    if hasattr(esm_embedding, 'detach'):
        esm_embedding = esm_embedding.detach().cpu().numpy()
    if hasattr(bc_embedding, 'detach'):
        bc_embedding = bc_embedding.detach().cpu().numpy()
    
    # Convert to numpy arrays
    try:
        esm_arr = np.array(esm_embedding)
        bc_arr = np.array(bc_embedding)
    except:
        return None
    
    # Check if both are valid 2D sequences
    if esm_arr.ndim != 2 or bc_arr.ndim != 2:
        return None
    
    # Check if sequence lengths match (or can be made to match)
    esm_seq_len, esm_dim = esm_arr.shape
    bc_seq_len, bc_dim = bc_arr.shape
    
    # Handle different sequence lengths by taking the minimum
    min_seq_len = min(esm_seq_len, bc_seq_len)
    if esm_seq_len != bc_seq_len:
        # Truncate to same length
        esm_arr = esm_arr[:min_seq_len, :]
        bc_arr = bc_arr[:min_seq_len, :]
    
    # Concatenate along feature dimension
    try:
        concatenated = np.concatenate([esm_arr, bc_arr], axis=1)
        return concatenated.tolist()
    except:
        return None

# ---- Validation & Coercion Helpers ---------------------------------------
def _coerce_to_seq2d(sample, emb_dim):
    """Coerce an embedding sample into a 2D sequence [T, D].
    Returns a Python list-of-lists or None if impossible.
    Handles: numpy arrays, torch tensors, list-of-lists, and 1D vectors of length D.
    """
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
        # sometimes shape could be (T,) where each element is a vector-like
        if arr.ndim == 1 and len(arr) > 0 and hasattr(arr[0], '__len__'):
            try:
                arr2 = np.vstack(arr)
                if arr2.ndim == 2 and arr2.shape[1] == emb_dim:
                    return arr2.tolist()
            except Exception:
                pass
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
    """Validate & coerce X to [T,D] lists; drop invalid pairs. Returns (X, y, n_kept, n_dropped).
    Also prints a few examples of problems for quick debugging.
    """
    X, y = [], []
    bad = 0
    for i, (xi, yi) in enumerate(zip(X_raw, y_raw)):
        coerced = _coerce_to_seq2d(xi, emb_dim)
        if coerced is None:
            if bad < max_bad_report:
                print(f"   ⚠️  Dropping {split_name} sample #{i}: invalid shape/type for embedding")
                if hasattr(xi, 'shape'):
                    print(f"      • shape={getattr(xi, 'shape', None)}")
                else:
                    t = type(xi)
                    lens = (len(xi) if isinstance(xi, list) else 'n/a')
                    print(f"      • type={t}, len={lens}")
            bad += 1
            continue
        X.append(coerced)
        y.append(yi)
    if bad > 0:
        print(f"   🧹 {split_name.capitalize()} cleanup: kept {len(X)}/{len(X_raw)} (dropped {bad})")
    # guard: avoid empty label strings -> keep as empty set "" (model handles via MultiLabelBinarizer)
    return X, y, len(X), bad

def convert_multilabel_to_binary(df):
    """Convert semicolon-separated product_class to binary columns."""
    print("\n🔄 Converting multi-label classes to binary columns...")
    
    # Get all unique classes
    all_classes = set()
    for class_str in df['product_class'].dropna():
        if pd.isna(class_str):
            continue
        classes = str(class_str).split(';')
        all_classes.update([cls.strip() for cls in classes])
    
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

def create_concatenated_embeddings(bc_data, df_esm, df_combined):
    """Create concatenated ESM+BigCarp embeddings for all BigCarp variants."""
    print("\n🔗 Creating concatenated ESM+BigCarp embeddings...")
    
    if df_esm is None:
        print("   ⚠️  ESM data not available for concatenation")
        return df_combined
    
    # Find ESM embedding column (sequence embeddings, not pooled)
    esm_embedding_col = None
    for col in df_esm.columns:
        if 'embedding' in col.lower() and 'mean' not in col.lower() and 'max' not in col.lower():
            esm_embedding_col = col
            break
    
    if esm_embedding_col is None:
        print("   ⚠️  Could not find ESM sequence embedding column")
        return df_combined
    
    print(f"   Using ESM embedding column: {esm_embedding_col}")
    
    # Prepare ESM data for merging
    df_esm_temp = df_esm.copy()
    df_esm_temp[esm_embedding_col] = df_esm_temp[esm_embedding_col].apply(tensor_to_list)
    
    # Merge ESM data first if not already merged
    if esm_embedding_col not in df_combined.columns:
        if 'bgc_id' in df_esm_temp.columns and 'bgc_id' in df_combined.columns:
            df_combined = df_combined.merge(
                df_esm_temp[['bgc_id', esm_embedding_col]], 
                on='bgc_id', how='inner'
            )
            print(f"   Merged ESM data: {df_combined.shape}")
        elif 'sequence_id' in df_esm_temp.columns and 'sequence_id' in df_combined.columns:
            df_combined = df_combined.merge(
                df_esm_temp[['sequence_id', esm_embedding_col]], 
                on='sequence_id', how='inner'
            )
            print(f"   Merged ESM data: {df_combined.shape}")
        else:
            print("   ⚠️  Cannot merge ESM data - no common ID column")
            return df_combined
    
    # Create concatenated embeddings only for BigCarp random variant
    if 'random' in bc_data and bc_data['random'] is not None:
        variant = 'random'
        df_bc = bc_data[variant]
        
        print(f"   Creating ESM + BigCarp {variant} concatenated embeddings...")
        
        # Find BigCarp embedding column (sequence embeddings, not pooled)
        bc_embedding_col = None
        for col in df_bc.columns:
            if 'embedding' in col.lower() and 'mean' not in col.lower() and 'max' not in col.lower():
                bc_embedding_col = col
                break
        
        if bc_embedding_col is None:
            print(f"     ⚠️  Could not find BigCarp {variant} sequence embedding column")
        else:
            print(f"     Using BigCarp embedding column: {bc_embedding_col}")
            
            # Prepare BigCarp data
            df_bc_temp = df_bc.copy()
            df_bc_temp[bc_embedding_col] = df_bc_temp[bc_embedding_col].apply(tensor_to_list)
            
            # Merge BigCarp data temporarily
            df_temp = df_combined.copy()
            if 'bgc_id' in df_bc_temp.columns and 'bgc_id' in df_temp.columns:
                df_temp = df_temp.merge(
                    df_bc_temp[['bgc_id', bc_embedding_col]], 
                    on='bgc_id', how='inner'
                )
                
                # Create concatenated embeddings
                concatenated_embeddings = []
                valid_indices = []
                
                for idx, row in df_temp.iterrows():
                    if esm_embedding_col in row and bc_embedding_col in row:
                        esm_emb = row[esm_embedding_col]
                        bc_emb = row[bc_embedding_col]
                        
                        concat_emb = concatenate_embeddings(esm_emb, bc_emb)
                        if concat_emb is not None:
                            concatenated_embeddings.append(concat_emb)
                            valid_indices.append(idx)
                
                if concatenated_embeddings:
                    # Add concatenated embedding column to main dataframe
                    concat_col_name = f'esm_bigcarp_{variant}_concatenated_embedding'
                    
                    # Initialize column with None
                    df_combined[concat_col_name] = None
                    
                    # Fill valid concatenated embeddings
                    for i, idx in enumerate(valid_indices):
                        if idx < len(df_combined):
                            df_combined.at[idx, concat_col_name] = concatenated_embeddings[i]
                    
                    # Check dimensions
                    if concatenated_embeddings:
                        sample_emb = concatenated_embeddings[0]
                        seq_len = len(sample_emb)
                        emb_dim = len(sample_emb[0]) if sample_emb else 0
                        print(f"     ✅ Created {len(concatenated_embeddings)} concatenated embeddings: {seq_len} x {emb_dim}")
                    else:
                        print(f"     ⚠️  No valid concatenated embeddings created for {variant}")
                else:
                    print(f"     ⚠️  No valid concatenated embeddings created for {variant}")
            else:
                print(f"     ⚠️  Cannot merge BigCarp {variant} - no common bgc_id column")
    else:
        print(f"   ⚠️  BigCarp random variant not available for concatenation")
    
    return df_combined

def prepare_combined_dataframe(bc_data, df_esm, df_p2v, df_preprocessed):
    """Combine embedding data into a single dataframe for evaluation."""
    print("\n🔧 Preparing combined dataframe...")
    
    # Start with preprocessed data as base (contains labels)
    if df_preprocessed is not None:
        df_combined = df_preprocessed.copy()
        print(f"   Base dataframe: {df_combined.shape}")
        
        # Convert multi-label string format to binary columns
        df_combined, class_names = convert_multilabel_to_binary(df_combined)
        print(f"   After converting to binary: {df_combined.shape}")
        
    else:
        raise ValueError("Preprocessed data is required for class labels")
    
    # Add BigCarp embeddings from all variants
    for variant, df_bc in bc_data.items():
        if df_bc is not None:
            print(f"   Adding {variant} BigCarp embeddings...")
            
            # Try to identify embedding columns in bigcarp data
            embedding_cols = []
            for col in df_bc.columns:
                if 'embedding' in col.lower():
                    embedding_cols.append(col)
            
            print(f"     Found {variant} embedding columns: {embedding_cols}")
            
            # Merge on bgc_id (most likely common key)
            if 'bgc_id' in df_bc.columns and 'bgc_id' in df_combined.columns:
                for col in embedding_cols:
                    df_bc[col] = df_bc[col].apply(tensor_to_list)
                    # Add variant prefix to column name
                    new_col_name = f'bigcarp_{variant}_{col}'
                    df_combined = df_combined.merge(
                        df_bc[['bgc_id', col]].rename(columns={col: new_col_name}),
                        on='bgc_id', how='left'
                    )
            else:
                print(f"     ⚠️  Cannot merge {variant} - no common bgc_id column")
    
    # Add ESM embeddings if available
    if df_esm is not None:
        esm_cols = []
        for col in df_esm.columns:
            if 'embedding' in col.lower() or 'esm' in col.lower():
                esm_cols.append(col)
        
        print(f"   Found ESM embedding columns: {esm_cols}")
        
        # Merge ESM data
        if 'sequence_id' in df_esm.columns and 'sequence_id' in df_combined.columns:
            for col in esm_cols:
                df_esm[col] = df_esm[col].apply(tensor_to_list)
            df_combined = df_combined.merge(df_esm[['sequence_id'] + esm_cols], 
                                          on='sequence_id', how='left')
        else:
            for col in esm_cols:
                if col in df_esm.columns:
                    df_esm[col] = df_esm[col].apply(tensor_to_list)
                    df_combined[f'esm_{col}'] = df_esm[col]
    
    # Add P2V embeddings if available
    if df_p2v is not None:
        print(f"   Adding P2V embeddings...")
        
        # Merge P2V data (assuming it has bgc_id or similar identifier)
        if 'bgc_id' in df_p2v.columns and 'bgc_id' in df_combined.columns:
            # For P2V, we specifically want the pfam2vec_seq column
            if 'pfam2vec_seq' in df_p2v.columns:
                print(f"     Found P2V embedding column: pfam2vec_seq")
                
                # Merge the P2V data
                df_combined = df_combined.merge(
                    df_p2v[['bgc_id', 'pfam2vec_seq']],
                    on='bgc_id', how='left'
                )
            else:
                print(f"     ⚠️  pfam2vec_seq column not found in P2V data")
        else:
            print(f"     ⚠️  Cannot merge P2V - no common bgc_id column")
    
    print(f"   Combined dataframe: {df_combined.shape}")
    
    # Create concatenated ESM+BigCarp embeddings
    df_combined = create_concatenated_embeddings(bc_data, df_esm, df_combined)
    
    print(f"   Final combined dataframe: {df_combined.shape}")
    print(f"   Final columns: {list(df_combined.columns)}")
    
    return df_combined

def identify_class_columns(df):
    """Identify binary class label columns."""
    print(f"\n🔍 Identifying class columns:")
    
    skip_cols = {
        'sequence_id', 'bgc_id', 'description', 'pfam_domains', 'domain_sequence',
        'pfam_sequence', 'sequence_length', 'product_class'
    }
    
    # Add any embedding columns to skip
    embedding_cols = []
    for col in df.columns:
        if 'embedding' in col.lower():
            skip_cols.add(col)
            embedding_cols.append(col)
    
    candidate_cols = [c for c in df.columns if c not in skip_cols]
    
    binary_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1}) and len(unique_vals) > 1:
                binary_cols.append(col)
    
    print(f"   Found {len(binary_cols)} binary class columns: {binary_cols}")
    
    if not binary_cols:
        print(f"   ⚠️  No binary columns found!")
        print(f"   Available candidates were: {candidate_cols}")
        print(f"   This suggests the multi-label conversion may have failed")
    
    return binary_cols

# -------- Stratified CV Functions ------------------------------------------
def create_stratified_splits(df, class_cols, n_splits=5, random_state=42):
    """Create stratified CV splits for multi-label classification."""
    y_binary = df[class_cols].values
    
    print(f"\n📊 Class Distribution Analysis:")
    class_counts = y_binary.sum(axis=0)
    for i, col in enumerate(class_cols):
        print(f"  {col}: {class_counts[i]:4d} samples ({class_counts[i]/len(df)*100:5.1f}%)")
    
    if STRATIFIED_AVAILABLE:
        try:
            # IterativeStratification doesn't have shuffle parameter, but we can shuffle indices manually
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

# -------- Metrics Functions ------------------------------------------------
def exact_match_accuracy(y_true, y_pred):
    """Exact match accuracy - all labels must be predicted correctly."""
    return np.mean(np.all(y_true == y_pred, axis=1))

def compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
    """Compute all requested metrics."""
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
    
    # 7. Weighted Macro AUC
    try:
        metrics['weighted_macro_auc'] = roc_auc_score(y_true, y_proba, average='weighted')
    except ValueError:
        metrics['weighted_macro_auc'] = float('nan')
    
    # 8. Per-class metrics
    per_class = {}
    for i, class_name in enumerate(class_names):
        pc_metrics = {}
        pc_metrics['support'] = int(y_true[:, i].sum())
        
        try:
            pc_metrics['precision'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            pc_metrics['recall'] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            pc_metrics['f1'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        except:
            pc_metrics['precision'] = 0.0
            pc_metrics['recall'] = 0.0
            pc_metrics['f1'] = 0.0
        
        try:
            pc_metrics['auc'] = roc_auc_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            pc_metrics['auc'] = float('nan')
        
        per_class[class_name] = pc_metrics
    
    metrics['per_class'] = per_class
    return metrics

# -------- Model Evaluation -------------------------------------------------
def evaluate_embedding_model(df, cv_splits, emb_col, emb_dim, model_name, class_cols, seed=42):
    """Evaluate single embedding approach."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"   Column: {emb_col} (dim={emb_dim})")
    print(f"{'='*70}")
    
    # Check if embedding column exists
    if emb_col not in df.columns:
        print(f"❌ Column {emb_col} not found in dataframe")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Prepare multi-label strings  
    label_strings = [";".join([c for c in class_cols if row[c]==1])
                     for _, row in df.iterrows()]
    
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []
    
    # Try to use GPU if available
    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n📁 Fold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")

        # Prepare data (raw)
        X_train_raw = [df.iloc[i][emb_col] for i in train_idx]
        X_test_raw  = [df.iloc[i][emb_col] for i in test_idx]
        y_train_raw = [label_strings[i] for i in train_idx]
        y_test_raw  = [label_strings[i] for i in test_idx]

        # Coerce/validate shapes -> drop bad samples to avoid runtime indexing errors
        X_train, y_train, kept_tr, drop_tr = _validate_and_prepare_XY(X_train_raw, y_train_raw, emb_dim, split_name="train")
        X_test,  y_test,  kept_te, drop_te = _validate_and_prepare_XY(X_test_raw,  y_test_raw,  emb_dim, split_name="test")

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"   ❌ After validation, not enough samples to train/test (train={len(X_train)}, test={len(X_test)})")
            continue

        # Train model
        try:
            model = MultiLabelBiLSTMClassifier(
                embed_dim=emb_dim, hidden_dim=512, num_layers=2,
                dropout_rate=0.2, pooling_strategy="mean",
                lr=1e-4, batch_size=32, early_stopping_patience=10, max_epochs=150,
                random_seed=seed, device=device
            )
            print(f"   🏋️  Training BiLSTM model...")
            model.fit(X_train, y_train, n_folds=1, show_progress=True)
            
            # Predict and evaluate
            print(f"   🔮 Making predictions on test set...")
            y_proba = model.predict_proba(X_test)
            y_true = model.mlb.transform([s.split(';') for s in y_test])
            y_pred = (y_proba > 0.5).astype(int)
            
            # Compute fold metrics
            print(f"   📊 Computing metrics...")
            fold_metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, model.mlb.classes_)
            fold_metrics['fold'] = fold_idx
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
    
    print(f"   🧮 Computing final aggregate metrics...")
    aggregate_metrics = compute_comprehensive_metrics(
        aggregate_y_true, aggregate_y_pred, aggregate_y_proba, model.mlb.classes_
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
        'class_names': model.mlb.classes_.tolist()
    }

def create_mean_pooled_features(embed_sequences):
    """
    Convert variable-length embedding sequences to fixed-length features
    by mean pooling across the sequence dimension.
    """
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
                print(f"      ⚠️  Error processing sequence: {e}")
                if fallback_dim is not None:
                    pooled_features.append(np.zeros(fallback_dim))
    
    if not pooled_features:
        print("      ❌ No valid features could be created")
        return np.empty((0, fallback_dim if fallback_dim is not None else 1))
    
    return np.stack(pooled_features)

def evaluate_p2v_random_forest(df, cv_splits, class_cols, seed=42):
    """Evaluate P2V embeddings with Random Forest classifier using mean pooling."""
    print(f"\n{'='*70}")
    print(f"🔬 Evaluating: P2V + Random Forest")
    print(f"{'='*70}")
    
    # Check for pfam2vec_seq column
    if 'pfam2vec_seq' not in df.columns:
        print("❌ pfam2vec_seq column not found!")
        return None
    
    print(f"   Found pfam2vec_seq column")
    
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
            print(f"   📊 Computing metrics...")
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
        print(f"❌ No successful folds for P2V + Random Forest")
        return None
    
    # Aggregate all folds
    print(f"\n   📈 Aggregating results across all folds...")
    aggregate_y_true = np.vstack(all_y_true)
    aggregate_y_pred = np.vstack(all_y_pred)
    aggregate_y_proba = np.vstack(all_y_proba)
    
    print(f"   🧮 Computing final aggregate metrics...")
    aggregate_metrics = compute_comprehensive_metrics(
        aggregate_y_true, aggregate_y_pred, aggregate_y_proba, mlb.classes_
    )
    
    print(f"\n🎯 P2V + Random Forest - Final Results:")
    print(f"   Exact Match Accuracy: {aggregate_metrics['exact_match_accuracy']:.4f}")
    print(f"   Macro F1: {aggregate_metrics['macro_f1']:.4f}")
    print(f"   Macro AUC: {aggregate_metrics['macro_auc']:.4f}")
    
    return {
        'model_name': "P2V + Random Forest",
        'embedding_column': f"P2V (pfam2vec_seq, {X_p2v.shape[1]} features)",
        'fold_results': fold_results,
        'aggregate_metrics': aggregate_metrics,
        'class_names': mlb.classes_.tolist()
    }

# -------- Results Analysis -------------------------------------------------
def create_comparison_table(all_results, outdir):
    """Create highlighted comparison table."""
    if not all_results:
        print("❌ No results to compare")
        return None
    
    main_metrics = ['macro_f1', 'macro_auc', 'exact_match_accuracy']
    
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
    print("🏆 MODEL COMPARISON - KEY METRICS")
    print(f"{'='*80}")
    
    print("\n📊 Performance Summary:")
    display_df = df_comparison.copy()
    display_df.columns = ['Model', 'Macro F1', 'Macro AUC', 'Exact Accuracy']
    print(display_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n🥇 Best Performance:")
    for i, metric in enumerate(['Macro F1', 'Macro AUC', 'Exact Accuracy']):
        col_name = display_df.columns[i+1]
        best_idx = display_df[col_name].idxmax()
        best_model = display_df.loc[best_idx, 'Model']
        best_value = display_df.loc[best_idx, col_name]
        print(f"   {metric}: {best_model} ({best_value:.4f})")
    
    # Save results
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(f"{outdir}/model_comparison.csv", index=False)
    
    return df_comparison

# -------- Main Pipeline ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BGC Multi-Label Classification with Stratified CV")
    parser.add_argument("--artifacts_dir", default="artifacts/classification/multiclass",
                        help="Directory containing embedding files")
    parser.add_argument("--outdir", default="results/bigcarp/classification/overall",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    print("🚀 BGC Multi-Label Classification with Stratified CV")
    print(f"📁 Artifacts: {args.artifacts_dir}")
    print(f"📁 Output: {args.outdir}")
    
    # Load data
    bc_data, df_esm, df_p2v, df_preprocessed = load_embedding_data(args.artifacts_dir)
    
    # Combine into single dataframe
    try:
        df_combined = prepare_combined_dataframe(bc_data, df_esm, df_p2v, df_preprocessed)
    except Exception as e:
        print(f"❌ Error preparing dataframe: {e}")
        return
    
    # Identify class columns
    class_cols = identify_class_columns(df_combined)
    print(f"\n📊 Found {len(class_cols)} class columns: {class_cols}")
    
    if not class_cols:
        print("❌ No binary class columns found!")
        return
    
    # Create stratified CV splits
    cv_splits = create_stratified_splits(df_combined, class_cols, n_splits=5, random_state=args.seed)
    
    # Define models to evaluate based on available columns
    models_to_test = []
    
    # Check what embedding columns we actually have
    for col in df_combined.columns:
        if 'embedding' in col.lower():
            # Get actual embedding data to determine properties
            sample_emb = df_combined[col].dropna().iloc[0]
            
            print(f"   Analyzing {col}:")
            print(f"     Type: {type(sample_emb)}")
            
            if isinstance(sample_emb, list):
                if len(sample_emb) > 0 and isinstance(sample_emb[0], list):
                    # Sequence of embeddings (for BiLSTM)
                    emb_dim = len(sample_emb[0])
                    seq_len = len(sample_emb)
                    print(f"     Sequence embeddings: {seq_len} steps x {emb_dim} dim")
                    is_sequence = True
                else:
                    # Single vector embedding
                    emb_dim = len(sample_emb)
                    print(f"     Single vector: {emb_dim} dim")
                    is_sequence = False
            elif hasattr(sample_emb, 'shape'):
                if len(sample_emb.shape) == 2:
                    # Sequence of embeddings
                    seq_len, emb_dim = sample_emb.shape
                    print(f"     Sequence embeddings: {seq_len} steps x {emb_dim} dim")
                    is_sequence = True
                else:
                    # Single vector
                    emb_dim = sample_emb.shape[0]
                    print(f"     Single vector: {emb_dim} dim")
                    is_sequence = False
            else:
                print(f"     Unknown format, skipping")
                continue
            
            # Only use sequence embeddings for BiLSTM
            if is_sequence:
                if 'pfam' in col.lower():
                    models_to_test.append((col, emb_dim, "Pfam + BiLSTM"))
                elif col.lower() == 'esm_embeddings' or 'esm_esm_embeddings' in col.lower():
                    # Only the raw ESM embeddings (per-token), not max/mean pooled
                    models_to_test.append((col, emb_dim, "ESM + BiLSTM"))
                elif 'concatenated_embedding' in col.lower():
                    # Handle concatenated embeddings - only for random variant
                    if 'esm_bigcarp_random_' in col.lower():
                        models_to_test.append((col, emb_dim, "ESM + BigCarp Random (Concatenated)"))
                    # Skip other concatenated variants
                elif 'bigcarp' in col.lower() and 'mean' not in col.lower() and 'max' not in col.lower():
                    # Identify BigCarp variants based on new naming convention
                    if 'bigcarp_random_' in col.lower():
                        models_to_test.append((col, emb_dim, "BigCarp Random + BiLSTM"))
                    elif 'bigcarp_pretrained_' in col.lower():
                        models_to_test.append((col, emb_dim, "BigCarp Pretrained + BiLSTM"))
                    elif 'bigcarp_frozen_' in col.lower():
                        models_to_test.append((col, emb_dim, "BigCarp Frozen + BiLSTM"))
                    elif 'bigcarp_embedder_' in col.lower():
                        models_to_test.append((col, emb_dim, "BigCarp Embedder + BiLSTM"))
                    else:
                        models_to_test.append((col, emb_dim, "BigCarp + BiLSTM"))
            else:
                if 'max' in col.lower() or 'mean' in col.lower():
                    print(f"     Skipping {col} - pooled embedding (single vector)")
                else:
                    print(f"     Skipping {col} - single vector not suitable for BiLSTM")
    
    # Count total models including P2V
    total_models = len(models_to_test)
    if df_p2v is not None:
        total_models += 1
    
    print(f"\n🎯 Models to evaluate: {total_models}")
    print(f"   📊 BiLSTM models ({len(models_to_test)}):")
    for emb_col, emb_dim, name in models_to_test:
        print(f"     • {name} ({emb_col}, dim={emb_dim})")
    
    if df_p2v is not None:
        print(f"   🌲 Random Forest models (1):")
        print(f"     • P2V + Random Forest")
    
    if not models_to_test and df_p2v is None:
        print("❌ No embedding columns found to evaluate!")
        return
    
    # Evaluate all models
    all_results = []
    print(f"\n🚀 Starting evaluation of {len(models_to_test)} models...")
    
    for i, (emb_col, emb_dim, model_name) in enumerate(models_to_test, 1):
        print(f"\n{'🔬' * 3} MODEL {i}/{len(models_to_test)} {'🔬' * 3}")
        result = evaluate_embedding_model(
            df_combined, cv_splits, emb_col, emb_dim, model_name, class_cols, args.seed
        )
        if result is not None:
            all_results.append(result)
            print(f"✅ Model {i} completed successfully!")
        else:
            print(f"❌ Model {i} failed!")
    
    print(f"\n🏁 Completed evaluation of BiLSTM models!")
    
    # Evaluate P2V + Random Forest
    if df_p2v is not None:
        print(f"\n{'🔬' * 3} P2V + RANDOM FOREST {'🔬' * 3}")
        p2v_result = evaluate_p2v_random_forest(df_combined, cv_splits, class_cols, args.seed)
        if p2v_result is not None:
            all_results.append(p2v_result)
            print(f"✅ P2V + Random Forest completed successfully!")
        else:
            print(f"❌ P2V + Random Forest failed!")
    else:
        print(f"\n⚠️  P2V data not available, skipping Random Forest evaluation")
    
    print(f"\n🏁 Completed evaluation of all models!")
    
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
