"""
MIBiG 3.0 Layer Ablation Study for BGC Classification
======================================================

This script evaluates BGC classification performance using embeddings extracted from
different layers of the BigCARP model. This is similar to layer ablation studies
performed on BERT and other transformer models.

Layer Configurations Evaluated:
    - last: Final layer (layer 31) - current baseline
    - embedder: Raw embedding layer (no contextualization)
    - early: Layer 4 (early contextualization)
    - early_mid: Layer 8
    - middle: Layer 16 (middle of network)
    - late_mid: Layer 24
    - second_last: Layer 30
    - avg_last_4: Average of layers 28-31
    - avg_middle_4: Average of middle 4 layers
    - avg_all: Average of all 32 layers

Usage:
    python scripts/classification/mibig3_layer_evaluation.py \
        --embeddings-dir artifacts/classification/mibig3_layer_analysis \
        --model-name random_init \
        --outdir results/bgc_classification/layer_analysis \
        --seed 42

Output:
    - layer_comparison.csv: Performance metrics for each layer configuration
    - complete_results.pkl: Full results with fold-level details
    - layer_performance_plot.png: Visualization of layer performance
"""

import os
import argparse
import pathlib
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.model_selection import IterativeStratification
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from the base evaluation script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mibig3_stratified_evaluation import (
    convert_product_classes_to_binary,
    create_stratified_splits,
    compute_comprehensive_metrics,
    _validate_and_prepare_XY,
    create_mean_pooled_features,
    tensor_to_list
)


# Layer configurations matching extract_mibig_layer_embeddings.py
LAYER_CONFIGS = [
    "last", "embedder", "early", "early_mid", "middle", 
    "late_mid", "second_last", "avg_last_4", "avg_middle_4", "avg_all"
]


def load_layer_embeddings(embeddings_dir, model_name, layer_config):
    """
    Load embeddings for a specific layer configuration.
    
    Args:
        embeddings_dir (str): Base directory containing embeddings
        model_name (str): Model name (e.g., 'random_init', 'esm_init')
        layer_config (str): Layer configuration name
        
    Returns:
        pandas.DataFrame or None: DataFrame with columns ['bgc_id', 'embeddings', 'product_class']
    """
    filepath = os.path.join(embeddings_dir, model_name, f"mibig3_bigcarp_{layer_config}.pkl")
    
    if not os.path.exists(filepath):
        print(f"   [WARNING] File not found: {filepath}")
        return None
    
    try:
        df = pd.read_pickle(filepath)
        print(f"   [OK] Loaded {layer_config}: {df.shape}")
        return df
    except Exception as e:
        print(f"   [ERROR] Failed to load {layer_config}: {e}")
        return None


def evaluate_layer_mlp(df, cv_splits, emb_col, emb_dim, layer_name, class_cols, seed=42):
    """
    Evaluate MLP classifier using embeddings from a specific layer.
    
    Args:
        df (pandas.DataFrame): DataFrame with embeddings and labels
        cv_splits (list): List of (train_idx, test_idx) tuples
        emb_col (str): Name of embedding column
        emb_dim (int): Expected embedding dimension
        layer_name (str): Name of layer configuration (for display)
        class_cols (list): List of class column names
        seed (int): Random seed
        
    Returns:
        dict: Results dictionary with metrics and fold details
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {layer_name}")
    print(f"{'='*70}")
    
    # Prepare embeddings - these are already mean-pooled 1D vectors
    print(f"   Preparing features from {emb_col}...")
    X_raw = df[emb_col].values
    y_raw = df[class_cols].values
    
    # Convert to numpy arrays if needed and validate dimensions
    X = []
    y = []
    for xi, yi in zip(X_raw, y_raw):
        if xi is None:
            continue
        arr = np.asarray(xi)
        # Layer embeddings are already mean-pooled 1D vectors
        if arr.ndim == 1 and len(arr) == emb_dim:
            X.append(arr)
            y.append(yi)
    
    if len(X) == 0:
        print(f"   [ERROR] No valid samples after filtering!")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Feature matrix shape: {X.shape}")
    
    # Prepare multi-label strings
    label_strings = [";".join([c for c in class_cols if row[c]==1])
                     for _, row in df.iterrows()]
    
    # Filter to valid samples
    label_strings_filtered = [label_strings[i] for i in range(len(y)) if i < len(label_strings)]
    
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform([s.split(';') if s else [] for s in label_strings_filtered])
    
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\nFold {fold_idx + 1}/5: Train={len(train_idx)}, Test={len(test_idx)}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            # Train OneVsRest MLP
            mlp_base = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # Use n_jobs=1 to avoid pickle protocol 5 issues in Python 3.7
            clf = OneVsRestClassifier(mlp_base, n_jobs=1)
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)
            
            # Compute fold metrics
            fold_metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba, mlb.classes_)
            fold_metrics['fold'] = fold_idx
            fold_metrics['y_true'] = y_test
            fold_metrics['y_pred'] = y_pred
            fold_metrics['y_proba'] = y_proba
            fold_results.append(fold_metrics)
            
            # Collect for aggregate
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            
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
    
    # Compute mean and std across folds
    fold_exact_match = [f['exact_match_accuracy'] for f in fold_results]
    fold_macro_f1 = [f['macro_f1'] for f in fold_results]
    fold_micro_f1 = [f['micro_f1'] for f in fold_results]
    fold_weighted_f1 = [f['weighted_macro_f1'] for f in fold_results]
    fold_macro_auc = [f['macro_auc'] for f in fold_results]
    fold_micro_auc = [f['micro_auc'] for f in fold_results]
    fold_weighted_auc = [f['weighted_auc'] for f in fold_results]
    
    # Create results summary
    results = {
        'model': layer_name,
        'exact_match_mean': np.mean(fold_exact_match),
        'exact_match_std': np.std(fold_exact_match),
        'macro_f1_mean': np.mean(fold_macro_f1),
        'macro_f1_std': np.std(fold_macro_f1),
        'micro_f1_mean': np.mean(fold_micro_f1),
        'micro_f1_std': np.std(fold_micro_f1),
        'weighted_f1_mean': np.mean(fold_weighted_f1),
        'weighted_f1_std': np.std(fold_weighted_f1),
        'macro_auc_mean': np.mean(fold_macro_auc),
        'macro_auc_std': np.std(fold_macro_auc),
        'micro_auc_mean': np.mean(fold_micro_auc),
        'micro_auc_std': np.std(fold_micro_auc),
        'weighted_auc_mean': np.mean(fold_weighted_auc),
        'weighted_auc_std': np.std(fold_weighted_auc),
        'aggregate_metrics': aggregate_metrics,
        'fold_results': fold_results,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    print(f"\n{'='*70}")
    print(f"Summary for {layer_name}:")
    print(f"   Exact Match: {results['exact_match_mean']:.4f} ± {results['exact_match_std']:.4f}")
    print(f"   Macro F1:    {results['macro_f1_mean']:.4f} ± {results['macro_f1_std']:.4f}")
    print(f"   Macro AUC:   {results['macro_auc_mean']:.4f} ± {results['macro_auc_std']:.4f}")
    print(f"{'='*70}")
    
    return results


def create_layer_comparison_table(all_results, outdir):
    """
    Create comparison table for layer ablation results.
    
    Args:
        all_results (list): List of result dictionaries
        outdir (str): Output directory
    """
    comparison_data = []
    
    for result in all_results:
        comparison_data.append({
            'Layer': result['model'],
            'N_Samples': result['n_samples'],
            'N_Features': result['n_features'],
            'Exact_Match_Mean': result['exact_match_mean'],
            'Exact_Match_Std': result['exact_match_std'],
            'Macro_F1_Mean': result['macro_f1_mean'],
            'Macro_F1_Std': result['macro_f1_std'],
            'Micro_F1_Mean': result['micro_f1_mean'],
            'Micro_F1_Std': result['micro_f1_std'],
            'Weighted_F1_Mean': result['weighted_f1_mean'],
            'Weighted_F1_Std': result['weighted_f1_std'],
            'Macro_AUC_Mean': result['macro_auc_mean'],
            'Macro_AUC_Std': result['macro_auc_std'],
            'Micro_AUC_Mean': result['micro_auc_mean'],
            'Micro_AUC_Std': result['micro_auc_std'],
            'Weighted_AUC_Mean': result['weighted_auc_mean'],
            'Weighted_AUC_Std': result['weighted_auc_std'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by Macro AUC (descending)
    comparison_df = comparison_df.sort_values('Macro_AUC_Mean', ascending=False)
    
    # Save to CSV
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    output_path = f"{outdir}/layer_comparison.csv"
    comparison_df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print("Layer Comparison Table")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    print(f"\nTable saved to: {output_path}")


def plot_layer_performance(all_results, outdir):
    """
    Create visualization of layer performance.
    
    Args:
        all_results (list): List of result dictionaries
        outdir (str): Output directory
    """
    # Prepare data for plotting
    layers = [r['model'] for r in all_results]
    macro_auc_means = [r['macro_auc_mean'] for r in all_results]
    macro_auc_stds = [r['macro_auc_std'] for r in all_results]
    macro_f1_means = [r['macro_f1_mean'] for r in all_results]
    macro_f1_stds = [r['macro_f1_std'] for r in all_results]
    exact_match_means = [r['exact_match_mean'] for r in all_results]
    exact_match_stds = [r['exact_match_std'] for r in all_results]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Macro AUC
    axes[0].barh(layers, macro_auc_means, xerr=macro_auc_stds, capsize=5, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Macro AUC', fontsize=12)
    axes[0].set_title('Macro AUC by Layer', fontsize=14, fontweight='bold')
    axes[0].set_xlim([min(macro_auc_means) - 0.05, max(macro_auc_means) + 0.05])
    axes[0].axvline(x=max(macro_auc_means), color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Macro F1
    axes[1].barh(layers, macro_f1_means, xerr=macro_f1_stds, capsize=5, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Macro F1', fontsize=12)
    axes[1].set_title('Macro F1 by Layer', fontsize=14, fontweight='bold')
    axes[1].set_xlim([min(macro_f1_means) - 0.05, max(macro_f1_means) + 0.05])
    axes[1].axvline(x=max(macro_f1_means), color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Exact Match
    axes[2].barh(layers, exact_match_means, xerr=exact_match_stds, capsize=5, color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('Exact Match Accuracy', fontsize=12)
    axes[2].set_title('Exact Match by Layer', fontsize=14, fontweight='bold')
    axes[2].set_xlim([min(exact_match_means) - 0.05, max(exact_match_means) + 0.05])
    axes[2].axvline(x=max(exact_match_means), color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{outdir}/layer_performance_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer ablation study for MIBiG 3.0 classification")
    parser.add_argument("--embeddings-dir", required=True,
                       help="Directory containing layer-specific embeddings")
    parser.add_argument("--model-name", required=True,
                       help="Model name (e.g., random_init, esm_init)")
    parser.add_argument("--layer-configs", nargs='+', 
                       choices=LAYER_CONFIGS,
                       default=LAYER_CONFIGS,
                       help="Layer configurations to evaluate")
    parser.add_argument("--outdir", default="results/bgc_classification/layer_analysis",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    print("="*70)
    print("MIBiG 3.0 Layer Ablation Study")
    print("="*70)
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Layer configs: {args.layer_configs}")
    print(f"Output dir: {args.outdir}")
    print(f"Seed: {args.seed}")
    print("="*70)
    print()
    
    all_results = []
    
    for layer_config in args.layer_configs:
        print(f"\n{'='*70}")
        print(f"Loading embeddings for: {layer_config}")
        print(f"{'='*70}")
        
        # Load embeddings
        df = load_layer_embeddings(args.embeddings_dir, args.model_name, layer_config)
        
        if df is None:
            print(f"   [SKIP] Could not load embeddings for {layer_config}")
            continue
        
        # Prepare data
        emb_col = 'embeddings'
        df_prep, class_cols = convert_product_classes_to_binary(df)
        
        # Create stratified splits
        cv_splits = create_stratified_splits(df_prep, class_cols, n_splits=5, random_state=args.seed)
        
        # Get embedding dimension
        sample_emb = df_prep[emb_col].dropna().iloc[0]
        if isinstance(sample_emb, np.ndarray):
            emb_dim = sample_emb.shape[0]
        else:
            print(f"   [ERROR] Unexpected embedding format for {layer_config}")
            continue
        
        # Evaluate
        result = evaluate_layer_mlp(
            df_prep, cv_splits, emb_col, emb_dim, 
            layer_config, class_cols, args.seed
        )
        
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\n[ERROR] No results to save!")
        return
    
    # Create comparison table
    create_layer_comparison_table(all_results, args.outdir)
    
    # Save complete results
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.outdir}/complete_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create visualization
    plot_layer_performance(all_results, args.outdir)
    
    print(f"\n{'='*70}")
    print("Layer ablation study complete!")
    print(f"Results saved to: {args.outdir}/")
    print("="*70)


if __name__ == "__main__":
    main()
