#!/usr/bin/env python3
"""
Halogen Presence Prediction: Comprehensive Embedding Comparison Pipeline

This script compares different protein embeddings for halogen presence prediction using:
- Leave-One-Out Cross-Validation (LOOCV)
- Bootstrap confidence intervals
- Paired bootstrap statistical testing

Embeddings compared:
1. BigCarp Domain
2. BigCarp Mean Pool
3. ESM
4. ESM + BigCarp Domain
5. ESM + BigCarp Mean Pool
6. ESM + BigCarp Domain + Mean Pool
"""

import os
import pickle
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_final_dataset(path: str) -> pd.DataFrame:
    """Load the halogen dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found: {path}')
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError('Expected a pickled pandas DataFrame.')
    required_cols = ['esm_domain_embedding', 'bigcarp_domain_embedding',
                     'bigcarp_embedding_mean_pool', 'has_halogen']
    missing = [c for c in required_cols if c not in obj.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')
    return obj


def stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack column values into numpy array."""
    vals = df[col].values
    try:
        return np.stack(vals)
    except Exception:
        return np.array(list(vals), dtype=float)


def loocv_eval(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict[str, float]:
    """Evaluate using Leave-One-Out Cross-Validation."""
    loo = LeaveOneOut()
    preds: List[int] = []
    probs: List[float] = []
    trues: List[int] = []

    for train_idx, test_idx in tqdm(loo.split(y), total=len(y), desc="LOOCV", leave=False):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=2000,
            alpha=0.1,
            random_state=random_state,
            early_stopping=False,
        )
        clf.fit(X_tr_scaled, y_tr)
        pred = int(clf.predict(X_te_scaled)[0])
        prob = float(clf.predict_proba(X_te_scaled)[0, 1])
        preds.append(pred)
        probs.append(prob)
        trues.append(int(y_te[0]))

    preds = np.array(preds)
    probs = np.array(probs)
    trues = np.array(trues)
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = 0.5
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def bootstrap_auc_roc(y_true: np.ndarray, y_proba: np.ndarray,
                      n_bootstrap: int = 10000, random_state: int = 42) -> Tuple[float, float, float]:
    """Bootstrap sampling for AUC-ROC with confidence intervals."""
    np.random.seed(random_state)
    bootstrap_aucs = []

    n_samples = len(y_true)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        # Bootstrap sample indices
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[boot_indices]
        boot_y_proba = y_proba[boot_indices]

        # Skip if only one class present
        if len(np.unique(boot_y_true)) < 2:
            continue

        try:
            boot_auc = roc_auc_score(boot_y_true, boot_y_proba)
            bootstrap_aucs.append(boot_auc)
        except Exception:
            continue

    bootstrap_aucs = np.array(bootstrap_aucs)
    mean_auc = np.mean(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)

    return mean_auc, ci_lower, ci_upper


def evaluate_with_bootstrap(X: np.ndarray, y: np.ndarray,
                           n_bootstrap: int = 10000, name: str = "") -> Dict[str, float]:
    """Evaluate model performance with bootstrap confidence intervals."""
    print(f"Evaluating {name}...")
    print(f"Feature dimensions: {X.shape}")

    # Single LOOCV run for metrics
    print("Running LOOCV evaluation...")
    results = loocv_eval(X, y, random_state=42)

    # Get predictions for bootstrap analysis
    print("Getting predictions for bootstrap analysis...")
    loo = LeaveOneOut()
    probs: List[float] = []
    trues: List[int] = []

    for train_idx, test_idx in tqdm(loo.split(y), total=len(y), desc="Final LOOCV", leave=False):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=2000,
            alpha=0.1,
            random_state=42,
            early_stopping=False,
        )
        clf.fit(X_tr_scaled, y_tr)
        prob = float(clf.predict_proba(X_te_scaled)[0, 1])
        probs.append(prob)
        trues.append(int(y_te[0]))

    # Bootstrap analysis
    print("Performing bootstrap analysis...")
    mean_boot_auc, ci_lower, ci_upper = bootstrap_auc_roc(
        np.array(trues), np.array(probs), n_bootstrap=n_bootstrap
    )

    return {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'auc_roc': results['auc_roc'],
        'bootstrap_mean_auc': mean_boot_auc,
        'bootstrap_ci_lower': ci_lower,
        'bootstrap_ci_upper': ci_upper,
        'predictions': np.array(probs),
        'true_labels': np.array(trues)
    }


def paired_bootstrap_test(y_true: np.ndarray, y_proba1: np.ndarray, y_proba2: np.ndarray,
                         n_bootstrap: int = 10000, random_state: int = 42) -> Dict[str, float]:
    """
    Paired bootstrap test comparing two models' AUC-ROC scores.
    Returns difference statistics and p-value.
    """
    np.random.seed(random_state)

    # Original AUC difference
    try:
        auc1 = roc_auc_score(y_true, y_proba1)
        auc2 = roc_auc_score(y_true, y_proba2)
        original_diff = auc2 - auc1  # Model 2 - Model 1
    except:
        return {"error": "Could not compute original AUCs"}

    bootstrap_diffs = []
    n_samples = len(y_true)

    for _ in tqdm(range(n_bootstrap), desc="Paired bootstrap", leave=False):
        # Bootstrap sample indices (same for both models - paired)
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[boot_indices]
        boot_y_proba1 = y_proba1[boot_indices]
        boot_y_proba2 = y_proba2[boot_indices]

        # Skip if only one class present
        if len(np.unique(boot_y_true)) < 2:
            continue

        try:
            boot_auc1 = roc_auc_score(boot_y_true, boot_y_proba1)
            boot_auc2 = roc_auc_score(boot_y_true, boot_y_proba2)
            boot_diff = boot_auc2 - boot_auc1
            bootstrap_diffs.append(boot_diff)
        except Exception:
            continue

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Statistics
    mean_diff = np.mean(bootstrap_diffs)
    std_diff = np.std(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # Two-sided p-value: proportion where difference includes 0
    p_value_two_sided = 2 * min(np.mean(bootstrap_diffs <= 0), np.mean(bootstrap_diffs >= 0))

    return {
        "original_auc1": auc1,
        "original_auc2": auc2,
        "original_diff": original_diff,
        "bootstrap_mean_diff": mean_diff,
        "bootstrap_std_diff": std_diff,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "p_value_two_sided": p_value_two_sided,
        "bootstrap_diffs": bootstrap_diffs,
        "n_bootstrap_samples": len(bootstrap_diffs)
    }


def create_visualization(results_df: pd.DataFrame, paired_results: Dict = None,
                        output_dir: str = "artifacts/halogen_presence"):
    """Create comprehensive visualizations of results."""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Performance table visualization (top left)
    ax1 = plt.subplot(2, 3, (1, 2))

    # Create performance comparison bar plot
    embedding_names = results_df['embedding_name']
    aucs = results_df['bootstrap_mean_auc']
    ci_lowers = results_df['bootstrap_ci_lower']
    ci_uppers = results_df['bootstrap_ci_upper']

    # Error bars
    errors = [aucs - ci_lowers, ci_uppers - aucs]

    # Color by embedding type
    colors = []
    for name in embedding_names:
        if 'ESM+' in name:
            colors.append('lightcoral')
        elif name == 'ESM':
            colors.append('skyblue')
        else:
            colors.append('lightgreen')

    bars = ax1.barh(range(len(embedding_names)), aucs, xerr=errors,
                    capsize=5, alpha=0.8, color=colors)

    ax1.set_yticks(range(len(embedding_names)))
    ax1.set_yticklabels(embedding_names, fontsize=10)
    ax1.set_xlabel('Bootstrap Mean AUC-ROC', fontsize=12)
    ax1.set_title('Embedding Performance Comparison\nBootstrap AUC-ROC with 95% Confidence Intervals', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1)

    # Add value labels
    for i, (bar, auc, ci_lower, ci_upper) in enumerate(zip(bars, aucs, ci_lowers, ci_uppers)):
        ax1.text(auc + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{auc:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]',
                 va='center', fontsize=9)

    # Plot 2: All metrics comparison (top right)
    ax2 = plt.subplot(2, 3, 3)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_data = results_df[metrics].values

    im = ax2.imshow(metric_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m.capitalize() for m in metrics], rotation=45)
    ax2.set_yticks(range(len(embedding_names)))
    ax2.set_yticklabels(embedding_names, fontsize=9)
    ax2.set_title('All Metrics Heatmap', fontsize=12)

    # Add text annotations
    for i in range(len(embedding_names)):
        for j in range(len(metrics)):
            text = ax2.text(j, i, f'{metric_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax2, shrink=0.6)

    # Plot 3: Feature dimensions bar chart (bottom left)
    ax3 = plt.subplot(2, 3, 4)

    n_features = results_df['n_features']
    bars3 = ax3.bar(range(len(embedding_names)), n_features, alpha=0.7, color=colors)
    ax3.set_xticks(range(len(embedding_names)))
    ax3.set_xticklabels(embedding_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Feature Dimensions by Embedding Type')
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, n_feat in zip(bars3, n_features):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(n_feat), ha='center', va='bottom', fontsize=9)

    # Plot 4: Bootstrap difference distribution (bottom center and right)
    if paired_results:
        ax4 = plt.subplot(2, 3, (5, 6))

        bootstrap_diffs = paired_results['bootstrap_diffs']

        ax4.hist(bootstrap_diffs, bins=50, alpha=0.7, density=True,
                color='lightsteelblue', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax4.axvline(paired_results['original_diff'], color='green', linestyle='-', linewidth=2,
                   label=f'Observed ({paired_results["original_diff"]:.4f})')
        ax4.axvline(paired_results['bootstrap_ci_lower'], color='orange', linestyle=':', linewidth=2)
        ax4.axvline(paired_results['bootstrap_ci_upper'], color='orange', linestyle=':', linewidth=2,
                   label=f'95% CI [{paired_results["bootstrap_ci_lower"]:.4f}, {paired_results["bootstrap_ci_upper"]:.4f}]')

        ax4.set_xlabel('AUC Difference (ESM+BigCarp Mean Pool - ESM)')
        ax4.set_ylabel('Density')
        ax4.set_title('Paired Bootstrap Difference Distribution\nESM vs ESM+BigCarp Mean Pool')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # Add statistical info as text
        textstr = f'''Statistical Summary:
Mean Difference: {paired_results["bootstrap_mean_diff"]:+.4f}
Std Difference: {paired_results["bootstrap_std_diff"]:.4f}
P-value (two-sided): {paired_results["p_value_two_sided"]:.4f}
Effective samples: {paired_results["n_bootstrap_samples"]}'''

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/comprehensive_embedding_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main pipeline execution."""

    # Configuration
    DATASET_PATH = 'data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl'
    N_BOOTSTRAP = 10000
    OUTPUT_DIR = 'results/halogen_presence'

    print("="*80)
    print("HALOGEN PRESENCE PREDICTION: COMPREHENSIVE EMBEDDING COMPARISON")
    print("="*80)

    # Load dataset
    print("\n1. Loading dataset...")
    df_final = load_final_dataset(DATASET_PATH)
    print(f'   Loaded dataset: {len(df_final)} samples')
    print(f'   Positive class: {df_final["has_halogen"].sum()} samples')
    print(f'   Negative class: {(~df_final["has_halogen"].astype(bool)).sum()} samples')

    # Prepare data
    y = df_final["has_halogen"].astype(int).values
    esm_emb = stack_col(df_final, "esm_domain_embedding")
    bc_domain_emb = stack_col(df_final, "bigcarp_domain_embedding")
    bc_mean_emb = stack_col(df_final, "bigcarp_embedding_mean_pool")

    print(f'\n   Embedding dimensions:')
    print(f'     ESM: {esm_emb.shape}')
    print(f'     BigCarp domain: {bc_domain_emb.shape}')
    print(f'     BigCarp mean pool: {bc_mean_emb.shape}')

    # Define embedding configurations
    embedding_configs = [
        ("BigCarp_Domain", bc_domain_emb),
        ("BigCarp_MeanPool", bc_mean_emb),
        ("ESM", esm_emb),
        ("ESM+BigCarp_Domain", np.concatenate([esm_emb, bc_domain_emb], axis=1)),
        ("ESM+BigCarp_MeanPool", np.concatenate([esm_emb, bc_mean_emb], axis=1)),
        ("ESM+BigCarp_Domain+MeanPool", np.concatenate([esm_emb, bc_domain_emb, bc_mean_emb], axis=1)),
    ]

    print(f"\n2. Evaluating {len(embedding_configs)} embedding configurations...")
    print("   (Using LOOCV + Bootstrap with {} samples)".format(N_BOOTSTRAP))

    # Evaluate all embeddings
    results = []
    evaluation_data = {}  # Store for paired analysis

    for name, embedding in embedding_configs:
        print(f"\n   {'-'*50}")
        eval_results = evaluate_with_bootstrap(embedding, y, n_bootstrap=N_BOOTSTRAP, name=name)

        # Store results
        result = {
            'embedding_name': name,
            'n_features': embedding.shape[1],
            'accuracy': eval_results['accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1': eval_results['f1'],
            'auc_roc': eval_results['auc_roc'],
            'bootstrap_mean_auc': eval_results['bootstrap_mean_auc'],
            'bootstrap_ci_lower': eval_results['bootstrap_ci_lower'],
            'bootstrap_ci_upper': eval_results['bootstrap_ci_upper']
        }
        results.append(result)

        # Store evaluation data for paired analysis
        evaluation_data[name] = {
            'predictions': eval_results['predictions'],
            'true_labels': eval_results['true_labels']
        }

        print(f"   {name}: AUC = {eval_results['bootstrap_mean_auc']:.4f} "
              f"[{eval_results['bootstrap_ci_lower']:.4f}, {eval_results['bootstrap_ci_upper']:.4f}]")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('bootstrap_mean_auc', ascending=False).reset_index(drop=True)

    print(f"\n3. Results Summary:")
    print("="*80)
    print(results_df.round(4).to_string(index=False))

    # Paired Bootstrap Analysis: ESM vs ESM+BigCarp_MeanPool
    print(f"\n4. Paired Bootstrap Analysis: ESM vs ESM+BigCarp_MeanPool")
    print("="*80)

    esm_data = evaluation_data['ESM']
    esm_bc_mean_data = evaluation_data['ESM+BigCarp_MeanPool']

    paired_results = paired_bootstrap_test(
        esm_data['true_labels'],
        esm_data['predictions'],
        esm_bc_mean_data['predictions'],
        n_bootstrap=N_BOOTSTRAP
    )

    print(f"\n   BASELINE PERFORMANCE COMPARISON:")
    print(f"   ESM only:                  Accuracy: {results_df[results_df['embedding_name']=='ESM']['accuracy'].iloc[0]:.4f}, "
          f"F1: {results_df[results_df['embedding_name']=='ESM']['f1'].iloc[0]:.4f}, "
          f"AUC: {results_df[results_df['embedding_name']=='ESM']['auc_roc'].iloc[0]:.4f}")
    print(f"   ESM + BC_mean:             Accuracy: {results_df[results_df['embedding_name']=='ESM+BigCarp_MeanPool']['accuracy'].iloc[0]:.4f}, "
          f"F1: {results_df[results_df['embedding_name']=='ESM+BigCarp_MeanPool']['f1'].iloc[0]:.4f}, "
          f"AUC: {results_df[results_df['embedding_name']=='ESM+BigCarp_MeanPool']['auc_roc'].iloc[0]:.4f}")

    # Calculate differences
    esm_results = results_df[results_df['embedding_name']=='ESM'].iloc[0]
    concat_results = results_df[results_df['embedding_name']=='ESM+BigCarp_MeanPool'].iloc[0]

    acc_diff = concat_results['accuracy'] - esm_results['accuracy']
    f1_diff = concat_results['f1'] - esm_results['f1']
    auc_diff = concat_results['auc_roc'] - esm_results['auc_roc']

    print(f"\n   Differences (Concat - ESM):")
    print(f"   Accuracy: {acc_diff:+.4f}")
    print(f"   F1:       {f1_diff:+.4f}")
    print(f"   AUC:      {auc_diff:+.4f}")

    print(f"\n   DETAILED STATISTICAL SUMMARY:")
    print(f"   {'-'*40}")
    print(f"   ESM only AUC-ROC: {paired_results['original_auc1']:.4f}")
    print(f"   ESM + BC_mean AUC-ROC: {paired_results['original_auc2']:.4f}")
    print(f"   Observed difference: {paired_results['original_diff']:+.4f}")
    print(f"   Bootstrap mean: {paired_results['bootstrap_mean_diff']:+.4f}")
    print(f"   Bootstrap std: {paired_results['bootstrap_std_diff']:.4f}")
    print(f"   95% CI: [{paired_results['bootstrap_ci_lower']:+.4f}, {paired_results['bootstrap_ci_upper']:+.4f}]")
    print(f"   CI includes zero: {'Yes' if paired_results['bootstrap_ci_lower'] <= 0 <= paired_results['bootstrap_ci_upper'] else 'No'}")
    print(f"   P-value: {paired_results['p_value_two_sided']:.4f}")
    print(f"   Significant (α=0.05): {'No' if paired_results['p_value_two_sided'] >= 0.05 else 'Yes'}")

    # Interpretation
    print(f"\n   INTERPRETATION:")
    if paired_results['bootstrap_ci_lower'] > 0:
        print(f"   ✅ ESM+BigCarp_MeanPool is SIGNIFICANTLY better than ESM alone")
    elif paired_results['bootstrap_ci_upper'] < 0:
        print(f"   ❌ ESM alone is SIGNIFICANTLY better than ESM+BigCarp_MeanPool")
    else:
        print(f"   ⚠️  No significant difference between ESM and ESM+BigCarp_MeanPool")

    if paired_results['p_value_two_sided'] < 0.05:
        print(f"   📊 Statistical significance: p < 0.05")
    else:
        print(f"   📊 No statistical significance: p ≥ 0.05")

    # Create visualizations
    print(f"\n5. Creating visualizations...")
    create_visualization(results_df, paired_results, OUTPUT_DIR)

    # Save results
    print(f"\n6. Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save comprehensive results
    results_file = f"{OUTPUT_DIR}/embedding_comparison_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"   📁 Results table saved to: {results_file}")

    # Save paired bootstrap results
    paired_file = f"{OUTPUT_DIR}/paired_bootstrap_esm_vs_esm_bc_mean.json"
    # Convert numpy arrays to lists for JSON serialization
    paired_results_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in paired_results.items()}

    with open(paired_file, 'w') as f:
        json.dump(paired_results_json, f, indent=2)
    print(f"   📁 Paired bootstrap results saved to: {paired_file}")

    # Save evaluation data
    eval_file = f"{OUTPUT_DIR}/evaluation_predictions.pkl"
    with open(eval_file, 'wb') as f:
        pickle.dump(evaluation_data, f)
    print(f"   📁 Evaluation predictions saved to: {eval_file}")

    print(f"\n🎉 Pipeline completed successfully!")
    print(f"   Best performing embedding: {results_df.iloc[0]['embedding_name']} "
          f"(AUC: {results_df.iloc[0]['bootstrap_mean_auc']:.4f})")


if __name__ == "__main__":
    main()