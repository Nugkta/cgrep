"""
Halogen Presence Prediction: MLP Embedding Comparison Pipeline

This script evaluates protein embeddings for binary classification of halogen presence
using AUC-ROC as the primary performance metric. Six embedding configurations are compared
using MLP classifiers with rigorous statistical validation.

Dataset:
    The input dataset (halogen_pf04820_final_dataset.pkl) contains biosynthetic gene clusters
    (BGCs) that possess the halogenase Pfam domain (PF04820). These BGCs may or may not produce
    halogenated final products. The dataset includes:
        - BGC sequences with halogenase domains
        - Binary labels indicating halogen presence in final products
        - Multiple embedding types (ESM-2, bigcarp domain, bigcarp mean-pooled)

Methodology:
    - Leave-One-Out Cross-Validation (LOOCV) for unbiased AUC-ROC estimation
    - Bootstrap resampling (10,000 samples) for AUC-ROC confidence intervals
    - Paired bootstrap testing for statistical significance of AUC-ROC differences
    - MLP architecture (512, 128 hidden units) for classification

Embeddings Evaluated:
    1. bigcarp Domain: Domain-level bigcarp embeddings
    2. bigcarp Mean Pool: Mean-pooled bigcarp embeddings across sequence
    3. ESM: ESM-2 protein language model embeddings
    4. ESM + bigcarp Domain: Concatenated ESM and bigcarp domain embeddings
    5. ESM + bigcarp Mean Pool: Concatenated ESM and bigcarp mean-pooled embeddings
    6. ESM + bigcarp Domain + Mean Pool: All three embeddings concatenated

Output:
    - AUC-ROC scores with 95% bootstrap confidence intervals
    - Paired statistical comparison of AUC-ROC (ESM vs. ESM+bigcarp MeanPool)
    - Comprehensive visualization plots
    - CSV results table and JSON statistical summaries

Environment Variables:
    DATASET_PATH: Path to pickled dataset (default: data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl)
    N_BOOTSTRAP_SAMPLES: Number of bootstrap samples (default: 10000)

Usage:
    python halogen_embedding_comparison.py

    Or with custom parameters:
    DATASET_PATH=path/to/data.pkl N_BOOTSTRAP_SAMPLES=5000 python halogen_embedding_comparison.py
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
    """
    Load and validate the halogen presence prediction dataset.

    Loads a pickled pandas DataFrame containing protein embeddings and halogen labels,
    then validates the presence of required columns.

    Dataset Description (data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl):
        This is a curated dataset containing biosynthetic gene clusters (BGCs) that possess the halogenase Pfam domain (PF04820) but may or may not produce halogenated final products. The dataset includes various sequence embeddings for prediction tasks.

    Args:
        path: Path to the pickled DataFrame file (.pkl format)

    Returns:
        DataFrame with the following required columns:
            - esm_domain_embedding: ESM-2 protein language model domain embeddings (numpy arrays)
            - bigcarp_domain_embedding: bigcarp domain-level embeddings of BGC sequences (numpy arrays)
            - bigcarp_embedding_mean_pool: Mean-pooled bigcarp embeddings across BGC sequences (numpy arrays)
            - has_halogen: Binary labels (0/1 or bool) indicating whether the BGC produces
                          a halogenated final product (True/1) or not (False/0)

    Raises:
        FileNotFoundError: If the dataset file does not exist at the specified path
        TypeError: If the loaded object is not a pandas DataFrame
        KeyError: If any required columns are missing from the DataFrame
    """
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
    """
    Convert a DataFrame column containing arrays into a stacked 2D numpy array.

    Takes a DataFrame column where each cell contains an array-like object (list, numpy array) and stacks them into a single 2D array suitable for machine learning models.

    Usage:
        Transforms embedding columns from the DataFrame into feature matrices for ML models.

    Args:
        df: DataFrame containing the column to stack
        col: Name of the column containing array-like objects (embeddings)

    Returns:
        2D numpy array of shape (n_samples, n_features) where each row corresponds
        to one DataFrame row and contains the stacked embedding values

    Note:
        Uses np.stack() first for efficiency. If that fails (e.g., inconsistent shapes),
        falls back to converting to list then array with float dtype.
    """
    vals = df[col].values
    try:
        return np.stack(vals)
    except Exception:
        return np.array(list(vals), dtype=float)


def get_mlp_config(input_dim: int) -> Dict:
    """
    Get configuration parameters for the MLP classifier.

    Returns a dictionary of hyperparameters optimized for the halogen prediction task
    using a two-layer neural network architecture.

    Args:
        input_dim: Dimensionality of input features (not currently used but reserved
                   for future adaptive architecture selection)

    Returns:
        Dictionary containing sklearn MLPClassifier parameters:
            - hidden_layer_sizes: Tuple of hidden layer sizes (512, 128)
            - max_iter: Maximum number of training iterations (3000)
            - alpha: L2 regularization parameter (0.01)
            - early_stopping: Whether to use early stopping (True)
            - validation_fraction: Fraction of training data for validation (0.1)
            - n_iter_no_change: Iterations with no improvement before stopping (30)
    """
    return {
        'hidden_layer_sizes': (512, 128),
        'max_iter': 3000,
        'alpha': 0.01,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 30
    }


def loocv_eval(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict:
    """
    Perform Leave-One-Out Cross-Validation (LOOCV) to estimate AUC-ROC.

    For each sample, trains an MLP on all other samples and evaluates on the held-out sample.
    This provides an unbiased estimate of AUC-ROC for small datasets.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,) with values 0 or 1
        random_state: Random seed for reproducible MLP initialization (default: 42)

    Returns:
        Dictionary containing:
            - auc_roc: Area Under the ROC Curve (primary metric)
            - predictions: Array of predicted probabilities for positive class
            - true_labels: Array of true binary labels
            - accuracy, precision, recall, f1: Additional metrics for reference

    Note:
        - Each fold uses independent StandardScaler fitted on training data only
        - If MLP training fails, falls back to random predictions (rare edge case)
    """
    loo = LeaveOneOut()
    preds: List[int] = []
    probs: List[float] = []
    trues: List[int] = []

    mlp_config = get_mlp_config(X.shape[1])

    for train_idx, test_idx in tqdm(loo.split(y), total=len(y), desc="LOOCV", leave=False):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        clf = MLPClassifier(**mlp_config, random_state=random_state)

        try:
            clf.fit(X_tr_scaled, y_tr)
            pred = int(clf.predict(X_te_scaled)[0])
            prob = float(clf.predict_proba(X_te_scaled)[0, 1])
        except Exception:
            pred = int(np.random.choice([0, 1]))
            prob = float(np.random.random())
            print("[WARNING:] MLP training failed, using random prediction.")

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

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
        "predictions": probs,
        "true_labels": trues
    }


def bootstrap_auc_roc(y_true: np.ndarray, y_proba: np.ndarray,
                      n_bootstrap: int = 10000, random_state: int = 42) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for AUC-ROC metric.

    Uses bootstrap resampling to estimate the sampling distribution of AUC-ROC
    and compute 95% confidence intervals around the point estimate.

    Args:
        y_true: True binary labels, array of shape (n_samples,)
        y_proba: Predicted probabilities for positive class, array of shape (n_samples,)
        n_bootstrap: Number of bootstrap samples to draw (default: 10000)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (mean_auc, ci_lower, ci_upper):
            - mean_auc: Mean AUC-ROC across bootstrap samples
            - ci_lower: Lower bound of 95% confidence interval (2.5th percentile)
            - ci_upper: Upper bound of 95% confidence interval (97.5th percentile)

    Note:
        - Bootstrap samples with only one class are skipped (degenerate cases)
        - Failed AUC calculations (rare edge cases) are also skipped
        - Confidence intervals represent uncertainty in the AUC estimate
    """
    np.random.seed(random_state)
    bootstrap_aucs = []

    n_samples = len(y_true)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[boot_indices]
        boot_y_proba = y_proba[boot_indices]

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
                           n_bootstrap: int = 10000, name: str = "") -> Dict:
    """
    Complete evaluation pipeline: LOOCV AUC-ROC + bootstrap confidence intervals.

    Combines Leave-One-Out Cross-Validation for AUC-ROC estimation with bootstrap
    resampling for confidence interval calculation.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Binary labels of shape (n_samples,)
        n_bootstrap: Number of bootstrap samples for CI estimation (default: 10000)
        name: Descriptive name of the embedding being evaluated (for logging)

    Returns:
        Dictionary containing:
            - auc_roc: LOOCV AUC-ROC score (primary metric)
            - bootstrap_mean_auc: Mean AUC from bootstrap resampling
            - bootstrap_ci_lower: Lower 95% CI bound for AUC-ROC
            - bootstrap_ci_upper: Upper 95% CI bound for AUC-ROC
            - predictions: Array of LOOCV predicted probabilities
            - true_labels: Array of true labels
            - mlp_architecture: String representation of MLP hidden layer sizes
            - accuracy, precision, recall, f1: Additional metrics for reference

    Note:
        Prints progress information including embedding name, feature dimensions,
        and MLP architecture during execution.
    """
    print(f"Evaluating {name}...")
    print(f"Feature dimensions: {X.shape}")

    mlp_config = get_mlp_config(X.shape[1])
    print(f"Using MLP architecture: {mlp_config['hidden_layer_sizes']}")

    print("Running LOOCV evaluation...")
    results = loocv_eval(X, y, random_state=42)

    print("Performing bootstrap analysis...")
    mean_boot_auc, ci_lower, ci_upper = bootstrap_auc_roc(
        results['true_labels'], results['predictions'], n_bootstrap=n_bootstrap
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
        'predictions': results['predictions'],
        'true_labels': results['true_labels'],
        'mlp_architecture': str(mlp_config['hidden_layer_sizes'])
    }


def paired_bootstrap_test(y_true: np.ndarray, y_proba1: np.ndarray, y_proba2: np.ndarray,
                         n_bootstrap: int = 10000, random_state: int = 42) -> Dict[str, float]:
    """
    Perform paired bootstrap hypothesis test comparing two models' AUC-ROC scores.

    Uses paired bootstrap resampling to test whether the difference in AUC-ROC between
    two models is statistically significant. The "paired" aspect means both models are
    evaluated on the same bootstrap samples, reducing variance in the difference estimate.

    Args:
        y_true: True binary labels, array of shape (n_samples,)
        y_proba1: Model 1 predicted probabilities, array of shape (n_samples,)
        y_proba2: Model 2 predicted probabilities, array of shape (n_samples,)
        n_bootstrap: Number of bootstrap samples (default: 10000)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing:
            - original_auc1: Model 1 AUC on full dataset
            - original_auc2: Model 2 AUC on full dataset
            - original_diff: Observed difference (Model 2 - Model 1)
            - bootstrap_mean_diff: Mean difference across bootstrap samples
            - bootstrap_std_diff: Standard deviation of difference
            - bootstrap_ci_lower: Lower 95% CI bound for difference
            - bootstrap_ci_upper: Upper 95% CI bound for difference
            - p_value_two_sided: Two-sided p-value for null hypothesis (no difference)
            - bootstrap_diffs: Array of all bootstrap difference values
            - n_bootstrap_samples: Number of valid bootstrap samples

    Note:
        - P-value < 0.05 indicates statistically significant difference at α=0.05
        - CI not including 0 also indicates significant difference
        - Bootstrap samples with only one class are skipped
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


def create_visualization(results_df: pd.DataFrame, paired_results: Dict, output_dir: str = "results/halogen_prediction"):
    """
    Generate 2-panel visualization of embedding comparison results.

    Creates a publication-quality figure showing AUC-ROC performance comparison
    and statistical testing between models.

    Args:
        results_df: DataFrame containing evaluation results with columns:
            - embedding_name: Name of the embedding configuration
            - bootstrap_mean_auc: Mean AUC from bootstrap
            - bootstrap_ci_lower: Lower 95% CI bound
            - bootstrap_ci_upper: Upper 95% CI bound
        paired_results: Dictionary from paired_bootstrap_test() containing:
            - original_diff, bootstrap_mean_diff, etc.
            - bootstrap_diffs: Array of bootstrap differences
        output_dir: Directory to save the visualization (default: "results/halogen_prediction")

    Output:
        Saves a PNG file "mlp_embedding_comparison.png" containing:
            - Panel 1 (left): Horizontal bar chart with AUC + 95% CI error bars
            - Panel 2 (right): Histogram of paired bootstrap differences

    Note:
        - Color coding: ESM+ (red), ESM (blue), bigcarp only (green)
        - Figure saved at 300 DPI for publication quality
        - Also calls plt.show() to display interactively
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Performance comparison bar plot
    embedding_names = results_df['embedding_name']
    aucs = results_df['bootstrap_mean_auc']
    ci_lowers = results_df['bootstrap_ci_lower']
    ci_uppers = results_df['bootstrap_ci_upper']

    errors = [aucs - ci_lowers, ci_uppers - aucs]

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
    ax1.set_title('MLP Embedding Performance Comparison\nBootstrap AUC-ROC with 95% Confidence Intervals', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1)

    for i, (bar, auc, ci_lower, ci_upper) in enumerate(zip(bars, aucs, ci_lowers, ci_uppers)):
        ax1.text(auc + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{auc:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]',
                 va='center', fontsize=9)

    # Plot 2: Bootstrap difference distribution
    bootstrap_diffs = paired_results['bootstrap_diffs']

    ax2.hist(bootstrap_diffs, bins=50, alpha=0.7, density=True,
            color='lightsteelblue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax2.axvline(paired_results['original_diff'], color='green', linestyle='-', linewidth=2,
               label=f'Observed ({paired_results["original_diff"]:.4f})')
    ax2.axvline(paired_results['bootstrap_ci_lower'], color='orange', linestyle=':', linewidth=2)
    ax2.axvline(paired_results['bootstrap_ci_upper'], color='orange', linestyle=':', linewidth=2,
               label=f'95% CI [{paired_results["bootstrap_ci_lower"]:.4f}, {paired_results["bootstrap_ci_upper"]:.4f}]')

    ax2.set_xlabel('AUC Difference (ESM+bigcarp MeanPool - ESM)')
    ax2.set_ylabel('Density')
    ax2.set_title('Paired Bootstrap Difference Distribution\nESM vs ESM+bigcarp MeanPool')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    textstr = f'''Statistical Summary:
Mean Difference: {paired_results["bootstrap_mean_diff"]:+.4f}
Std Difference: {paired_results["bootstrap_std_diff"]:.4f}
P-value (two-sided): {paired_results["p_value_two_sided"]:.4f}
Effective samples: {paired_results["n_bootstrap_samples"]}'''

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mlp_embedding_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Execute complete halogen prediction embedding comparison pipeline.

    Main orchestration function that:
        1. Loads dataset and extracts embeddings
        2. Evaluates 6 embedding configurations using LOOCV + bootstrap
        3. Performs paired statistical comparison (ESM vs ESM+bigcarp MeanPool)
        4. Generates comprehensive visualizations
        5. Saves results to CSV, JSON, and pickle files

    Environment Variables:
        DATASET_PATH: Path to input dataset pickle file
            (default: 'data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl')
        N_BOOTSTRAP_SAMPLES: Number of bootstrap samples for CI estimation
            (default: 10000)

    Output Files (saved to results/halogen_prediction/):
        - mlp_embedding_comparison_results.csv: Performance metrics table
        - mlp_paired_bootstrap_esm_vs_esm_bc_mean.json: Statistical test results
        - mlp_evaluation_predictions.pkl: Predictions for all embeddings
        - mlp_embedding_comparison.png: Visualization figure

    Prints:
        - Progress updates for each pipeline stage
        - Summary table of all results
        - Top 5 performing embeddings
        - Statistical comparison between ESM and ESM+bigcarp MeanPool
    """
    DATASET_PATH = os.getenv('DATASET_PATH', 'data/processed/halogen_prediction/halogen_pf04820_final_dataset.pkl')
    N_BOOTSTRAP = int(os.getenv('N_BOOTSTRAP_SAMPLES', '10000'))
    OUTPUT_DIR = 'results/halogen_prediction'

    print("="*80)
    print("HALOGEN PRESENCE PREDICTION: MLP EMBEDDING COMPARISON")
    print("="*80)

    print("\n1. Loading dataset...")
    df_final = load_final_dataset(DATASET_PATH)
    print(f'   Loaded dataset: {len(df_final)} samples')
    print(f'   Positive class: {df_final["has_halogen"].sum()} samples')
    print(f'   Negative class: {(~df_final["has_halogen"].astype(bool)).sum()} samples')

    y = df_final["has_halogen"].astype(int).values
    esm_emb = stack_col(df_final, "esm_domain_embedding")
    bc_domain_emb = stack_col(df_final, "bigcarp_domain_embedding")
    bc_mean_emb = stack_col(df_final, "bigcarp_embedding_mean_pool")

    print(f'\n   Embedding dimensions:')
    print(f'     ESM: {esm_emb.shape}')
    print(f'     bigcarp domain: {bc_domain_emb.shape}')
    print(f'     bigcarp mean pool: {bc_mean_emb.shape}')

    embedding_configs = [
        ("bigcarp_Domain", bc_domain_emb),
        ("bigcarp_MeanPool", bc_mean_emb),
        ("ESM", esm_emb),
        ("ESM+bigcarp_Domain", np.concatenate([esm_emb, bc_domain_emb], axis=1)),
        ("ESM+bigcarp_MeanPool", np.concatenate([esm_emb, bc_mean_emb], axis=1)),
        ("ESM+bigcarp_Domain+MeanPool", np.concatenate([esm_emb, bc_domain_emb, bc_mean_emb], axis=1)),
    ]

    print(f"\n2. Evaluating {len(embedding_configs)} embedding configurations with MLPs...")
    print(f"   (Using LOOCV + Bootstrap with {N_BOOTSTRAP} samples)")

    results = []
    evaluation_data = {}

    for name, embedding in embedding_configs:
        print(f"\n   {'-'*50}")
        eval_results = evaluate_with_bootstrap(embedding, y, n_bootstrap=N_BOOTSTRAP, name=name)

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
            'bootstrap_ci_upper': eval_results['bootstrap_ci_upper'],
            'mlp_architecture': eval_results['mlp_architecture']
        }
        results.append(result)

        evaluation_data[name] = {
            'predictions': eval_results['predictions'],
            'true_labels': eval_results['true_labels']
        }

        print(f"   {name}: AUC = {eval_results['bootstrap_mean_auc']:.4f} "
              f"[{eval_results['bootstrap_ci_lower']:.4f}, {eval_results['bootstrap_ci_upper']:.4f}] "
              f"| MLP: {eval_results['mlp_architecture']}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('bootstrap_mean_auc', ascending=False).reset_index(drop=True)

    print(f"\n3. Results Summary (AUC-ROC Rankings):")
    print("="*80)
    for idx, row in results_df.iterrows():
        print(f"{idx+1:2}. {row['embedding_name']:35} | AUC: {row['bootstrap_mean_auc']:.4f} "
              f"[{row['bootstrap_ci_lower']:.4f}, {row['bootstrap_ci_upper']:.4f}]")

    # Paired Bootstrap Analysis: ESM vs ESM+bigcarp_MeanPool
    print(f"\n4. Paired Bootstrap Analysis: ESM vs ESM+bigcarp_MeanPool")
    print("="*80)

    esm_data = evaluation_data['ESM']
    esm_bc_mean_data = evaluation_data['ESM+bigcarp_MeanPool']

    paired_results = paired_bootstrap_test(
        esm_data['true_labels'],
        esm_data['predictions'],
        esm_bc_mean_data['predictions'],
        n_bootstrap=N_BOOTSTRAP
    )

    esm_results = results_df[results_df['embedding_name']=='ESM'].iloc[0]
    concat_results = results_df[results_df['embedding_name']=='ESM+bigcarp_MeanPool'].iloc[0]

    print(f"   ESM AUC-ROC:                {esm_results['bootstrap_mean_auc']:.4f} [{esm_results['bootstrap_ci_lower']:.4f}, {esm_results['bootstrap_ci_upper']:.4f}]")
    print(f"   ESM+bigcarp MeanPool AUC:   {concat_results['bootstrap_mean_auc']:.4f} [{concat_results['bootstrap_ci_lower']:.4f}, {concat_results['bootstrap_ci_upper']:.4f}]")
    print(f"   Difference (ESM+BC - ESM):  {paired_results['original_diff']:+.4f} [{paired_results['bootstrap_ci_lower']:+.4f}, {paired_results['bootstrap_ci_upper']:+.4f}]")
    print(f"   P-value:                    {paired_results['p_value_two_sided']:.4f} ({'significant' if paired_results['p_value_two_sided'] < 0.05 else 'not significant'})")

    # Create visualizations
    print(f"\n5. Creating visualizations...")
    create_visualization(results_df, paired_results, OUTPUT_DIR)

    # Save results
    print(f"\n6. Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_file = f"{OUTPUT_DIR}/mlp_embedding_comparison_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"   All metrics saved to: {results_file}")

    paired_file = f"{OUTPUT_DIR}/mlp_paired_bootstrap_esm_vs_esm_bc_mean.json"
    paired_results_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in paired_results.items()}

    with open(paired_file, 'w') as f:
        json.dump(paired_results_json, f, indent=2)
    print(f"   Statistical test saved to: {paired_file}")

    eval_file = f"{OUTPUT_DIR}/mlp_evaluation_predictions.pkl"
    with open(eval_file, 'wb') as f:
        pickle.dump(evaluation_data, f)
    print(f"   Predictions saved to: {eval_file}")

    print(f"\n" + "="*80)
    print(f"Pipeline completed successfully!")
    print(f"Best performing: {results_df.iloc[0]['embedding_name']} (AUC-ROC: {results_df.iloc[0]['bootstrap_mean_auc']:.4f})")
    print("="*80)


if __name__ == "__main__":
    main()