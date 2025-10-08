"""
A-Domain Substrate Properties Prediction with Bootstrap Confidence Intervals

This script evaluates protein embeddings for multi-property binary classification of
A-domain substrate characteristics using stratified cross-validation with bootstrap
confidence intervals. Multiple embedding configurations are compared using MLP classifiers
with rigorous statistical validation.

Dataset:
    The input dataset (adomain_training_dataset_full.pkl) contains A-domain sequences
    from nonribosomal peptide synthetases (NRPS) with their substrate properties.
    Each A-domain is labeled with multiple binary substrate characteristics:
        - is_aromatic: Whether substrate has aromatic ring(s)
        - has_heterocycle: Whether substrate contains heterocyclic structure
        - high_polarity: Whether substrate has high polarity
        - is_Val: Whether substrate is valine
        - is_Gly: Whether substrate is glycine
        - is_canonical_aa: Whether substrate is a canonical amino acid

    The dataset includes multiple embedding representations:
        - ESM-2 embeddings: Protein language model embeddings (per-residue)
        - Stachelhaus code embeddings: Domain-specific code embeddings (per-residue)
        - BigCarp embeddings: Functional domain embeddings (fixed-size)

Methodology:
    - Stratified K-Fold Cross-Validation (default: 5 folds) for balanced evaluation
    - Bootstrap resampling (default: 10 seeds) for confidence interval estimation
    - Multiple metrics: Accuracy, F1-score, and AUC-ROC
    - MLP architecture: (512, 256) hidden layers with early stopping
    - StandardScaler preprocessing for feature normalization

Embeddings Evaluated:
    1. ESM Mean Pool: Mean-pooled ESM-2 embeddings across sequence
    2. Stachelhaus Mean Pool: Mean-pooled Stachelhaus code embeddings
    3. BigCarp: BigCarp domain embeddings (as-is)
    4. Random 256: Random baseline (256-dimensional, fixed seed=42)
    5. BigCarp + ESM: Concatenated BigCarp and ESM mean-pooled embeddings
    6. BigCarp + Stachelhaus: Concatenated BigCarp and Stachelhaus mean-pooled embeddings

Output:
    - Accuracy, F1-score, and AUC-ROC with 95% confidence intervals
    - Comprehensive results tables for all properties and methods
    - Best performing method per property (ranked by AUC-ROC)
    - Two JSON files:
        * summary_results.json: Means, standard deviations, and confidence intervals
        * raw_results.json: All raw bootstrap metric values

Command-Line Arguments:
    --data-path: Path to input pickle file (default: data/processed/adomain_prediction/adomain_training_dataset_full.pkl)
    --output-dir: Output directory for results (default: results/adomain_prediction/bootstrap)
    --n-bootstrap: Number of bootstrap seeds (default: 10)
    --n-folds: Number of CV folds (default: 5)
    --quiet: Suppress progress output

Usage:
    python adomain_properties_prediction.py

    Or with custom parameters:
    python adomain_properties_prediction.py --data-path path/to/data.pkl --n-bootstrap 20 --n-folds 10

    With minimal output:
    python adomain_properties_prediction.py --quiet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
from collections import defaultdict
import argparse
import json
import os

warnings.filterwarnings('ignore')


class AdomainPropertiesPredictor:
    """Main class for A-domain properties prediction with bootstrap sampling.

    This class implements a complete pipeline for predicting A-domain substrate properties
    using various embedding approaches and evaluating them with stratified cross-validation
    and bootstrap confidence intervals.
    """

    def __init__(self, data_path, n_bootstrap=10, n_folds=5):
        """Initialize the A-domain properties predictor.

        Args:
            data_path (str): Path to the pickle file containing the A-domain dataset.
                Expected to have columns: esm2_embeddings, stachel_embeddings,
                bigcarp_embedding, and property columns (is_aromatic, has_heterocycle, etc.)
            n_bootstrap (int, optional): Number of bootstrap seeds for confidence intervals.
                Defaults to 10.
            n_folds (int, optional): Number of folds for stratified cross-validation.
                Defaults to 5.
        """
        self.data_path = data_path
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds

        # Load data
        print(f"Loading data from {data_path}")
        self.df = pd.read_pickle(data_path)
        print(f"Dataset shape: {self.df.shape}")

        # Define properties
        all_properties = ['is_aromatic', 'has_heterocycle', 'high_polarity', 'is_Val', 'is_Gly', 'is_canonical_aa']
        self.properties = [prop for prop in all_properties if prop in self.df.columns]
        print(f"Available properties: {self.properties}")

        # Prepare embeddings
        self.embeddings = self._prepare_embeddings()

        # Define experiments
        self.all_experiments = ['esm_mean', 'stachel_mean', 'bigcarp', 'random_256', 'bigcarp_esm', 'bigcarp_stachel']

    def _prepare_embeddings(self):
        """Prepare all embedding representations from the dataset.

        Creates multiple embedding variants:
        - esm_mean: Mean-pooled ESM2 embeddings
        - stachel_mean: Mean-pooled Stachelhaus code embeddings
        - bigcarp: BigCarp embeddings (as-is)
        - bigcarp_esm: Concatenation of BigCarp and ESM mean
        - bigcarp_stachel: Concatenation of BigCarp and Stachelhaus mean
        - random_256: Random baseline (256-dimensional, seed=42)

        Returns:
            dict: Dictionary mapping embedding names to numpy arrays of shape (n_samples, n_features).
        """
        embeddings = {}

        # ESM mean pool
        esm_mean = np.array([emb.mean(axis=0) for emb in self.df['esm2_embeddings']])
        embeddings['esm_mean'] = esm_mean

        # Stachelhaus mean pool
        stachel_mean = np.array([emb.mean(axis=0) for emb in self.df['stachel_embeddings']])
        embeddings['stachel_mean'] = stachel_mean

        # BigCarp
        bigcarp = np.array([emb for emb in self.df['bigcarp_embedding']])
        embeddings['bigcarp'] = bigcarp

        # Concatenations
        embeddings['bigcarp_esm'] = np.hstack([bigcarp, esm_mean])
        embeddings['bigcarp_stachel'] = np.hstack([bigcarp, stachel_mean])

        # Random control with fixed seed for reproducibility
        np.random.seed(42)
        embeddings['random_256'] = np.random.normal(0, 1, (len(self.df), 256))

        print("Embedding shapes:")
        for name, emb in embeddings.items():
            print(f"  {name}: {emb.shape}")

        return embeddings

    def _get_mlp_config(self):
        """Get MLP classifier configuration.

        Returns:
            dict: Configuration dictionary for sklearn MLPClassifier with:
                - hidden_layer_sizes: (512, 256) - two hidden layers
                - max_iter: 3000 - maximum training iterations
                - alpha: 0.01 - L2 regularization strength
                - early_stopping: True - uses validation set for early stopping
                - validation_fraction: 0.1 - 10% of training data for validation
                - n_iter_no_change: 30 - patience for early stopping
                - random_state: 42 - seed (will be overridden per bootstrap)
        """
        return {
            'hidden_layer_sizes': (512, 256),
            'max_iter': 3000,
            'alpha': 0.01,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 30,
            'random_state': 42
        }

    def _train_model(self, X_train, y_train, X_val, y_val, random_seed=42):
        """Train MLP model with StandardScaler preprocessing.

        Args:
            X_train (np.ndarray): Training features of shape (n_train_samples, n_features).
            y_train (np.ndarray): Training labels of shape (n_train_samples,).
            X_val (np.ndarray): Validation features of shape (n_val_samples, n_features).
            y_val (np.ndarray): Validation labels of shape (n_val_samples,).
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            tuple: (val_preds, val_probs) where:
                - val_preds (np.ndarray): Binary predictions on validation set, shape (n_val_samples,).
                - val_probs (np.ndarray): Predicted probabilities for positive class, shape (n_val_samples,).
        """
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train MLP
        mlp_config = self._get_mlp_config()
        mlp_config['random_state'] = random_seed
        clf = MLPClassifier(**mlp_config)

        try:
            clf.fit(X_train_scaled, y_train)
            val_probs = clf.predict_proba(X_val_scaled)[:, 1]
            val_preds = clf.predict(X_val_scaled)
        except Exception as e:
            # Fallback to random predictions if training fails
            print(f"Warning: MLP training failed: {e}")
            val_probs = np.random.random(len(y_val))
            val_preds = (val_probs > 0.5).astype(int)

        return val_preds, val_probs

    def _evaluate_predictions(self, y_true, y_pred, y_prob):
        """Calculate evaluation metrics for binary classification.

        Args:
            y_true (np.ndarray): True binary labels of shape (n_samples,).
            y_pred (np.ndarray): Predicted binary labels of shape (n_samples,).
            y_prob (np.ndarray): Predicted probabilities for positive class of shape (n_samples,).

        Returns:
            dict: Dictionary with three metrics:
                - accuracy (float): Accuracy score [0, 1].
                - f1 (float): F1 score [0, 1] (zero_division=0).
                - auc (float): AUC-ROC score [0, 1] (0.5 if only one class present).
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        }

    def _calculate_summary_stats(self, values, confidence=0.95):
        """Calculate mean, standard deviation, and confidence intervals.

        Uses the percentile method for confidence interval estimation.

        Args:
            values (list or np.ndarray): List of metric values from bootstrap samples.
            confidence (float, optional): Confidence level for intervals. Defaults to 0.95.

        Returns:
            dict: Dictionary containing:
                - mean (float): Mean of values.
                - std (float): Standard deviation (with ddof=1).
                - ci_lower (float): Lower bound of confidence interval.
                - ci_upper (float): Upper bound of confidence interval.
                - n_samples (int): Number of bootstrap samples.
        """
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)

        # Calculate confidence interval using percentile method
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100

        ci_lower = np.percentile(values, lower_percentile)
        ci_upper = np.percentile(values, upper_percentile)

        return {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': len(values)
        }

    def run_evaluation(self, verbose=True):
        """Run complete evaluation with bootstrap sampling and stratified CV.

        For each property and embedding method:
        1. Run stratified k-fold CV with multiple bootstrap seeds
        2. Collect metrics (accuracy, F1, AUC) across all folds and seeds
        3. Calculate summary statistics with confidence intervals

        Args:
            verbose (bool, optional): If True, print progress and intermediate results.
                Defaults to True.

        Returns:
            tuple: (summary_results, results_all_runs) where:
                - summary_results (dict): Nested dict with structure:
                    {property: {method: {metric: {mean, std, ci_lower, ci_upper, n_samples}}}}
                - results_all_runs (dict): Nested dict with raw bootstrap values:
                    {property: {method: {metric: [list of values]}}}
        """
        bootstrap_seeds = list(range(self.n_bootstrap))
        results_all_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        if verbose:
            print(f"Starting evaluation with {self.n_bootstrap} bootstrap seeds...")
            print(f"Properties: {self.properties}")
            print(f"Methods: {self.all_experiments}")

        for prop in self.properties:
            if verbose:
                print(f"\\n{'='*60}")
                print(f"PROPERTY: {prop}")
                print(f"{'='*60}")

            y = self.df[prop].values

            # Skip if no variation
            if len(np.unique(y)) < 2:
                if verbose:
                    print(f"Skipping {prop} - no variation in labels")
                continue

            for exp_name in self.all_experiments:
                if verbose:
                    print(f"\\n--- {exp_name} ---")

                X_full = self.embeddings[exp_name]

                for seed in bootstrap_seeds:
                    # 5-fold stratified CV with current seed
                    kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
                    fold_metrics = []

                    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_full, y)):
                        X_train, X_val = X_full[train_idx], X_full[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        val_preds, val_probs = self._train_model(
                            X_train, y_train, X_val, y_val, random_seed=seed
                        )

                        fold_metrics.append(self._evaluate_predictions(y_val, val_preds, val_probs))

                    # Store metrics for this bootstrap run
                    if fold_metrics:
                        for metric_name in ['accuracy', 'f1', 'auc']:
                            avg_metric = np.mean([fm[metric_name] for fm in fold_metrics])
                            results_all_runs[prop][exp_name][metric_name].append(avg_metric)

                # Print intermediate results
                if verbose and exp_name in results_all_runs[prop]:
                    acc_mean = np.mean(results_all_runs[prop][exp_name]['accuracy'])
                    f1_mean = np.mean(results_all_runs[prop][exp_name]['f1'])
                    auc_mean = np.mean(results_all_runs[prop][exp_name]['auc'])
                    print(f"  Mean scores: Acc={acc_mean:.3f}, F1={f1_mean:.3f}, AUC={auc_mean:.3f}")

        if verbose:
            print(f"\\n\\nCompleted all evaluations!")

        # Calculate summary statistics
        summary_results = defaultdict(lambda: defaultdict(dict))
        for prop in self.properties:
            for exp_name in self.all_experiments:
                for metric_name in ['accuracy', 'f1', 'auc']:
                    if metric_name in results_all_runs[prop][exp_name]:
                        values = results_all_runs[prop][exp_name][metric_name]
                        summary_results[prop][exp_name][metric_name] = self._calculate_summary_stats(values)

        return summary_results, results_all_runs

    def print_results(self, summary_results):
        """Print comprehensive results with confidence intervals.

        Outputs formatted tables showing:
        1. F1 scores with 95% CIs for all properties and methods
        2. AUC scores with 95% CIs for all properties and methods
        3. Accuracy scores with 95% CIs for all properties and methods
        4. Best performing method per property (ranked by AUC)

        Args:
            summary_results (dict): Nested dictionary from run_evaluation() containing
                statistics for each property/method/metric combination.
        """
        def print_table(metric_name, title):
            print("\\n" + "="*200)
            print(f"{title} - {metric_name.upper()} SCORES WITH 95% CONFIDENCE INTERVALS")
            print("="*200)

            # Header
            header_parts = ["Property".ljust(15)]
            for exp_name in self.all_experiments:
                header_parts.append(f"{exp_name}".ljust(25))
            print(" ".join(header_parts))
            print("-" * 200)

            # Data rows
            for prop in self.properties:
                if prop in summary_results:
                    row_parts = [prop.ljust(15)]
                    for exp_name in self.all_experiments:
                        if exp_name in summary_results[prop] and metric_name in summary_results[prop][exp_name]:
                            stats = summary_results[prop][exp_name][metric_name]
                            mean_val = stats['mean']
                            ci_lower = stats['ci_lower']
                            ci_upper = stats['ci_upper']
                            result_str = f"{mean_val:.3f} [{ci_lower:.3f}-{ci_upper:.3f}]"
                            row_parts.append(result_str.ljust(25))
                        else:
                            row_parts.append("N/A".ljust(25))
                    print(" ".join(row_parts))

        # Print all results tables
        print_table('f1', 'F1 Score')
        print_table('auc', 'AUC Score')
        print_table('accuracy', 'Accuracy')

        # Print analysis
        print("\\n" + "="*120)
        print("BEST PERFORMING METHODS PER PROPERTY (by AUC-ROC):")
        print("-" * 80)

        for prop in self.properties:
            if prop in summary_results:
                best_auc = -1
                best_method = None

                for exp_name in self.all_experiments:
                    if exp_name in summary_results[prop] and 'auc' in summary_results[prop][exp_name]:
                        auc_mean = summary_results[prop][exp_name]['auc']['mean']
                        if auc_mean > best_auc:
                            best_auc = auc_mean
                            best_method = exp_name

                if best_method:
                    stats = summary_results[prop][best_method]['auc']
                    print(f"{prop:20} | {best_method:25} | AUC: {stats['mean']:.3f} [{stats['ci_lower']:.3f}-{stats['ci_upper']:.3f}]")

    def save_results(self, summary_results, raw_results, output_dir):
        """Save results to JSON files.

        Creates two JSON files in the output directory:
        1. summary_results.json - Contains means, stds, and confidence intervals
        2. raw_results.json - Contains all raw bootstrap metric values

        Args:
            summary_results (dict): Summary statistics dictionary from run_evaluation().
            raw_results (dict): Raw bootstrap results dictionary from run_evaluation().
            output_dir (str): Directory path where JSON files will be saved.
                Created if it doesn't exist.
        """
        os.makedirs(output_dir, exist_ok=True)

        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization.

            Args:
                obj: Object to convert (can be numpy types, dict, list, or native types).

            Returns:
                Native Python type equivalent (int, float, list, dict, or unchanged).
            """
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        # Convert and save results
        summary_json = convert_to_native(summary_results)
        raw_json = convert_to_native(raw_results)

        with open(os.path.join(output_dir, 'summary_results.json'), 'w') as f:
            json.dump(summary_json, f, indent=2)

        with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
            json.dump(raw_json, f, indent=2)

        print(f"Results saved to {output_dir}/")


def main():
    """Main entry point for A-domain properties prediction pipeline.

    Parses command-line arguments, initializes the predictor, runs evaluation with
    bootstrap sampling, prints results, and saves outputs to JSON files.

    Command-line Arguments:
        --data-path (str): Path to input pickle file with A-domain dataset.
        --output-dir (str): Directory for saving results JSON files.
        --n-bootstrap (int): Number of bootstrap seeds for confidence intervals.
        --n-folds (int): Number of cross-validation folds.
        --quiet (flag): Suppress progress output if set.
    """
    parser = argparse.ArgumentParser(description='A-domain properties prediction with bootstrap confidence intervals')
    parser.add_argument('--data-path', type=str,
                       default='data/processed/adomain_prediction/adomain_training_dataset_full.pkl',
                       help='Path to the input dataset (pickle file)')
    parser.add_argument('--output-dir', type=str, default='results/adomain_prediction/bootstrap',
                       help='Output directory for results')
    parser.add_argument('--n-bootstrap', type=int, default=10,
                       help='Number of bootstrap seeds (default: 10)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bootstrap seeds: {args.n_bootstrap}")
    print(f"CV folds: {args.n_folds}")
    print(f"MLP architecture: (512, 256)")

    # Initialize predictor
    predictor = AdomainPropertiesPredictor(
        data_path=args.data_path,
        n_bootstrap=args.n_bootstrap,
        n_folds=args.n_folds
    )

    # Run evaluation
    summary_results, raw_results = predictor.run_evaluation(verbose=not args.quiet)

    # Print results
    if not args.quiet:
        predictor.print_results(summary_results)

    # Save results
    predictor.save_results(summary_results, raw_results, args.output_dir)

    print("\\n" + "="*120)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*120)
    print(f"• Evaluated {len(predictor.all_experiments)} methods on {len(predictor.properties)} properties")
    print(f"• Used {args.n_bootstrap} bootstrap seeds with {args.n_folds}-fold CV each")
    print(f"• Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()