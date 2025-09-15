#!/usr/bin/env python3
"""
A-Domain Substrate Properties Prediction with Bootstrap Confidence Intervals

This script combines all substrate properties prediction with robust statistical evaluation:
- All properties: is_aromatic, has_heterocycle, high_polarity, is_Val, is_Gly, is_canonical_aa
- Multiple embedding approaches: ESM, Stachelhaus, BigCarp, and combinations
- Stratified 5-fold cross-validation
- Bootstrap sampling with configurable random seeds for confidence intervals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')


class MLP(nn.Module):
    """Simple MLP model for binary classification"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x).squeeze()


class MultiFairConcatHead(nn.Module):
    """Fair concatenation model with projection layers"""
    def __init__(self, in_dims, proj_dim=256, hidden_dim=128):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, proj_dim) for d in in_dims])
        self.norm = nn.LayerNorm(proj_dim * len(in_dims))
        self.head = nn.Sequential(
            nn.Linear(proj_dim * len(in_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, *xs):
        projections = [proj(x) for proj, x in zip(self.projs, xs)]
        concatenated = torch.cat(projections, dim=1)
        normalized = self.norm(concatenated)
        return self.head(normalized).squeeze()


class AdomainPropertiesPredictor:
    """Main class for A-domain properties prediction with bootstrap sampling"""

    def __init__(self, data_path, device=None, n_bootstrap=10, n_folds=5, epochs=50):
        self.data_path = data_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds
        self.epochs = epochs

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
        self.single_embeddings = ['esm_mean', 'stachel_mean', 'bigcarp', 'random_256', 'random_1280']
        self.naive_concatenations = ['naive_bigcarp_esm', 'naive_bigcarp_stachel']
        self.fair_concatenations = {
            'fair_bigcarp_esm': ['bigcarp', 'esm_mean'],
            'fair_bigcarp_stachel': ['bigcarp', 'stachel_mean']
        }
        self.all_experiments = self.single_embeddings + self.naive_concatenations + list(self.fair_concatenations.keys())

    def _prepare_embeddings(self):
        """Prepare all embedding representations"""
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

        # Naive concatenations
        embeddings['naive_bigcarp_esm'] = np.hstack([bigcarp, esm_mean])
        embeddings['naive_bigcarp_stachel'] = np.hstack([bigcarp, stachel_mean])

        # Random controls with fixed seeds for reproducibility
        np.random.seed(42)
        embeddings['random_256'] = np.random.normal(0, 1, (len(self.df), 256))
        embeddings['random_1280'] = np.random.normal(0, 1, (len(self.df), 1280))

        # Store parts for fair concatenation
        embeddings['_parts'] = {
            'bigcarp': bigcarp,
            'esm_mean': esm_mean,
            'stachel_mean': stachel_mean
        }

        print("Embedding shapes:")
        for name, emb in embeddings.items():
            if name != '_parts':
                print(f"  {name}: {emb.shape}")

        return embeddings

    def _zscore_l2norm(self, train_X, val_X, eps=1e-8):
        """Z-score normalization using train stats, then L2 normalization"""
        mean = train_X.mean(axis=0, keepdims=True)
        std = train_X.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        train_z = (train_X - mean) / std
        val_z = (val_X - mean) / std

        def l2_normalize(x):
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            norm = np.where(norm < eps, 1.0, norm)
            return x / norm

        return l2_normalize(train_z), l2_normalize(val_z)

    def _train_single_model(self, X_train, y_train, X_val, y_val, input_dim, random_seed=42):
        """Train single embedding model with StandardScaler preprocessing"""
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = MLP(input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled).to(self.device),
            torch.FloatTensor(y_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val_scaled).to(self.device))
            val_probs = val_outputs.cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)

        return val_preds, val_probs

    def _train_fair_concat_model(self, branch_trains, y_train, branch_vals, y_val, proj_dim=256, random_seed=42):
        """Train fair concatenation model with per-branch normalization"""
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        in_dims = [b.shape[1] for b in branch_trains]
        model = MultiFairConcatHead(in_dims, proj_dim=proj_dim, hidden_dim=128).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Build dataset with multiple inputs
        tensors_train = [torch.FloatTensor(b).to(self.device) for b in branch_trains]
        t_y_train = torch.FloatTensor(y_train).to(self.device)
        train_dataset = TensorDataset(*tensors_train, t_y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            for *batch_xs, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(*batch_xs)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            tensors_val = [torch.FloatTensor(b).to(self.device) for b in branch_vals]
            val_probs = model(*tensors_val).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)

        return val_preds, val_probs

    def _evaluate_predictions(self, y_true, y_pred, y_prob):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        }

    def _calculate_summary_stats(self, values, confidence=0.95):
        """Calculate mean, std, and confidence intervals"""
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
        """Run complete evaluation with bootstrap sampling"""
        bootstrap_seeds = list(range(self.n_bootstrap))
        results_all_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        if verbose:
            print(f"Starting evaluation with {self.n_bootstrap} bootstrap seeds...")
            print(f"Properties: {self.properties}")
            print(f"Methods: {self.all_experiments}")
            total_runs = len(self.properties) * len(self.all_experiments) * len(bootstrap_seeds)
            print(f"Total evaluations: {total_runs}")

        current_run = 0

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

                for seed in bootstrap_seeds:
                    current_run += 1
                    if verbose and current_run % 50 == 0:
                        print(f"Progress: {current_run}/{total_runs} ({100*current_run/total_runs:.1f}%)")

                    # 5-fold stratified CV with current seed
                    kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
                    fold_metrics = []

                    if exp_name in self.single_embeddings or exp_name in self.naive_concatenations:
                        # Single tensor experiments
                        X_full = self.embeddings[exp_name]

                        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_full, y)):
                            X_train, X_val = X_full[train_idx], X_full[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]

                            val_preds, val_probs = self._train_single_model(
                                X_train, y_train, X_val, y_val,
                                X_full.shape[1], random_seed=seed
                            )

                            fold_metrics.append(self._evaluate_predictions(y_val, val_preds, val_probs))

                    else:
                        # Fair concatenation experiments
                        parts = self.fair_concatenations[exp_name]
                        X_parts = [self.embeddings['_parts'][p] for p in parts]

                        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_parts[0], y)):
                            y_train, y_val = y[train_idx], y[val_idx]

                            # Per-branch normalization using train stats only
                            branch_trains, branch_vals = [], []
                            for part_data in X_parts:
                                train_part, val_part = part_data[train_idx], part_data[val_idx]
                                train_norm, val_norm = self._zscore_l2norm(train_part, val_part)
                                branch_trains.append(train_norm)
                                branch_vals.append(val_norm)

                            val_preds, val_probs = self._train_fair_concat_model(
                                branch_trains, y_train, branch_vals, y_val,
                                proj_dim=256, random_seed=seed
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
                    print(f"  Interim: Acc={acc_mean:.3f}, F1={f1_mean:.3f}, AUC={auc_mean:.3f}")

        if verbose:
            print(f"\\n\\nCompleted all {current_run} evaluations!")

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
        """Print comprehensive results with confidence intervals"""
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
        """Save results to JSON files"""
        os.makedirs(output_dir, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Save summary results
        summary_json = {}
        for prop in summary_results:
            summary_json[prop] = {}
            for exp_name in summary_results[prop]:
                summary_json[prop][exp_name] = {}
                for metric in summary_results[prop][exp_name]:
                    summary_json[prop][exp_name][metric] = {
                        k: convert_numpy(v) for k, v in summary_results[prop][exp_name][metric].items()
                    }

        with open(os.path.join(output_dir, 'summary_results.json'), 'w') as f:
            json.dump(summary_json, f, indent=2)

        # Save raw results
        raw_json = {}
        for prop in raw_results:
            raw_json[prop] = {}
            for exp_name in raw_results[prop]:
                raw_json[prop][exp_name] = {}
                for metric in raw_results[prop][exp_name]:
                    raw_json[prop][exp_name][metric] = [convert_numpy(v) for v in raw_results[prop][exp_name][metric]]

        with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
            json.dump(raw_json, f, indent=2)

        print(f"Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='A-domain properties prediction with bootstrap confidence intervals')
    parser.add_argument('--data-path', type=str,
                       default='/home/u5bb/han00.u5bb/workspace/cgrep/data/processed/adomain_prediction/adomain_training_dataset_full.pkl',
                       help='Path to the input dataset (pickle file)')
    parser.add_argument('--output-dir', type=str, default='results/adomain_prediction/bootstrap',
                       help='Output directory for results')
    parser.add_argument('--n-bootstrap', type=int, default=10,
                       help='Number of bootstrap seeds (default: 10)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bootstrap seeds: {args.n_bootstrap}")
    print(f"CV folds: {args.n_folds}")
    print(f"Training epochs: {args.epochs}")

    # Initialize predictor
    predictor = AdomainPropertiesPredictor(
        data_path=args.data_path,
        device=device,
        n_bootstrap=args.n_bootstrap,
        n_folds=args.n_folds,
        epochs=args.epochs
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