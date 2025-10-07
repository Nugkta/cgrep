#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap Evaluation for MIBiG Classification Models
====================================================

Bootstrap evaluation with multiple random seeds to assess performance stability and
statistical significance. Calculates mean ± std and 95% confidence intervals.

Usage:
    $ python scripts/classification/bootstrap_evaluation.py --dataset mibig3 [--n_seeds N]
    $ python scripts/classification/bootstrap_evaluation.py --dataset mibig3 --n_seeds 10 --focus_metric macro_auc
    $ sbatch scripts/classification/run_mibig3_bootstrap.sh  # For SLURM cluster

Arguments:
    --dataset: Dataset version (mibig1 or mibig3)
    --n_seeds: Number of random seeds (default: 10)
    --focus_metric: Primary metric for comparison (default: macro_auc)
    --base_seed: Base seed for reproducibility (default: 42)

Outputs:
    bootstrap_analysis/
        - {dataset}_bootstrap_summary.csv: Summary with mean, std, 95% CI
        - {dataset}_bootstrap_statistics.json: Detailed statistics
        - {dataset}_bootstrap_raw_results.pkl: Raw results from all seeds

Author: [Your Name]
Date: 2025
"""

import os
import sys
import argparse
import json
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# Single Evaluation Execution
# ==============================================================================

def run_single_evaluation(
    script_path: str,
    artifacts_dir: str,
    outdir: str,
    seed: int,
    dataset_version: str
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Run a single evaluation with the specified random seed.

    Args:
        script_path: Path to evaluation script
        artifacts_dir: Directory containing embedding artifacts
        outdir: Output directory for results
        seed: Random seed for this run
        dataset_version: Dataset version identifier

    Returns:
        Tuple of (results list, error message if failed)
    """
    print(f"Running evaluation with seed {seed}...")

    # Create seed-specific output directory
    seed_outdir = f"{outdir}/seed_{seed}"

    # Build command
    cmd = [
        sys.executable, script_path,
        "--artifacts_dir", artifacts_dir,
        "--outdir", seed_outdir,
        "--seed", str(seed)
    ]

    try:
        # Run evaluation
        result = subprocess.run(cmd, check=True, text=True)
        print(f"Seed {seed} completed successfully")

        # Load results
        results_file = f"{seed_outdir}/complete_results.pkl"
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            return results, None
        else:
            return None, f"Results file not found: {results_file}"

    except subprocess.CalledProcessError as e:
        error_msg = f"Seed {seed} failed with exit code {e.returncode}"
        print(f"{error_msg}")
        return None, error_msg


# ==============================================================================
# Metrics Extraction
# ==============================================================================

def extract_metrics(results: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract key metrics from evaluation results.

    Args:
        results: List of model evaluation results

    Returns:
        List of metric dictionaries for each model
    """
    metrics_data = []

    for result in results:
        if result is not None and 'aggregate_metrics' in result:
            model_name = result['model_name']
            agg_metrics = result['aggregate_metrics']

            metrics_data.append({
                'model_name': model_name,
                'macro_f1': agg_metrics.get('macro_f1', np.nan),
                'macro_auc': agg_metrics.get('macro_auc', np.nan),
                'weighted_auc': agg_metrics.get('weighted_auc', np.nan),
                'exact_match_accuracy': agg_metrics.get('exact_match_accuracy', np.nan),
                'micro_f1': agg_metrics.get('micro_f1', np.nan),
                'weighted_macro_f1': agg_metrics.get('weighted_macro_f1', np.nan)
            })

    return metrics_data


# ==============================================================================
# Bootstrap Evaluation Pipeline
# ==============================================================================

def bootstrap_evaluation(
    script_path: str,
    artifacts_dir: str,
    outdir: str,
    dataset_version: str,
    n_seeds: int = 10,
    base_seed: int = 42
) -> Optional[Tuple[List[Dict], List[int], List[int]]]:
    """
    Run bootstrap evaluation with multiple random seeds.

    Args:
        script_path: Path to evaluation script
        artifacts_dir: Directory containing embedding artifacts
        outdir: Output directory for results
        dataset_version: Dataset version identifier
        n_seeds: Number of random seeds to evaluate
        base_seed: Base seed for generating random seeds

    Returns:
        Tuple of (all_results, successful_seeds, failed_seeds) or None if too few successful
    """
    print(f"\nBootstrap Evaluation - {dataset_version}")
    print(f"Running {n_seeds} evaluations with different random seeds")
    print(f"Script: {script_path}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Output: {outdir}")

    # Generate random seeds
    np.random.seed(base_seed)
    seeds = np.random.randint(1, 10000, n_seeds)
    print(f"Using seeds: {seeds}")

    # Storage for results
    all_results = []
    successful_seeds = []
    failed_seeds = []

    # Run evaluations
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Evaluation {i+1}/{n_seeds} - Seed {seed}")
        print(f"{'='*60}")

        # Progress bar
        progress = int((i / n_seeds) * 40)
        bar = '█' * progress + '░' * (40 - progress)
        print(f"Progress: [{bar}] {i}/{n_seeds} ({i/n_seeds*100:.1f}%)")
        print()

        results, error = run_single_evaluation(
            script_path, artifacts_dir, outdir, seed, dataset_version
        )

        if results is not None:
            metrics = extract_metrics(results)
            if metrics:
                all_results.append({
                    'seed': seed,
                    'metrics': metrics
                })
                successful_seeds.append(seed)
            else:
                print(f"Warning: No valid metrics extracted for seed {seed}")
                failed_seeds.append(seed)
        else:
            print(f"Failed to get results for seed {seed}: {error}")
            failed_seeds.append(seed)

    # Final progress bar
    final_bar = '█' * 40
    print(f"\nFinal Progress: [{final_bar}] {n_seeds}/{n_seeds} (100.0%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Bootstrap Evaluation Summary")
    print(f"{'='*60}")
    print(f"Successful runs: {len(successful_seeds)}/{n_seeds}")
    print(f"Failed runs: {len(failed_seeds)}")
    if failed_seeds:
        print(f"   Failed seeds: {failed_seeds}")

    if len(successful_seeds) < 3:
        print("Too few successful runs for statistical analysis!")
        return None

    return all_results, successful_seeds, failed_seeds


# ==============================================================================
# Statistical Analysis
# ==============================================================================

def calculate_bootstrap_statistics(all_results: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate mean, std, and 95% confidence intervals for each model and metric.

    Args:
        all_results: List of all evaluation results across seeds

    Returns:
        Dictionary mapping model names to their statistics
    """
    # Organize results by model
    model_results = {}

    for run_result in all_results:
        seed = run_result['seed']
        metrics = run_result['metrics']

        for metric_data in metrics:
            model_name = metric_data['model_name']

            if model_name not in model_results:
                model_results[model_name] = {
                    'seeds': [],
                    'macro_f1': [],
                    'macro_auc': [],
                    'weighted_auc': [],
                    'exact_match_accuracy': [],
                    'micro_f1': [],
                    'weighted_macro_f1': []
                }

            model_results[model_name]['seeds'].append(seed)
            for metric in ['macro_f1', 'macro_auc', 'weighted_auc',
                          'exact_match_accuracy', 'micro_f1', 'weighted_macro_f1']:
                value = metric_data.get(metric, np.nan)
                model_results[model_name][metric].append(value)

    # Calculate statistics
    statistics = {}

    for model_name, data in model_results.items():
        stats_data = {
            'model_name': model_name,
            'n_runs': len(data['seeds'])
        }

        for metric in ['macro_f1', 'macro_auc', 'weighted_auc',
                      'exact_match_accuracy', 'micro_f1', 'weighted_macro_f1']:
            values = np.array(data[metric])
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0

                # 95% confidence interval
                if len(valid_values) > 2:
                    sem = stats.sem(valid_values)
                    ci = stats.t.interval(0.95, len(valid_values)-1, loc=mean_val, scale=sem)
                else:
                    ci = (mean_val, mean_val)

                stats_data[f'{metric}_mean'] = mean_val
                stats_data[f'{metric}_std'] = std_val
                stats_data[f'{metric}_ci_lower'] = ci[0]
                stats_data[f'{metric}_ci_upper'] = ci[1]
                stats_data[f'{metric}_values'] = valid_values.tolist()
            else:
                for suffix in ['_mean', '_std', '_ci_lower', '_ci_upper']:
                    stats_data[f'{metric}{suffix}'] = np.nan
                stats_data[f'{metric}_values'] = []

        statistics[model_name] = stats_data

    return statistics


# ==============================================================================
# Results Presentation
# ==============================================================================

def create_bootstrap_summary_table(
    statistics: Dict[str, Dict],
    focus_metric: str = 'macro_auc'
) -> pd.DataFrame:
    """
    Create summary table focused on primary metric with confidence intervals.

    Args:
        statistics: Dictionary of model statistics
        focus_metric: Primary metric to focus on

    Returns:
        Summary DataFrame
    """
    print(f"\n{'='*100}")
    print(f"BOOTSTRAP EVALUATION RESULTS - {focus_metric.upper()} FOCUS")
    print(f"{'='*100}")

    # Prepare data for table
    table_data = []
    for model_name, stats_dict in statistics.items():
        n_runs = stats_dict['n_runs']

        # Primary metric
        mean_val = stats_dict.get(f'{focus_metric}_mean', np.nan)
        std_val = stats_dict.get(f'{focus_metric}_std', np.nan)
        ci_lower = stats_dict.get(f'{focus_metric}_ci_lower', np.nan)
        ci_upper = stats_dict.get(f'{focus_metric}_ci_upper', np.nan)

        # Additional metrics for context
        macro_f1_mean = stats_dict.get('macro_f1_mean', np.nan)
        macro_f1_std = stats_dict.get('macro_f1_std', np.nan)
        exact_acc_mean = stats_dict.get('exact_match_accuracy_mean', np.nan)
        exact_acc_std = stats_dict.get('exact_match_accuracy_std', np.nan)

        table_data.append({
            'Model': model_name,
            'N_Runs': n_runs,
            f'{focus_metric.replace("_", " ").title()} Mean': mean_val,
            f'{focus_metric.replace("_", " ").title()} Std': std_val,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Macro F1 Mean': macro_f1_mean,
            'Macro F1 Std': macro_f1_std,
            'Exact Acc Mean': exact_acc_mean,
            'Exact Acc Std': exact_acc_std
        })

    # Sort by primary metric mean (descending)
    metric_col = f'{focus_metric.replace("_", " ").title()} Mean'
    table_data.sort(
        key=lambda x: x[metric_col] if not np.isnan(x[metric_col]) else -1,
        reverse=True
    )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Display table
    print("\nPerformance Summary:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    # Format numbers for display
    numeric_cols = [col for col in df.columns if col not in ['Model', 'N_Runs']]
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else 'N/A')

    print(df.to_string(index=False))

    # Highlight best performing model
    if not df.empty:
        best_model = df.iloc[0]['Model']
        best_score = df.iloc[0][metric_col]
        print(f"\nBest Model (by {focus_metric.replace('_', ' ').title()}): {best_model} ({best_score})")

    return df


# ==============================================================================
# Results Export
# ==============================================================================

def save_bootstrap_results(
    statistics: Dict[str, Dict],
    all_results: List[Dict],
    outdir: str,
    dataset_version: str
) -> None:
    """
    Save detailed bootstrap results to files.

    Args:
        statistics: Dictionary of model statistics
        all_results: List of all evaluation results
        outdir: Output directory
        dataset_version: Dataset version identifier
    """
    # Create output directory
    bootstrap_outdir = f"{outdir}/bootstrap_analysis"
    Path(bootstrap_outdir).mkdir(parents=True, exist_ok=True)

    # Save detailed statistics as JSON
    stats_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_statistics.json"
    with open(stats_file, 'w') as f:
        # Convert numpy types to JSON serializable
        json_stats = {}
        for model, stats_dict in statistics.items():
            json_stats[model] = {}
            for key, value in stats_dict.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_stats[model][key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_stats[model][key] = value.tolist()
                else:
                    json_stats[model][key] = value

        json.dump(json_stats, f, indent=2)

    print(f"Detailed statistics saved to: {stats_file}")

    # Save raw results
    raw_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_raw_results.pkl"
    with open(raw_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Raw results saved to: {raw_file}")

    # Save summary CSV
    summary_data = []
    for model_name, stats_dict in statistics.items():
        row = {'model_name': model_name}
        for metric in ['macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy']:
            row[f'{metric}_mean'] = stats_dict.get(f'{metric}_mean', np.nan)
            row[f'{metric}_std'] = stats_dict.get(f'{metric}_std', np.nan)
            row[f'{metric}_ci_lower'] = stats_dict.get(f'{metric}_ci_lower', np.nan)
            row[f'{metric}_ci_upper'] = stats_dict.get(f'{metric}_ci_upper', np.nan)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"Summary CSV saved to: {summary_file}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    """Main bootstrap evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Evaluation for MIBiG Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", choices=['mibig1', 'mibig3'], required=True,
                        help="Dataset version to evaluate")
    parser.add_argument("--artifacts_dir",
                        help="Directory containing embedding files (auto-detected if not provided)")
    parser.add_argument("--outdir",
                        help="Output directory (auto-generated if not provided)")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random seeds to run")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generating random seeds")
    parser.add_argument("--focus_metric", default="macro_auc",
                        choices=['macro_auc', 'macro_f1', 'weighted_auc', 'exact_match_accuracy'],
                        help="Primary metric to focus on")

    args = parser.parse_args()

    # Auto-detect paths if not provided
    if args.artifacts_dir is None:
        args.artifacts_dir = f"artifacts/classification/{args.dataset}"

    if args.outdir is None:
        args.outdir = f"results/{args.dataset}_bootstrap_evaluation"

    # Determine script path
    script_name = f"{args.dataset}_stratified_evaluation.py"
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.exists(script_path):
        print(f"Evaluation script not found: {script_path}")
        return 1

    print(f"Bootstrap Evaluation for {args.dataset.upper()}")
    print(f"Focus metric: {args.focus_metric}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Base seed: {args.base_seed}")

    # Run bootstrap evaluation
    results = bootstrap_evaluation(
        script_path, args.artifacts_dir, args.outdir,
        args.dataset.upper(), args.n_seeds, args.base_seed
    )

    if results is None:
        return 1

    all_results, successful_seeds, failed_seeds = results

    # Calculate statistics
    print(f"\nCalculating bootstrap statistics...")
    statistics = calculate_bootstrap_statistics(all_results)

    # Create summary table
    summary_df = create_bootstrap_summary_table(statistics, args.focus_metric)

    # Save results
    save_bootstrap_results(statistics, all_results, args.outdir, args.dataset.upper())

    print(f"\nBootstrap evaluation completed!")
    print(f"Results saved in: {args.outdir}/bootstrap_analysis/")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Successful evaluations: {len(successful_seeds)}/{args.n_seeds}")
    print(f"Primary metric ({args.focus_metric}): Focus on mean ± std and 95% CI")
    print(f"Most robust comparison: Look for non-overlapping confidence intervals")
    print("Models with small std and tight CI are more reliable")

    return 0


if __name__ == "__main__":
    sys.exit(main())
