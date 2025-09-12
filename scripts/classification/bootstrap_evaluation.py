#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap Evaluation for MIBiG Classification Models
===================================================
• Runs multiple random seeds to assess performance stability
• Calculates mean ± std for all metrics, focusing on macro AUC
• Provides statistical comparison with confidence intervals
• Can run on both MIBiG 1.0 and MIBiG 3.0 datasets
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import subprocess
import pickle
from pathlib import Path
import json
from scipy import stats

def run_single_evaluation(script_path, artifacts_dir, outdir, seed, dataset_version):
    """Run a single evaluation with given seed."""
    print(f"🎲 Running evaluation with seed {seed}...")
    
    # Create seed-specific output directory
    seed_outdir = f"{outdir}/seed_{seed}"
    
    # Run the evaluation script
    cmd = [
        sys.executable, script_path,
        "--artifacts_dir", artifacts_dir,
        "--outdir", seed_outdir,
        "--seed", str(seed)
    ]
    
    try:
        # Run with live output but capture for error handling
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✅ Seed {seed} completed successfully")
        
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
        print(f"❌ {error_msg}")
        return None, error_msg

def extract_metrics(results):
    """Extract key metrics from results for all models."""
    metrics_data = []
    
    for result in results:
        if result is not None and 'aggregate_metrics' in result:
            model_name = result['model_name']
            agg_metrics = result['aggregate_metrics']
            
            # Extract key metrics
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

def bootstrap_evaluation(script_path, artifacts_dir, outdir, dataset_version, n_seeds=10, base_seed=42):
    """Run bootstrap evaluation with multiple random seeds."""
    
    print(f"🔬 Bootstrap Evaluation - {dataset_version}")
    print(f"📊 Running {n_seeds} evaluations with different random seeds")
    print(f"📁 Script: {script_path}")
    print(f"📁 Artifacts: {artifacts_dir}")
    print(f"📁 Output: {outdir}")
    
    # Generate random seeds
    np.random.seed(base_seed)
    seeds = np.random.randint(1, 10000, n_seeds)
    print(f"🎲 Using seeds: {seeds}")
    
    # Store all results
    all_results = []
    successful_seeds = []
    failed_seeds = []
    
    # Run evaluations
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"🔄 Evaluation {i+1}/{n_seeds} - Seed {seed}")
        print(f"{'='*60}")
        
        # Progress bar
        progress = int((i / n_seeds) * 40)
        bar = '█' * progress + '░' * (40 - progress)
        print(f"📊 Progress: [{bar}] {i}/{n_seeds} ({i/n_seeds*100:.1f}%)")
        print()
        
        results, error = run_single_evaluation(script_path, artifacts_dir, outdir, seed, dataset_version)
        
        if results is not None:
            metrics = extract_metrics(results)
            if metrics:
                all_results.append({
                    'seed': seed,
                    'metrics': metrics
                })
                successful_seeds.append(seed)
            else:
                print(f"⚠️  No valid metrics extracted for seed {seed}")
                failed_seeds.append(seed)
        else:
            print(f"❌ Failed to get results for seed {seed}: {error}")
            failed_seeds.append(seed)
    
    # Final progress bar
    final_bar = '█' * 40
    print(f"\n📊 Final Progress: [{final_bar}] {n_seeds}/{n_seeds} (100.0%)")
    
    print(f"\n{'='*60}")
    print(f"📊 Bootstrap Evaluation Summary")
    print(f"{'='*60}")
    print(f"✅ Successful runs: {len(successful_seeds)}/{n_seeds}")
    print(f"❌ Failed runs: {len(failed_seeds)}")
    if failed_seeds:
        print(f"   Failed seeds: {failed_seeds}")
    
    if len(successful_seeds) < 3:
        print("❌ Too few successful runs for statistical analysis!")
        return None
    
    return all_results, successful_seeds, failed_seeds

def calculate_bootstrap_statistics(all_results):
    """Calculate mean, std, and confidence intervals for each model and metric."""
    
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
            for metric in ['macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy', 'micro_f1', 'weighted_macro_f1']:
                value = metric_data.get(metric, np.nan)
                model_results[model_name][metric].append(value)
    
    # Calculate statistics
    statistics = {}
    
    for model_name, data in model_results.items():
        stats_data = {'model_name': model_name, 'n_runs': len(data['seeds'])}
        
        for metric in ['macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy', 'micro_f1', 'weighted_macro_f1']:
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

def create_bootstrap_summary_table(statistics, focus_metric='macro_auc'):
    """Create a summary table focused on the primary metric with confidence intervals."""
    
    print(f"\n{'='*100}")
    print(f"🎯 BOOTSTRAP EVALUATION RESULTS - {focus_metric.upper()} FOCUS")
    print(f"{'='*100}")
    
    # Prepare data for table
    table_data = []
    for model_name, stats in statistics.items():
        n_runs = stats['n_runs']
        
        # Primary metric (macro_auc)
        mean_val = stats.get(f'{focus_metric}_mean', np.nan)
        std_val = stats.get(f'{focus_metric}_std', np.nan)
        ci_lower = stats.get(f'{focus_metric}_ci_lower', np.nan)
        ci_upper = stats.get(f'{focus_metric}_ci_upper', np.nan)
        
        # Additional metrics for context
        macro_f1_mean = stats.get('macro_f1_mean', np.nan)
        macro_f1_std = stats.get('macro_f1_std', np.nan)
        
        exact_acc_mean = stats.get('exact_match_accuracy_mean', np.nan)
        exact_acc_std = stats.get('exact_match_accuracy_std', np.nan)
        
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
    table_data.sort(key=lambda x: x[f'{focus_metric.replace("_", " ").title()} Mean'] if not np.isnan(x[f'{focus_metric.replace("_", " ").title()} Mean']) else -1, reverse=True)
    
    # Create DataFrame for nice formatting
    df = pd.DataFrame(table_data)
    
    # Display table
    print("\n📊 Performance Summary:")
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
        best_score = df.iloc[0][f'{focus_metric.replace("_", " ").title()} Mean']
        print(f"\n🏆 Best Model (by {focus_metric.replace('_', ' ').title()}): {best_model} ({best_score})")
    
    return df

def save_bootstrap_results(statistics, all_results, outdir, dataset_version):
    """Save detailed bootstrap results."""
    
    # Create output directory
    bootstrap_outdir = f"{outdir}/bootstrap_analysis"
    Path(bootstrap_outdir).mkdir(parents=True, exist_ok=True)
    
    # Save detailed statistics
    stats_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_statistics.json"
    with open(stats_file, 'w') as f:
        # Convert numpy types to JSON serializable
        json_stats = {}
        for model, stats in statistics.items():
            json_stats[model] = {}
            for key, value in stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_stats[model][key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_stats[model][key] = value.tolist()
                else:
                    json_stats[model][key] = value
        
        json.dump(json_stats, f, indent=2)
    
    print(f"💾 Detailed statistics saved to: {stats_file}")
    
    # Save raw results
    raw_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_raw_results.pkl"
    with open(raw_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"💾 Raw results saved to: {raw_file}")
    
    # Save summary CSV
    summary_data = []
    for model_name, stats in statistics.items():
        row = {'model_name': model_name}
        for metric in ['macro_f1', 'macro_auc', 'weighted_auc', 'exact_match_accuracy']:
            row[f'{metric}_mean'] = stats.get(f'{metric}_mean', np.nan)
            row[f'{metric}_std'] = stats.get(f'{metric}_std', np.nan)
            row[f'{metric}_ci_lower'] = stats.get(f'{metric}_ci_lower', np.nan)
            row[f'{metric}_ci_upper'] = stats.get(f'{metric}_ci_upper', np.nan)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"💾 Summary CSV saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Bootstrap Evaluation for MIBiG Classification")
    parser.add_argument("--dataset", choices=['mibig1', 'mibig3'], required=True,
                        help="Dataset version to evaluate")
    parser.add_argument("--artifacts_dir", 
                        help="Directory containing embedding files (auto-detected if not provided)")
    parser.add_argument("--outdir", 
                        help="Output directory (auto-generated if not provided)")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random seeds to run (default: 10)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generating random seeds (default: 42)")
    parser.add_argument("--focus_metric", default="macro_auc",
                        choices=['macro_auc', 'macro_f1', 'weighted_auc', 'exact_match_accuracy'],
                        help="Primary metric to focus on (default: macro_auc)")
    
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
        print(f"❌ Evaluation script not found: {script_path}")
        return 1
    
    print(f"🚀 Bootstrap Evaluation for {args.dataset.upper()}")
    print(f"📊 Focus metric: {args.focus_metric}")
    print(f"🎲 Number of seeds: {args.n_seeds}")
    print(f"🌱 Base seed: {args.base_seed}")
    
    # Run bootstrap evaluation
    results = bootstrap_evaluation(
        script_path, args.artifacts_dir, args.outdir, 
        args.dataset.upper(), args.n_seeds, args.base_seed
    )
    
    if results is None:
        return 1
    
    all_results, successful_seeds, failed_seeds = results
    
    # Calculate statistics
    print(f"\n🧮 Calculating bootstrap statistics...")
    statistics = calculate_bootstrap_statistics(all_results)
    
    # Create summary table
    summary_df = create_bootstrap_summary_table(statistics, args.focus_metric)
    
    # Save results
    save_bootstrap_results(statistics, all_results, args.outdir, args.dataset.upper())
    
    print(f"\n✅ Bootstrap evaluation completed!")
    print(f"📁 Results saved in: {args.outdir}/bootstrap_analysis/")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"🎲 Successful evaluations: {len(successful_seeds)}/{args.n_seeds}")
    print(f"📈 Primary metric ({args.focus_metric}): Focus on mean ± std and 95% CI")
    print(f"🏆 Most robust comparison: Look for non-overlapping confidence intervals")
    print("💡 Models with small std and tight CI are more reliable")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())