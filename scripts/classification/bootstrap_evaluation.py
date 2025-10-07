#!/usr/bin/env python
"""
Bootstrap Evaluation for MIBiG Classification Models.

This script performs bootstrap evaluation of machine learning models for biosynthetic gene 
cluster (BGC) classification using the MIBiG database. It runs multiple evaluations with 
different random seeds to assess model performance variability and statistical significance.

The bootstrap evaluation provides:
- Mean performance metrics across multiple random seeds
- Standard deviations to measure performance variability
- 95% confidence intervals for statistical inference
- Comprehensive results export in multiple formats

Typical usage:
    python bootstrap_evaluation.py --dataset mibig3 --n_seeds 10 --focus_metric macro_auc


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
    Execute a single model evaluation with a specific random seed.

    This function runs the evaluation script for all models using a given random seed.
    It handles subprocess execution, result loading, and error management for individual
    evaluation runs within the bootstrap procedure.

    Args:
        script_path (str): Absolute path to the evaluation script to execute 
            (e.g., 'mibig3_stratified_evaluation.py')
        artifacts_dir (str): Directory path containing model artifacts and embeddings
            required for evaluation
        outdir (str): Base output directory where seed-specific results will be saved
        seed (int): Random seed value for reproducible evaluation (1-9999)
        dataset_version (str): Dataset identifier for naming and organization 
            (e.g., 'MIBIG3')

    Returns:
        Tuple[Optional[List[Dict]], Optional[str]]: A tuple containing:
            - results (List[Dict] or None): List of model evaluation results if successful.
              Each dict contains model_name, aggregate_metrics, and detailed performance data.
            - error_message (str or None): Error description if evaluation failed, 
              None if successful.

    Raises:
        subprocess.CalledProcessError: When the evaluation script fails execution.
        FileNotFoundError: When the results pickle file cannot be found.
        pickle.UnpicklingError: When the results file is corrupted or invalid.
    """
    seed_outdir = f"{outdir}/seed_{seed}"
    cmd = [
        sys.executable, script_path,
        "--artifacts_dir", artifacts_dir,
        "--outdir", seed_outdir,
        "--seed", str(seed)
    ]

    try:
        subprocess.run(cmd, check=True, text=True)
        results_file = f"{seed_outdir}/complete_results.pkl"
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            return results, None
        else:
            return None, f"Results file not found: {results_file}"
    except subprocess.CalledProcessError as e:
        return None, f"Seed {seed} failed with exit code {e.returncode}"


# ==============================================================================
# Metrics Extraction
# ==============================================================================

def extract_metrics(results: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract and standardize key performance metrics from evaluation results.

    This function processes raw evaluation results and extracts standardized performance
    metrics for statistical analysis. It handles missing metrics gracefully and ensures
    consistent data structure across different model types.

    Args:
        results (List[Dict]): Raw evaluation results from model evaluation script.
            Each dictionary should contain:
            - 'model_name' (str): Name/identifier of the evaluated model
            - 'aggregate_metrics' (Dict): Dictionary of computed performance metrics
              including macro_f1, macro_auc, weighted_auc, exact_match_accuracy, etc.

    Returns:
        List[Dict[str, Any]]: Standardized metrics for each model. Each dictionary contains:
            - 'model_name' (str): Model identifier
            - 'macro_f1' (float): Macro-averaged F1 score (0.0-1.0)
            - 'macro_auc' (float): Macro-averaged AUC-ROC score (0.0-1.0)  
            - 'weighted_auc' (float): Sample-weighted AUC-ROC score (0.0-1.0)
            - 'exact_match_accuracy' (float): Exact label match accuracy (0.0-1.0)
            - 'micro_f1' (float): Micro-averaged F1 score (0.0-1.0)
            - 'weighted_macro_f1' (float): Sample-weighted macro F1 score (0.0-1.0)
            Missing metrics are filled with np.nan for statistical consistency.

    Note:
        This function filters out invalid results (None values or missing aggregate_metrics)
        and only returns metrics for successfully evaluated models.
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
    Perform bootstrap evaluation across multiple random seeds for statistical robustness.

    This is the core function that orchestrates the bootstrap evaluation process. It generates
    multiple random seeds and runs the evaluation script with each seed to assess model 
    performance variability and statistical significance. The function tracks successful
    and failed runs for quality assessment.

    Args:
        script_path (str): Absolute path to the evaluation script (e.g., 'mibig3_stratified_evaluation.py').
            The script must accept --artifacts_dir, --outdir, and --seed arguments.
        artifacts_dir (str): Directory containing model artifacts, embeddings, and data files
            required for evaluation. Should contain subdirectories for each model type.
        outdir (str): Base output directory where all bootstrap results will be saved.
            Individual seed results will be saved in subdirectories named 'seed_{N}'.
        dataset_version (str): Dataset version identifier for result organization and naming
            (e.g., 'MIBIG1', 'MIBIG3').
        n_seeds (int, optional): Number of random seeds to evaluate. More seeds provide
            better statistical estimates but increase computation time. Default: 10.
            Recommended: 10-30 for research, 5 for testing.
        base_seed (int, optional): Master seed for reproducible random seed generation.
            Same base_seed will always generate the same sequence of evaluation seeds.
            Default: 42.

    Returns:
        Optional[Tuple[List[Dict], List[int], List[int]]]: Returns None if insufficient
        successful runs (< 3), otherwise returns tuple containing:
            - all_results (List[Dict]): Complete evaluation results for all successful runs.
              Each dict contains 'seed' (int) and 'metrics' (List[Dict]) with model performance.
            - successful_seeds (List[int]): List of seeds that completed successfully.
            - failed_seeds (List[int]): List of seeds that failed evaluation.

    Raises:
        ValueError: When n_seeds < 1 or base_seed is invalid.
        FileNotFoundError: When script_path doesn't exist.

    Note:
        - Requires at least 3 successful runs for meaningful statistical analysis
        - Failed seeds are tracked but don't stop the overall evaluation
        - Progress is reported for each evaluation run
        - Results are immediately processed to extract metrics
    """
    # Generate random seeds
    np.random.seed(base_seed)
    seeds = np.random.randint(1, 10000, n_seeds)

    all_results = []
    successful_seeds = []
    failed_seeds = []

    # Run evaluations
    for i, seed in enumerate(seeds):
        print(f"Running evaluation {i+1}/{n_seeds} with seed {seed}")
        
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
                failed_seeds.append(seed)
        else:
            failed_seeds.append(seed)

    if len(successful_seeds) < 3:
        print("Too few successful runs for statistical analysis!")
        return None

    return all_results, successful_seeds, failed_seeds


# ==============================================================================
# Statistical Analysis
# ==============================================================================

def calculate_bootstrap_statistics(all_results: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive bootstrap statistics for model performance analysis.

    This function computes mean, standard deviation, and 95% confidence intervals for each
    model and metric across all bootstrap runs. It provides the statistical foundation for
    comparing model performance and assessing significance of differences.

    Args:
        all_results (List[Dict]): Complete bootstrap evaluation results from all successful runs.
            Each dictionary contains:
            - 'seed' (int): Random seed used for this evaluation
            - 'metrics' (List[Dict]): Performance metrics for each model in this run
              Each metrics dict contains model_name and performance values

    Returns:
        Dict[str, Dict[str, Any]]: Comprehensive statistics for each model. Structure:
        {
            'model_name': {
                'model_name' (str): Model identifier
                'n_runs' (int): Number of successful evaluation runs
                
                For each metric (macro_f1, macro_auc, weighted_auc, exact_match_accuracy, 
                micro_f1, weighted_macro_f1):
                '{metric}_mean' (float): Mean performance across runs
                '{metric}_std' (float): Standard deviation (sample std with ddof=1)
                '{metric}_ci_lower' (float): Lower bound of 95% confidence interval
                '{metric}_ci_upper' (float): Upper bound of 95% confidence interval  
                '{metric}_values' (List[float]): Raw values from all runs
            }
        }

    Statistical Details:
        - Uses Student's t-distribution for confidence intervals when n > 2
        - Sample standard deviation (ddof=1) for unbiased estimation
        - Handles missing values (NaN) gracefully in calculations
        - Single-run case: std=0, CI bounds equal to mean value

    Note:
        - Models with fewer runs may have wider confidence intervals
        - NaN values in metrics are excluded from statistical calculations
        - Results are suitable for publication and statistical comparison
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
    Create a publication-ready summary table of bootstrap evaluation results.

    This function generates a pandas DataFrame summarizing model performance statistics
    with focus on a primary metric. The table is sorted by performance and includes
    confidence intervals for statistical comparison between models.

    Args:
        statistics (Dict[str, Dict]): Bootstrap statistics from calculate_bootstrap_statistics().
            Contains comprehensive performance statistics for each model including means,
            standard deviations, and confidence intervals for all metrics.
        focus_metric (str, optional): Primary metric for table sorting and emphasis.
            Must be one of: 'macro_auc', 'macro_f1', 'weighted_auc', 'exact_match_accuracy',
            'micro_f1', 'weighted_macro_f1'. Default: 'macro_auc'.

    Returns:
        pd.DataFrame: Summary table with columns:
            - 'Model' (str): Model name/identifier
            - 'N_Runs' (int): Number of successful bootstrap runs
            - '{Focus_Metric} Mean' (float): Mean performance for the focus metric
            - '{Focus_Metric} Std' (float): Standard deviation for the focus metric
            - '95% CI Lower' (float): Lower confidence interval bound
            - '95% CI Upper' (float): Upper confidence interval bound
            
            Rows are sorted by focus metric mean in descending order (best first).

    Usage Notes:
        - Non-overlapping confidence intervals indicate statistically significant differences
        - Models with higher standard deviations show more performance variability
        - Table is suitable for inclusion in research papers and reports
        - Focus metric selection should match research objectives
    """
    table_data = []
    for model_name, stats_dict in statistics.items():
        n_runs = stats_dict['n_runs']
        mean_val = stats_dict.get(f'{focus_metric}_mean', np.nan)
        std_val = stats_dict.get(f'{focus_metric}_std', np.nan)
        ci_lower = stats_dict.get(f'{focus_metric}_ci_lower', np.nan)
        ci_upper = stats_dict.get(f'{focus_metric}_ci_upper', np.nan)

        table_data.append({
            'Model': model_name,
            'N_Runs': n_runs,
            f'{focus_metric.replace("_", " ").title()} Mean': mean_val,
            f'{focus_metric.replace("_", " ").title()} Std': std_val,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper
        })

    # Sort by primary metric mean (descending)
    metric_col = f'{focus_metric.replace("_", " ").title()} Mean'
    table_data.sort(
        key=lambda x: x[metric_col] if not np.isnan(x[metric_col]) else -1,
        reverse=True
    )

    df = pd.DataFrame(table_data)
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
    Save comprehensive bootstrap evaluation results to multiple file formats.

    This function exports bootstrap evaluation results in multiple formats for different
    use cases: JSON for detailed analysis, pickle for programmatic access, and CSV for
    spreadsheet applications and publication tables.

    Args:
        statistics (Dict[str, Dict]): Computed bootstrap statistics from 
            calculate_bootstrap_statistics(). Contains means, standard deviations,
            confidence intervals, and raw values for each model and metric.
        all_results (List[Dict]): Raw bootstrap evaluation results from all successful runs.
            Contains complete evaluation data including individual seed results and
            detailed model performance metrics.
        outdir (str): Base output directory where results will be saved. A subdirectory
            'bootstrap_analysis' will be created containing all output files.
        dataset_version (str): Dataset identifier used for file naming (e.g., 'MIBIG3').
            Files will be prefixed with lowercase version (e.g., 'mibig3_bootstrap_*').

    Output Files:
        1. '{dataset}_bootstrap_statistics.json': Detailed statistics in JSON format
           - Human-readable structure with all computed statistics
           - Includes raw values for custom analysis
           - Suitable for programmatic processing
           
        2. '{dataset}_bootstrap_raw_results.pkl': Complete raw results in pickle format
           - Contains all evaluation data from successful runs
           - Preserves exact data structures for reproduction
           - Fastest loading for Python applications
           
        3. '{dataset}_bootstrap_summary.csv': Summary table in CSV format
           - Mean, std, and confidence intervals for key metrics
           - Ready for spreadsheet import and publication
           - Columns: model_name, {metric}_mean, {metric}_std, {metric}_ci_lower, {metric}_ci_upper

    Note:
        - Creates output directory if it doesn't exist
        - Handles numpy data type conversion for JSON serialization
        - CSV includes only core metrics: macro_f1, macro_auc, weighted_auc, exact_match_accuracy
        - Files are suitable for version control and sharing

    Raises:
        OSError: When output directory cannot be created or files cannot be written
        ValueError: When statistics or results contain invalid data structures
    """
    bootstrap_outdir = f"{outdir}/bootstrap_analysis"
    Path(bootstrap_outdir).mkdir(parents=True, exist_ok=True)

    # Save detailed statistics as JSON
    stats_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_statistics.json"
    with open(stats_file, 'w') as f:
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

    # Save raw results
    raw_file = f"{bootstrap_outdir}/{dataset_version.lower()}_bootstrap_raw_results.pkl"
    with open(raw_file, 'wb') as f:
        pickle.dump(all_results, f)

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


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    """
    Main bootstrap evaluation pipeline with command-line interface.

    This function provides a comprehensive command-line interface for bootstrap evaluation
    of MIBiG classification models. It handles argument parsing, path detection, execution
    coordination, and results presentation.

    The pipeline performs the following steps:
    1. Parse and validate command-line arguments
    2. Auto-detect file paths if not provided
    3. Execute bootstrap evaluation across multiple seeds
    4. Calculate comprehensive statistics with confidence intervals
    5. Generate publication-ready summary tables
    6. Export results in multiple formats (JSON, pickle, CSV)
    7. Display formatted results summary

    Command-line Arguments:
        --dataset: Dataset version ('mibig1' or 'mibig3') [REQUIRED]
        --artifacts_dir: Directory with model artifacts [auto-detected if not provided]
        --outdir: Output directory [auto-generated if not provided]  
        --n_seeds: Number of random seeds (default: 10)
        --base_seed: Master seed for reproducibility (default: 42)
        --focus_metric: Primary metric for analysis (default: 'macro_auc')

    Auto-Detection Logic:
        - artifacts_dir: "artifacts/classification/{dataset}" 
        - outdir: "results/{dataset}_bootstrap_evaluation"
        - script_path: "{dataset}_stratified_evaluation.py" in same directory

    Returns:
        int: Exit code (0 for success, 1 for failure)
            - 0: Bootstrap evaluation completed successfully
            - 1: Evaluation script not found, insufficient successful runs, or other errors

    Output Display:
        - Progress updates during evaluation
        - Success/failure summary
        - Performance summary table with focus metric
        - File locations for saved results

    Statistical Output:
        The function displays a formatted table showing:
        - Model names sorted by performance
        - Mean ± standard deviation for focus metric
        - 95% confidence intervals for statistical comparison
        - File paths for detailed results
    """
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

    # Run bootstrap evaluation
    results = bootstrap_evaluation(
        script_path, args.artifacts_dir, args.outdir,
        args.dataset.upper(), args.n_seeds, args.base_seed
    )

    if results is None:
        return 1

    all_results, successful_seeds, failed_seeds = results

    # Calculate statistics and save results
    statistics = calculate_bootstrap_statistics(all_results)
    summary_df = create_bootstrap_summary_table(statistics, args.focus_metric)
    save_bootstrap_results(statistics, all_results, args.outdir, args.dataset.upper())

    # Display final summary
    print(f"\nBootstrap evaluation completed: {len(successful_seeds)}/{args.n_seeds} successful runs")
    print(f"Results saved in: {args.outdir}/bootstrap_analysis/")
    print(f"Summary CSV: {args.outdir}/bootstrap_analysis/{args.dataset.lower()}_bootstrap_summary.csv")
    
    # Show performance summary table
    print(f"\nPerformance Summary ({args.focus_metric.replace('_', ' ').title()}):")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        model = row['Model']
        mean_val = row[f'{args.focus_metric.replace("_", " ").title()} Mean']
        std_val = row[f'{args.focus_metric.replace("_", " ").title()} Std']
        ci_lower = row['95% CI Lower']
        ci_upper = row['95% CI Upper']
        print(f"{model:<35} {mean_val:8.4f} ± {std_val:6.4f} [{ci_lower:7.4f}, {ci_upper:7.4f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
