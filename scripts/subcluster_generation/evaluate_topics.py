#!/usr/bin/env python3
"""
Evaluate TNTM checkpoints (with optional cumulative filtering) and compare
against known subclusters. Optionally include an iPRESTO topics export.

Example:
  python scripts/subcluster_generation/evaluate_topics.py \
      --model_path results/subcluster_generation/2025-10-31_17-31-52 \
      --model_path results/subcluster_generation/2025-10-20_10-27-49 \
      --ipresto_topics external/iPRESTO/outputs_top_scan/presto_top/topics.txt \
      --n_topwords 120 \
      --thresholds 0.4 0.5 0.6 0.7 \
      --cumulative_threshold 0.95 \
      --min_contrib 0.001
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import cgrep.utils as utils
from cgrep.sc_gen import TNTM_inference, TNTM_trainer
from cgrep.sc_gen import known_sub_evaluation as kse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TNTM checkpoints (optionally against iPRESTO topics)."
    )
    parser.add_argument(
        "--model_path",
        action="append",
        default=[],
        help="Path to a TNTM run directory (metadata.pkl + model_state.pth). "
             "Pass multiple times to evaluate several models.",
    )
    parser.add_argument(
        "--ipresto_topics",
        type=str,
        default=None,
        help="Path to iPRESTO topics.txt (optional).",
    )
    parser.add_argument(
        "--known_subclusters_path",
        type=str,
        default="data/raw/known_subc_domains.txt",
        help="File containing known subclusters (default: data/raw/known_subc_domains.txt).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6, 0.7],
        help="Similarity thresholds to evaluate (default: 0.4 0.5 0.6 0.7).",
    )
    parser.add_argument(
        "--n_topwords",
        type=int,
        default=120,
        help="Number of candidate words to extract before filtering (default: 120).",
    )
    parser.add_argument(
        "--cumulative_threshold",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for filtering (default: 0.95).",
    )
    parser.add_argument(
        "--min_contrib",
        type=float,
        default=0.001,
        help="Minimum individual probability contribution (default: 0.001).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum number of terms to keep per topic after filtering (default: 50).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write aggregated results in JSON format.",
    )
    return parser.parse_args()


def load_tntm_topics(
    model_dir: Path,
    n_topwords: int,
    cumulative_threshold: float,
    min_contrib: float,
    max_length: int,
) -> Tuple[Dict[int, List[str]], Dict[str, float]]:
    """Extract and filter topics from a TNTM checkpoint."""
    metadata_path = model_dir / "metadata.pkl"
    state_path = model_dir / "model_state.pth"

    if not metadata_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Missing metadata or model state in {model_dir}")

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    config_dict = metadata["train_config"]
    TrainConfig = namedtuple("TrainConfig", config_dict.keys())
    config = TrainConfig(**config_dict)

    idx2word_raw = metadata["idx2word"]
    if isinstance(next(iter(idx2word_raw.keys())), str):
        idx2word = {int(k): v for k, v in idx2word_raw.items()}
    else:
        idx2word = idx2word_raw

    embeddings = torch.tensor(metadata["embeddings_proj"])
    state = torch.load(state_path, map_location="cpu")

    topwords_arr, probs_arr = TNTM_inference.get_topwords(
        n_topwords=n_topwords,
        mus_res=state["decoder.mus"],
        L_lower_res=state["decoder.L_lower"],
        D_log_res=state["decoder.log_diag"],
        emb_vocab_mat=embeddings,
        idx2word=idx2word,
        config=config,
        log_beta_prodlda=state.get("decoder.log_beta_prodlda"),
        lambda_logit=state.get("decoder.lambda_logit"),
    )

    topwords_df = pd.DataFrame(topwords_arr)
    probs_df = pd.DataFrame(probs_arr)
    filtered_topwords, filtered_probs = TNTM_trainer.filter_topic_word_matrices(
        topwords_df,
        probs_df,
        cumulative_threshold=cumulative_threshold,
        min_contrib=min_contrib,
    )

    # Trim topics to the requested maximum length
    if max_length is not None:
        filtered_topwords = {
            topic_idx: words[:max_length]
            for topic_idx, words in filtered_topwords.items()
        }

    lengths = [len(v) for v in filtered_topwords.values()]
    stats = {
        "count": len(lengths),
        "mean": float(np.mean(lengths)) if lengths else 0.0,
        "median": float(np.median(lengths)) if lengths else 0.0,
        "min": int(min(lengths)) if lengths else 0,
        "max": int(max(lengths)) if lengths else 0,
        "q25": float(np.quantile(lengths, 0.25)) if lengths else 0.0,
        "q75": float(np.quantile(lengths, 0.75)) if lengths else 0.0,
        "q90": float(np.quantile(lengths, 0.90)) if lengths else 0.0,
        "q95": float(np.quantile(lengths, 0.95)) if lengths else 0.0,
    }

    return filtered_topwords, stats


def parse_ipresto_topics(
    topics_path: Path,
    max_length: int,
) -> Tuple[Dict[int, List[str]], Dict[str, float]]:
    """Parse iPRESTO topics.txt into a dict compatible with TNTM evaluation."""
    topics: Dict[int, List[str]] = {}
    lengths: List[int] = []

    with topics_path.open("r") as f:
        header = next(f, None)  # skip header
        if header is None:
            raise ValueError(f"{topics_path} appears to be empty.")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            topic_id = int(parts[0])
            selected_domains = parts[3]
            tokens = [tok.split(":")[0] for tok in selected_domains.split(",") if tok]
            topics[topic_id] = tokens[:max_length] if max_length else tokens
            lengths.append(len(tokens))

    stats = {
        "count": len(lengths),
        "mean": float(np.mean(lengths)) if lengths else 0.0,
        "median": float(np.median(lengths)) if lengths else 0.0,
        "min": int(min(lengths)) if lengths else 0,
        "max": int(max(lengths)) if lengths else 0,
        "q25": float(np.quantile(lengths, 0.25)) if lengths else 0.0,
        "q75": float(np.quantile(lengths, 0.75)) if lengths else 0.0,
        "q90": float(np.quantile(lengths, 0.90)) if lengths else 0.0,
        "q95": float(np.quantile(lengths, 0.95)) if lengths else 0.0,
    }
    return topics, stats


def evaluate_topics(
    topics: Dict[int, List[str]],
    thresholds: Sequence[float],
    known_clusters: List[List[str]],
) -> List[Tuple[float, int, float]]:
    results = []
    for thr in thresholds:
        comparison = kse.compare_subclusters(
            topics,
            known_clusters,
            threshold=thr,
            ratio_type="known",
        )
        match_count = sum(val is not None for val in comparison.values())
        match_rate = match_count / len(known_clusters) if known_clusters else 0.0
        results.append((thr, match_count, match_rate))
    return results


def format_stats(stats: Dict[str, float]) -> str:
    return (
        f"count={stats['count']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
        f"min={stats['min']}, max={stats['max']}, "
        f"q25={stats['q25']:.2f}, q75={stats['q75']:.2f}, "
        f"q90={stats['q90']:.2f}, q95={stats['q95']:.2f}"
    )


def main() -> None:
    args = parse_args()

    model_paths = [Path(p) for p in args.model_path]
    if not model_paths and not args.ipresto_topics:
        raise SystemExit("Provide at least one --model_path or --ipresto_topics.")

    known_clusters = utils.read_known_clusters(args.known_subclusters_path)

    aggregate_results = {}

    for model_dir in model_paths:
        label = model_dir.name
        print(f"\n=== Evaluating TNTM model: {label} ===")
        topics, stats = load_tntm_topics(
            model_dir,
            n_topwords=args.n_topwords,
            cumulative_threshold=args.cumulative_threshold,
            min_contrib=args.min_contrib,
            max_length=args.max_length,
        )
        print("Topic length stats:", format_stats(stats))

        results = evaluate_topics(topics, args.thresholds, known_clusters)
        for thr, count, rate in results:
            print(f"  τ={thr:.2f} -> {count}/{len(known_clusters)} ({rate*100:.2f}%)")

        aggregate_results[label] = {
            "topic_length_stats": stats,
            "matches": [
                {"threshold": thr, "match_count": count, "match_rate": rate}
                for thr, count, rate in results
            ],
        }

        # Save filtered outputs next to the model for later inspection
        suffix = f"cum{int(args.cumulative_threshold*100):02d}_min{args.min_contrib}".replace(".", "")
        topwords_out = model_dir / f"topwords_{suffix}.csv"
        probs_out = model_dir / f"probs_{suffix}.csv"

        topwords_df = pd.DataFrame.from_dict(topics, orient="index")
        topwords_df.to_csv(topwords_out, header=False)

        # also keep the probabilities if desired (optional)
        # To keep script light, we do not save filtered probabilities again.

    if args.ipresto_topics:
        print(f"\n=== Evaluating iPRESTO topics: {args.ipresto_topics} ===")
        ip_topics, stats = parse_ipresto_topics(Path(args.ipresto_topics), args.max_length)
        print("Topic length stats:", format_stats(stats))

        results = evaluate_topics(ip_topics, args.thresholds, known_clusters)
        for thr, count, rate in results:
            print(f"  τ={thr:.2f} -> {count}/{len(known_clusters)} ({rate*100:.2f}%)")

        aggregate_results["iPRESTO"] = {
            "topic_length_stats": stats,
            "matches": [
                {"threshold": thr, "match_count": count, "match_rate": rate}
                for thr, count, rate in results
            ],
        }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(aggregate_results, f, indent=2)
        print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
