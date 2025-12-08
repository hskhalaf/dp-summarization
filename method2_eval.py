import argparse
import os

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker

from dpsummarizer.log import set_level, logging
from dpsummarizer.evaluate import Evaluator
from dpsummarizer.api import DPSummarizer

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def format_axis(ax, epsilons):
    """Apply consistent axis formatting without scientific notation."""
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(left=True, labelleft=True)

    # Avoid scientific notation on x-axis
    if max(epsilons) / min(epsilons) > 10:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:g}"))
    else:
        sf = ticker.ScalarFormatter(useOffset=False, useMathText=False)
        sf.set_scientific(False)
        ax.xaxis.set_major_formatter(sf)


def plot_all_metrics(metrics_data, seed_k_pairs, epsilons, output_dir):
    """Plot all metrics. For each metric, create 3 subplots (one per seed) with lines for each k.
    metrics_data: dict[seed][k][metric]->list of scores
    seed_k_pairs: dict[seed]->list of k values
    """
    metric_labels = {
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "bertf1": "BERTScore F1",
        "bertprecision": "BERTScore Precision",
        "bertrecall": "BERTScore Recall"
    }

    # Get unique metrics and seeds
    metrics = sorted(list(set(
        metric for seed_dict in metrics_data.values() 
        for k_dict in seed_dict.values() 
        for metric in k_dict.keys()
    )))
    seeds = sorted(metrics_data.keys())
    
    # For each metric, create a figure with 3 subplots (one per seed)
    for metric in metrics:
        fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5))
        if len(seeds) == 1:
            axes = [axes]
        
        # Line styles for k values
        linestyles = ['-', '--', '-.', ':']
        
        for seed_idx, seed in enumerate(seeds):
            ax = axes[seed_idx]
            k_values = sorted(seed_k_pairs.get(seed, []))
            
            # Plot a line for each k value
            for k_idx, k in enumerate(k_values):
                if seed in metrics_data and k in metrics_data[seed] and metric in metrics_data[seed][k]:
                    scores = metrics_data[seed][k][metric]
                    linestyle = linestyles[k_idx % len(linestyles)]
                    ax.plot(epsilons, scores, marker='o', linewidth=1.5, markersize=5, 
                           label=f"k={k}",
                           linestyle=linestyle)
            
            ax.set_xlabel('Privacy Budget (ε)', fontweight='medium')
            ax.set_ylabel(f"{metric_labels.get(metric, metric.upper())}", fontweight='medium')
            ax.set_title(f"Seed {seed}", fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            format_axis(ax, epsilons)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"method2_utility_{metric}.pdf")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        logging.info(f"Plot saved to {output_path}")


def write_all_scores_csv(metrics_data, seed_k_pairs, epsilons, output_dir):
    """Write all scores to a single CSV."""
    import csv
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "method2_all_scores.csv")
    
    # Get unique metrics
    metrics = sorted(list(set(
        metric for seed_dict in metrics_data.values() 
        for k_dict in seed_dict.values() 
        for metric in k_dict.keys()
    )))
    
    seeds = sorted(metrics_data.keys())
    
    # Build fieldnames: epsilon, then metric_seed_k combinations
    fieldnames = ['epsilon']
    for metric in metrics:
        for seed in seeds:
            k_values = sorted(seed_k_pairs.get(seed, []))
            for k in k_values:
                fieldnames.append(f"{metric}_seed{seed}_k{k}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, eps in enumerate(epsilons):
            row = {'epsilon': eps}
            for metric in metrics:
                for seed in seeds:
                    k_values = sorted(seed_k_pairs.get(seed, []))
                    for k in k_values:
                        col_name = f"{metric}_seed{seed}_k{k}"
                        if seed in metrics_data and k in metrics_data[seed] and metric in metrics_data[seed][k]:
                            scores = metrics_data[seed][k][metric]
                            row[col_name] = scores[idx] if idx < len(scores) else ''
                        else:
                            row[col_name] = ''
            writer.writerow(row)
    
    logging.info(f"CSV saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries from JSON files across seeds and k values")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs='+',
        required=True,
        help="Paths to input JSON files (e.g., results/seed0_k5.json results/seed1_k5.json ...)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save plots and CSVs."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=["rouge1", "rouge2", "rougeL", "bertf1", "bertprecision", "bertrecall"],
        help="Metrics to compute"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference summary for evaluation. If None, infers from JSON file."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Adapter checkpoint name"
    )
    args = parser.parse_args()

    set_level(args.log_level)
    logging.info("Starting evaluation across seeds and k values...")

    # Parse filenames to extract seed and k values
    # Expected format: seed<N>_k<M>.json
    import re
    seed_k_map = {}  # {seed: set of k values}
    file_to_seed_k = {}  # {filepath: (seed, k)}
    
    for filepath in args.inputs:
        match = re.search(r'seed(\d+)_k(\d+)', filepath)
        if not match:
            logging.warning(f"Could not parse seed/k from {filepath}, skipping")
            continue
        seed, k = match.group(1), int(match.group(2))
        file_to_seed_k[filepath] = (seed, k)
        if seed not in seed_k_map:
            seed_k_map[seed] = set()
        seed_k_map[seed].add(k)

    # Collect results per seed/k
    seed_results = {}  # {seed: {k: {epsilon: {metric: score}}}}
    eps_union = set()
    
    for filepath, (seed, k) in file_to_seed_k.items():
        if not os.path.exists(filepath):
            logging.error(f"Input file not found: {filepath}")
            continue
        logging.info(f"Processing seed={seed}, k={k} from {filepath}")
        summarizer = DPSummarizer()
        evaluator = Evaluator(summarizer, input_json=filepath, adapter_checkpoint=args.adapter)
        res = evaluator.compute_from_json(metrics=args.metrics, reference=args.reference)
        
        if seed not in seed_results:
            seed_results[seed] = {}
        seed_results[seed][k] = res['scores']
        eps_union.update(res['scores'].keys())

    epsilons = sorted(eps_union)

    # Reorganize data: metrics_data[seed][k][metric] = list of scores
    metrics_data = {}
    for seed in seed_results.keys():
        metrics_data[seed] = {}
        for k in seed_results[seed].keys():
            metrics_data[seed][k] = {}
            for metric in args.metrics:
                scores_by_eps = seed_results[seed][k]
                metrics_data[seed][k][metric] = [scores_by_eps.get(eps, {}).get(metric, 0.0) for eps in epsilons]

    # Generate single PDF and CSV
    plot_all_metrics(metrics_data, seed_k_map, epsilons, args.output_dir)
    write_all_scores_csv(metrics_data, seed_k_map, epsilons, args.output_dir)

    logging.info("Plot and CSV generated.")

if __name__ == "__main__":
    main()
