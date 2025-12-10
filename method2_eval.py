import argparse
import os
import json

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


def plot_all_metrics(metrics_data, ci_data, epsilons, output_dir):
    """Plot metrics with one line per k value (no seeds)."""
    metric_labels = {
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "bertf1": "BERTScore F1",
        "bertprecision": "BERTScore Precision",
        "bertrecall": "BERTScore Recall"
    }

    metrics = sorted(list(set(
        metric for k_dict in metrics_data.values()
        for metric in k_dict.keys()
    )))
    k_values = sorted(metrics_data.keys())

    linestyles = ['-', '--', '-.', ':']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 5))

        for k_idx, k in enumerate(k_values):
            scores = metrics_data[k].get(metric, [])
            linestyle = linestyles[k_idx % len(linestyles)]
            ax.plot(
                epsilons,
                scores,
                marker='o',
                linewidth=1.8,
                markersize=5,
                label=f"k={k}",
                linestyle=linestyle,
            )

            # CI shading if available
            if ci_data and k in ci_data and metric in ci_data[k]:
                lows = [ci_data[k][metric].get(eps, (s, s))[0] for eps, s in zip(epsilons, scores)]
                highs = [ci_data[k][metric].get(eps, (s, s))[1] for eps, s in zip(epsilons, scores)]
                ax.fill_between(epsilons, lows, highs, alpha=0.15, color=ax.lines[-1].get_color())

        ax.set_xlabel('Privacy Budget (ε)', fontweight='medium')
        ax.set_ylabel(f"{metric_labels.get(metric, metric.upper())}", fontweight='medium')
        ax.set_title(metric_labels.get(metric, metric.upper()), fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        format_axis(ax, epsilons)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"method2_utility_{metric}.pdf")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        logging.info(f"Plot saved to {output_path}")


def write_all_scores_csv(metrics_data, ci_data, epsilons, output_dir):
    """Write aggregated scores (mean) and optional CI to CSV."""
    import csv
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "method2_all_scores.csv")

    metrics = sorted(list(set(
        metric for k_dict in metrics_data.values()
        for metric in k_dict.keys()
    )))
    k_values = sorted(metrics_data.keys())

    fieldnames = ['epsilon']
    for metric in metrics:
        for k in k_values:
            fieldnames.append(f"{metric}_k{k}")
            if ci_data and k in ci_data and metric in ci_data[k]:
                fieldnames.append(f"{metric}_k{k}_ci_lower")
                fieldnames.append(f"{metric}_k{k}_ci_upper")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, eps in enumerate(epsilons):
            row = {'epsilon': eps}
            for metric in metrics:
                for k in k_values:
                    scores = metrics_data.get(k, {}).get(metric, [])
                    row[f"{metric}_k{k}"] = scores[idx] if idx < len(scores) else ''

                    if ci_data and k in ci_data and metric in ci_data[k]:
                        low, high = ci_data[k][metric].get(eps, (None, None))
                        row[f"{metric}_k{k}_ci_lower"] = low if low is not None else ''
                        row[f"{metric}_k{k}_ci_upper"] = high if high is not None else ''
            writer.writerow(row)

    logging.info(f"CSV saved to {output_path}")


def write_per_product_json(k_results, output_dir):
    """Write per-product scores and aggregates to JSON per k."""
    os.makedirs(output_dir, exist_ok=True)

    for k, res in k_results.items():
        payload = {
            "k": k,
            "scores": res.get("scores", {}),
            "ci": res.get("ci", {}),
            "per_product": res.get("per_product", []),
        }

        path = os.path.join(output_dir, f"method2_scores_k{k}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info(f"Per-product scores saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries from JSON files across k values (multi-product)")
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

    # Parse filenames to extract k values (e.g., results/k1.json or results/seed0_k1.json)
    import re
    file_to_k = {}
    k_values = set()

    for filepath in args.inputs:
        match = re.search(r'k(\d+)', filepath)
        if not match:
            logging.warning(f"Could not parse k from {filepath}, skipping")
            continue
        k = int(match.group(1))
        file_to_k[filepath] = k
        k_values.add(k)

    k_results = {}  # {k: result dict from evaluator}
    eps_union = set()

    for filepath, k in file_to_k.items():
        if not os.path.exists(filepath):
            logging.error(f"Input file not found: {filepath}")
            continue
        logging.info(f"Processing k={k} from {filepath}")
        summarizer = DPSummarizer()
        evaluator = Evaluator(summarizer, input_json=filepath, adapter_checkpoint=args.adapter)
        res = evaluator.compute_from_json(metrics=args.metrics, reference=args.reference)
        k_results[k] = res
        eps_union.update(res.get('scores', {}).keys())

    epsilons = sorted(eps_union)

    # Reorganize data: metrics_data[k][metric] = list of mean scores aligned to epsilons
    metrics_data = {}
    ci_data = {}
    for k, res in k_results.items():
        metrics_data[k] = {}
        ci_data[k] = {}
        for metric in args.metrics:
            scores_by_eps = res.get('scores', {})
            metrics_data[k][metric] = [scores_by_eps.get(eps, {}).get(metric, 0.0) for eps in epsilons]

            ci_by_eps = res.get('ci', {}) or {}
            if ci_by_eps:
                ci_data[k][metric] = {eps: ci_by_eps.get(eps, {}).get(metric, (None, None)) for eps in epsilons if eps in ci_by_eps}

    # Generate plots and CSV
    plot_all_metrics(metrics_data, ci_data, epsilons, args.output_dir)
    write_all_scores_csv(metrics_data, ci_data, epsilons, args.output_dir)
    write_per_product_json(k_results, args.output_dir)

    logging.info("Plots and CSV generated.")

if __name__ == "__main__":
    main()
