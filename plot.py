#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def load_scores_from_csv(csv_file):
    """Load all scores from CSV file."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean_val = np.mean(data)
    return mean_val, mean_val - lower, upper - mean_val


def aggregate_scores_from_csv(data, metric, temp):
    """Aggregate scores: average across seeds per product, then across products."""
    # Group by epsilon and product_id
    epsilon_product_scores = defaultdict(lambda: defaultdict(list))
    
    metric_prefix = f"{metric}_temp{temp}_seed"
    
    for row in data:
        epsilon = float(row['epsilon'])
        product_id = row['product_id']
        
        # Collect scores from all seeds
        seed_scores = []
        for seed in ['1', '2', '3']:
            col_name = f"{metric_prefix}{seed}"
            if col_name in row and row[col_name]:
                try:
                    score = float(row[col_name])
                    seed_scores.append(score)
                except (ValueError, TypeError):
                    pass
        
        # Average across seeds for this product-epsilon
        if seed_scores:
            product_mean = np.mean(seed_scores)
            epsilon_product_scores[epsilon][product_id].append(product_mean)
    
    # For each epsilon, average across products and compute CI
    all_epsilons = sorted(epsilon_product_scores.keys())
    avg_scores = []
    ci_lower = []
    ci_upper = []
    
    for eps in all_epsilons:
        product_means = []
        for product_id, scores in epsilon_product_scores[eps].items():
            # Average if multiple entries per product (shouldn't happen, but just in case)
            product_mean = np.mean(scores) if scores else 0.0
            product_means.append(product_mean)
        
        if product_means:
            mean_score, ci_l, ci_u = bootstrap_ci(product_means, n_bootstrap=1000, confidence=0.95)
            avg_scores.append(mean_score)
            ci_lower.append(ci_l)
            ci_upper.append(ci_u)
        else:
            avg_scores.append(0.0)
            ci_lower.append(0.0)
            ci_upper.append(0.0)
    
    return all_epsilons, avg_scores, (ci_lower, ci_upper)


def plot_single_panel(ax, epsilons, avg_scores, error_bars, metric="rougeL", temp_str="", show_labels=True):
    """Plot a single panel with error bars."""
    yerr = error_bars if isinstance(error_bars, tuple) else error_bars
    
    color = '#2E86AB'
    ax.errorbar(epsilons, avg_scores, yerr=yerr, 
                marker='o', linestyle='-', linewidth=1.5, markersize=5,
                capsize=3, capthick=1.2, elinewidth=1.2,
                color=color, markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.5, alpha=0.8)
    
    metric_labels = {
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2", 
        "rougeL": "ROUGE-L",
        "bertf1": "BERTScore F1",
        "bertprecision": "BERTScore Precision",
        "bertrecall": "BERTScore Recall"
    }
    metric_label = metric_labels.get(metric, metric.upper())
    
    ax.set_xlabel('Privacy Budget (ε)', fontweight='medium')
    if show_labels:
        ax.set_ylabel(f'{metric_label} Score', fontweight='medium')
    else:
        ax.set_ylabel('')
    
    temp_display = temp_str.replace('_', '.')
    ax.text(0.98, 0.02, f'T = {temp_display}', 
            transform=ax.transAxes, fontsize=10, 
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(left=True, labelleft=True)


def plot_three_temps(metric, temp_data, output_file):
    """Plot three temperatures side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    temps = ["0.3", "0.7", "1.0"]
    
    for idx, temp in enumerate(temps):
        temp_key = f"temp{temp}"
        if temp_key in temp_data:
            epsilons, avg_scores, error_bars = temp_data[temp_key]
            show_labels = (idx == 0)
            plot_single_panel(axes[idx], epsilons, avg_scores, error_bars, 
                            metric, temp, show_labels=show_labels)
        else:
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="rougeL",
                       choices=["rouge1", "rouge2", "rougeL", "bertf1", "bertprecision", "bertrecall"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--csv_file", type=str, default="results/all_scores.csv",
                       help="Path to CSV file with all scores")
    parser.add_argument("--temp", type=str, default=None, help="Single temperature to process (e.g., 0.3)")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    print(f"Loading scores from {args.csv_file}...")
    data = load_scores_from_csv(args.csv_file)
    print(f"Loaded {len(data)} rows")
    
    temp_data = {}
    if args.temp:
        temps = [args.temp]
    else:
        temps = ["0.3", "0.7", "1.0"]
    
    for temp in temps:
        print(f"Processing temperature {temp} for metric {args.metric}...")
        epsilons, avg_scores, error_bars = aggregate_scores_from_csv(data, args.metric, temp)
        temp_data[f"temp{temp}"] = (epsilons, avg_scores, error_bars)
    
    if temp_data:
        output_file = os.path.join(args.output_dir, f"utility_{args.metric}.pdf")
        plot_three_temps(args.metric, temp_data, output_file)
    else:
        print("No data to plot")


if __name__ == "__main__":
    main()
