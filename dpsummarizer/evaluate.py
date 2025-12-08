
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from .api import DPSummarizer
from . import utils

# Configure matplotlib styling
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

_bert_scorer = None
_bert_device = None

def _get_bert_scorer():
    global _bert_scorer, _bert_device
    if _bert_scorer is None:
        try:
            if torch.backends.mps.is_available():
                _bert_device = "mps"
            elif torch.cuda.is_available():
                _bert_device = "cuda"
            else:
                _bert_device = "cpu"
            _bert_scorer = BERTScorer(lang='en', device=_bert_device, batch_size=256)
        except ImportError:
            return None
    return _bert_scorer

class Evaluator:
    def __init__(self, summarizer: DPSummarizer, input_json: str = "results/summaries_export.json", adapter_checkpoint: str | None = None):
        self.summarizer = summarizer
        self.input_json = input_json
        self.adapter = None
        
        # Load adapter if checkpoint is provided
        if adapter_checkpoint:
            self._load_adapter(adapter_checkpoint)
    
    def _load_adapter(self, checkpoint_path: str):
        """Load adapter checkpoint from dpsummarizer/adapter_checkpoints."""
        from .adapter import SoftPromptAdapter
        
        # If it's just a name without extension, use utils.load_adapter
        if not checkpoint_path.endswith('.pt') and '/' not in checkpoint_path:
            # This is just a name like "Llama_3_2_1B_Instruct_v4"
            # We need an adapter instance to load into
            # For now, just use utils.load_adapter helper to get the path
            checkpoint_path_obj = utils.ADAPTER_DIR / f"{checkpoint_path}.pt"
            if checkpoint_path_obj.exists():
                try:
                    checkpoint = torch.load(checkpoint_path_obj, map_location='cpu')
                    self.adapter = checkpoint
                    logging.info(f"Loaded adapter from {checkpoint_path_obj}")
                except Exception as e:
                    logging.error(f"Failed to load adapter from {checkpoint_path_obj}: {e}")
            else:
                logging.warning(f"Adapter checkpoint not found: {checkpoint_path_obj}")
        else:
            # Full path provided
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.adapter = checkpoint
                logging.info(f"Loaded adapter from {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to load adapter from {checkpoint_path}: {e}")


    def compute_rouge_scores(self, summary: str, reference: str) -> dict:
        """Compute ROUGE scores between a summary and reference."""
        if not reference or not summary:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, summary)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }

    def compute_bertscore_batch(self, summaries: list[str], references: list[str]) -> list[dict]:
        """Compute BERTScore for a batch of summaries and references."""
        scorer = _get_bert_scorer()
        if scorer is None:
            return [{"f1": 0.0, "precision": 0.0, "recall": 0.0} for _ in summaries]
        try:
            print(f"Computing BERTScore for {len(summaries)} pairs in batches of 256...")
            P, R, F1 = scorer.score(summaries, references, batch_size=32, verbose=True)
            results = []
            for i in tqdm(range(len(summaries)), desc="Processing BERTScore results", leave=False):
                results.append({
                    "f1": float(F1[i].item()),
                    "precision": float(P[i].item()),
                    "recall": float(R[i].item())
                })
            return results
        except Exception as e:
            print(f"Error in BERTScore batch: {e}")
            return [{"f1": 0.0, "precision": 0.0, "recall": 0.0} for _ in summaries]

    def compute_metric_score(self, summary: str, reference: str, metric: str) -> float:
        """Compute a specific metric score between summary and reference."""
        if metric in ["rouge1", "rouge2", "rougeL"]:
            score_dict = self.compute_rouge_scores(summary, reference)
            return score_dict[metric]
        elif metric == "bertf1":
            results = self.compute_bertscore_batch([summary], [reference])
            return results[0]["f1"]
        elif metric == "bertprecision":
            results = self.compute_bertscore_batch([summary], [reference])
            return results[0]["precision"]
        elif metric == "bertrecall":
            results = self.compute_bertscore_batch([summary], [reference])
            return results[0]["recall"]
        else:
            return 0.0

    def compute_from_json(self, 
        metrics: list[str] | None = None, 
        reference: str | None = None
    ) -> dict:
        """
        Compute evaluation metrics from summaries in the JSON file.
        
        :param metrics: List of metrics to compute (e.g., ['rouge1', 'rouge2', 'rougeL', 'bertf1'])
        :param reference: Reference summary for evaluation. If None, uses product metadata summary or first summary.
        :return: Dictionary with results organized by epsilon and metric
        """
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]
        
        with open(self.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get reference summary
        if reference is None:
            # Try to find reference in product metadata or use None
            reference = data.get("product_metadata", {}).get("summary", "")
            if not reference and data.get("summaries"):
                # If no reference in metadata, use the summary with highest epsilon as proxy
                summaries = sorted(data["summaries"], key=lambda x: x["epsilon"], reverse=True)
                reference = summaries[0]["summary"] if summaries else ""
        
        results = {
            "product_metadata": data.get("product_metadata", {}),
            "scores": {}
        }
        
        summaries_list = data.get("summaries", [])
        
        # Prepare for batch BERT computation if needed
        use_bert = any(m.startswith("bert") for m in metrics)
        bert_results = {}
        if use_bert and reference:
            summaries_text = [s["summary"] for s in summaries_list]
            references_text = [reference] * len(summaries_text)
            bert_batch = self.compute_bertscore_batch(summaries_text, references_text)
            for i, s in enumerate(summaries_list):
                bert_results[s["epsilon"]] = bert_batch[i]
        
        # Compute all metrics
        for summary_data in tqdm(summaries_list, desc="Computing scores"):
            epsilon = summary_data["epsilon"]
            summary = summary_data["summary"]
            
            results["scores"][epsilon] = {}
            
            for metric in metrics:
                if metric in ["rouge1", "rouge2", "rougeL"]:
                    score = self.compute_metric_score(summary, reference, metric)
                elif metric.startswith("bert") and epsilon in bert_results:
                    bert_data = bert_results[epsilon]
                    if metric == "bertf1":
                        score = bert_data["f1"]
                    elif metric == "bertprecision":
                        score = bert_data["precision"]
                    elif metric == "bertrecall":
                        score = bert_data["recall"]
                    else:
                        score = 0.0
                else:
                    score = 0.0
                
                results["scores"][epsilon][metric] = score
        
        return results

    def plot_results(self, results: dict | None = None, output_path: str | None = None):
        """
        Plot evaluation results.
        
        :param results: Results dictionary from compute_from_json(). If None, uses self.results.
        :param output_path: Path to save the plot. If None, displays the plot.
        """
        if results is None:
            if not hasattr(self, 'results'):
                raise ValueError("No results to plot. Call compute_from_json() first or pass results.")
            results = self.results
        
        scores = results.get("scores", {})
        if not scores:
            print("No scores to plot")
            return
        
        # Organize data by metric
        epsilons = sorted(scores.keys())
        metrics = set()
        for epsilon_scores in scores.values():
            metrics.update(epsilon_scores.keys())
        metrics = sorted(metrics)
        
        # Create subplots
        num_metrics = len(metrics)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        metric_labels = {
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2",
            "rougeL": "ROUGE-L",
            "bertf1": "BERTScore F1",
            "bertprecision": "BERTScore Precision",
            "bertrecall": "BERTScore Recall"
        }
        
        color = '#2E86AB'
        
        for idx, metric in enumerate(metrics):
            ax: plt.Axes = axes[idx]
            metric_scores = [scores[eps].get(metric, 0) for eps in epsilons]
            
            # Plot with error bars (using scores as point estimates)
            ax.errorbar(epsilons, metric_scores, 
                       marker='o', linestyle='-', linewidth=1.5, markersize=5,
                       color=color, markerfacecolor=color, markeredgecolor='white',
                       markeredgewidth=0.5, alpha=0.8)
            
            metric_label = metric_labels.get(metric, metric.upper())
            ax.set_xlabel('Privacy Budget (ε)', fontweight='medium')
            ax.set_ylabel(f'{metric_label} Score', fontweight='medium')
            ax.set_title(f'{metric_label}', fontweight='bold')
            
            # Styling
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
            
            # Use log scale if epsilon values span multiple orders of magnitude
            if max(epsilons) / min(epsilons) > 10:
                ax.set_xscale('log')
        
        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()