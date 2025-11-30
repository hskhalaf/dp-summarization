#!/usr/bin/env python3
import argparse
import re
import csv
import os
from collections import defaultdict
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

_bert_scorer = None
_bert_device = None

def get_bert_scorer():
    global _bert_scorer, _bert_device
    if _bert_scorer is None:
        try:
            from bert_score import BERTScorer
            import torch
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

def parse_results_file(filepath, max_products=88):
    products = []
    current_product = None
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and (max_products is None or len(products) < max_products):
        line = lines[i].strip()
        if line.startswith("Product:"):
            if current_product:
                products.append(current_product)
                if max_products is not None and len(products) >= max_products:
                    break
            current_product = {
                "title": line.replace("Product:", "").strip(),
                "reference": "",
                "summaries": {}
            }
        elif line.startswith("Reference summary:"):
            if current_product:
                current_product["reference"] = line.replace("Reference summary:", "").strip()
        elif "ε_global" in line:
            if current_product:
                eps_match = re.search(r'ε_global = ([\d.]+)', line)
                if eps_match:
                    epsilon = float(eps_match.group(1))
                    if ':' in line:
                        summary = line.split(':', 1)[1].strip()
                        current_product["summaries"][epsilon] = summary
        i += 1
    if current_product and (max_products is None or len(products) < max_products):
        products.append(current_product)
    if max_products is not None:
        return products[:max_products]
    return products

def compute_rouge_scores(summary, reference):
    if not reference or not summary:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

def compute_bertscore_batch(summaries, references):
    scorer = get_bert_scorer()
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

def match_products_across_seeds(all_seed_products):
    product_map = {}
    for seed_id, seed_products in all_seed_products.items():
        for product in seed_products:
            product_key = product["title"][:60]
            if product_key not in product_map:
                product_map[product_key] = {}
            product_map[product_key][seed_id] = product
    return product_map

def compute_all_scores_for_product(product_data, reference, metric, epsilon):
    scores = []
    for seed_id in sorted(product_data.keys()):
        product = product_data[seed_id]
        if epsilon in product["summaries"]:
            summary = product["summaries"][epsilon]
            if metric in ["rouge1", "rouge2", "rougeL"]:
                score_dict = compute_rouge_scores(summary, reference)
                scores.append(score_dict[metric])
            elif metric == "bertf1":
                results = compute_bertscore_batch([summary], [reference])
                scores.append(results[0]["f1"])
            elif metric == "bertprecision":
                results = compute_bertscore_batch([summary], [reference])
                scores.append(results[0]["precision"])
            elif metric == "bertrecall":
                results = compute_bertscore_batch([summary], [reference])
                scores.append(results[0]["recall"])
            else:
                scores.append(0.0)
        else:
            scores.append(None)
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_csv", type=str, default="results/all_scores.csv")
    parser.add_argument("--max_products", type=int, default=88)
    args = parser.parse_args()
    
    metrics = ["rouge1", "rouge2", "rougeL", "bertf1", "bertprecision", "bertrecall"]
    temps = ["0.3", "0.7", "1.0"]
    seeds = ["1", "2", "3"]
    
    all_products_data = {}
    
    print(f"Starting computation for {len(metrics)} metrics, {len(temps)} temperatures, {len(seeds)} seeds")
    print(f"Metrics: {', '.join(metrics)}")
    
    for temp in tqdm(temps, desc="Loading data"):
        temp_file = temp if temp != "1.0" else "1"
        all_seed_products = {}
        
        for seed in seeds:
            filepath = os.path.join(args.results_dir, f"results_seed{seed}_temp{temp_file}.txt")
            if os.path.exists(filepath):
                products = parse_results_file(filepath, max_products=args.max_products)
                all_seed_products[seed] = products
        
        if not all_seed_products:
            continue
        
        product_map = match_products_across_seeds(all_seed_products)
        
        for product_key, product_data in product_map.items():
            if product_key not in all_products_data:
                all_products_data[product_key] = {}
            all_products_data[product_key][temp] = product_data
    
    all_product_keys = sorted(all_products_data.keys())
    product_id_map = {key: idx + 1 for idx, key in enumerate(all_product_keys)}
    
    all_epsilons = set()
    for product_data_dict in all_products_data.values():
        for temp, product_data in product_data_dict.items():
            for seed_id, product in product_data.items():
                all_epsilons.update(product["summaries"].keys())
    all_epsilons = sorted(all_epsilons)
    
    print(f"Found {len(all_product_keys)} products, {len(all_epsilons)} epsilon values")
    
    use_bert = any(m.startswith("bert") for m in metrics)
    if use_bert:
        print("Preparing BERTScore computation...")
        all_summaries = []
        all_references = []
        all_product_keys_list = []
        all_epsilons_list = []
        all_seed_indices = []
        all_temp_list = []
        
        for product_key in all_product_keys:
            product_data_dict = all_products_data[product_key]
            reference = None
            for temp in temps:
                if temp in product_data_dict:
                    for seed_id in seeds:
                        if seed_id in product_data_dict[temp]:
                            reference = product_data_dict[temp][seed_id]["reference"]
                            break
                    if reference:
                        break
            
            if not reference:
                continue
            
            for temp in temps:
                if temp not in product_data_dict:
                    continue
                product_data = product_data_dict[temp]
                for epsilon in all_epsilons:
                    for seed_id in seeds:
                        if seed_id in product_data and epsilon in product_data[seed_id]["summaries"]:
                            all_summaries.append(product_data[seed_id]["summaries"][epsilon])
                            all_references.append(reference)
                            all_product_keys_list.append((product_key, epsilon, seed_id, temp))
                            all_epsilons_list.append(epsilon)
                            all_seed_indices.append(seed_id)
                            all_temp_list.append(temp)
        
        if all_summaries:
            bert_results = compute_bertscore_batch(all_summaries, all_references)
            bert_scores = {}
            for (product_key, epsilon, seed_id, temp), result in zip(all_product_keys_list, bert_results):
                key = (product_key, epsilon, temp)
                if key not in bert_scores:
                    bert_scores[key] = {}
                bert_scores[key][seed_id] = result
    
    all_data = []
    for product_key in tqdm(all_product_keys, desc="Computing scores"):
        product_data_dict = all_products_data[product_key]
        product_id = product_id_map[product_key]
        
        reference = None
        for temp in temps:
            if temp in product_data_dict:
                for seed_id in seeds:
                    if seed_id in product_data_dict[temp]:
                        reference = product_data_dict[temp][seed_id]["reference"]
                        break
                if reference:
                    break
        
        if not reference:
            continue
        
        for epsilon in all_epsilons:
            row = {
                "product_id": product_id,
                "epsilon": epsilon
            }
            
            for metric in metrics:
                for temp in temps:
                    if temp not in product_data_dict:
                        for seed in seeds:
                            row[f"{metric}_temp{temp}_seed{seed}"] = ""
                        continue
                    
                    product_data = product_data_dict[temp]
                    
                    if use_bert and metric.startswith("bert"):
                        key = (product_key, epsilon, temp)
                        if key in bert_scores:
                            scores = []
                            for seed in seeds:
                                if seed in bert_scores[key]:
                                    score_dict = bert_scores[key][seed]
                                    if metric == "bertf1":
                                        scores.append(score_dict["f1"])
                                    elif metric == "bertprecision":
                                        scores.append(score_dict["precision"])
                                    elif metric == "bertrecall":
                                        scores.append(score_dict["recall"])
                                    else:
                                        scores.append(0.0)
                                else:
                                    scores.append("")
                        else:
                            scores = [""] * len(seeds)
                    else:
                        scores = compute_all_scores_for_product(product_data, reference, metric, epsilon)
                    
                    for i, seed in enumerate(seeds):
                        col_name = f"{metric}_temp{temp}_seed{seed}"
                        if i < len(scores) and scores[i] is not None:
                            row[col_name] = scores[i]
                        else:
                            row[col_name] = ""
            
            all_data.append(row)
    
    if not all_data:
        print("No data to save")
        return
    
    fieldnames = ["product_id", "epsilon"]
    for metric in metrics:
        for temp in temps:
            for seed in seeds:
                fieldnames.append(f"{metric}_temp{temp}_seed{seed}")
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Saved {len(all_data)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()

