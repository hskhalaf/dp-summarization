#!/usr/bin/env python3
"""
Extract results from output file and format them as results_seedX_tempY.txt
"""
import argparse
import re
import json
import os
import glob


def load_reference_summaries(data_root: str, n: int = 100):
    """Load reference summaries from data files."""
    files = sorted(glob.glob(os.path.join(data_root, "test", "*.json")))[:n]
    references = {}
    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)
        title = data.get("product_meta", {}).get("title", "Unknown")[:60]
        reference = data.get("website_summaries", [{}])[0].get("verdict", "No reference")
        references[title] = reference
    return references


def parse_output_file(filepath, data_root=None):
    """Parse the output file and extract results."""
    products = []
    current_product = None
    current_epsilon = None
    current_eps_tok = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Load reference summaries if data_root is provided
    references = {}
    if data_root:
        try:
            references = load_reference_summaries(data_root, n=100)
        except Exception as e:
            print(f"Warning: Could not load reference summaries: {e}")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for product header: "Product X/100: [name]"
        product_match = re.match(r'Product (\d+)/100: (.+)', line)
        if product_match:
            if current_product:
                products.append(current_product)
            
            product_num = int(product_match.group(1))
            product_name = product_match.group(2).strip()
            
            current_product = {
                "title": product_name,
                "reference": references.get(product_name, "No reference"),
                "clip_norm": None,
                "num_reviews": None,
                "summaries": []
            }
        
        # Check for clip_norm: "Warmup: clip_norm=92.39 ..."
        elif current_product and "Warmup: clip_norm=" in line:
            clip_match = re.search(r'clip_norm=([\d.]+)', line)
            if clip_match:
                current_product["clip_norm"] = float(clip_match.group(1))
        
        # Check for number of reviews: "Using all X reviews..."
        elif current_product and "Using all" in line and "reviews" in line:
            reviews_match = re.search(r'Using all (\d+) reviews', line)
            if reviews_match:
                current_product["num_reviews"] = int(reviews_match.group(1))
        
        # Check for epsilon processing: "[X/20] Processing ε=..."
        elif current_product and re.match(r'\[\d+/\d+\] Processing', line):
            eps_match = re.search(r'Processing ε=([\d.]+?)(?:\.\.\.|$)', line)
            if eps_match:
                current_epsilon = float(eps_match.group(1))
        
        # Check for GDP line to extract epsilon_tok: "GDP: (ε=..., δ=...) -> per-token (ε=..., δ=...)"
        elif current_product and "GDP:" in line and "per-token" in line:
            eps_tok_match = re.search(r'per-token \(ε=([\d.]+)', line)
            if eps_tok_match:
                current_eps_tok = float(eps_tok_match.group(1))
        
        # Check for summary: "Summary: ..."
        elif current_product and line.startswith("Summary:"):
            summary = line.replace("Summary:", "").strip()
            # Handle multi-line summaries (check next lines if they don't start with patterns)
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # Stop if we hit a new section
                if (next_line.startswith("[") and "/20]" in next_line) or \
                   next_line.startswith("Product ") or \
                   next_line.startswith("GDP:") or \
                   next_line.startswith("======================================================================") or \
                   (next_line and not next_line.startswith("Summary:")):
                    # Check if it's a continuation of the summary
                    if next_line and not any([
                        next_line.startswith("["),
                        next_line.startswith("Product "),
                        next_line.startswith("GDP:"),
                        next_line.startswith("======================================================================"),
                        next_line.startswith("Computing"),
                        next_line.startswith("Using all"),
                        next_line.startswith("Warmup:"),
                    ]):
                        # It might be a continuation, but be careful
                        # Only add if it looks like part of the summary (not a new command)
                        if len(next_line) > 10 and not next_line.startswith("**"):
                            summary += " " + next_line
                            j += 1
                        else:
                            break
                    else:
                        break
                else:
                    break
            
            if current_epsilon is not None:
                current_product["summaries"].append({
                    "epsilon": current_epsilon,
                    "eps_tok": current_eps_tok if current_eps_tok is not None else 0.0,
                    "summary": summary
                })
                current_epsilon = None
                current_eps_tok = None
        
        i += 1
    
    # Don't forget the last product
    if current_product:
        products.append(current_product)
    
    return products


def format_results(products, delta=1e-6):
    """Format products into results file format."""
    lines = []
    
    # Header
    lines.append(f"Delta: {delta}")
    lines.append(f"Number of products: {len(products)}")
    
    # Count epsilon values (should be same for all products)
    num_epsilons = len(products[0]["summaries"]) if products else 0
    lines.append(f"Number of epsilon values: {num_epsilons}")
    lines.append("=" * 70)
    lines.append("")
    
    # Each product
    for product in products:
        lines.append(f"Product: {product['title']}")
        lines.append(f"Reference summary: {product['reference']}")
        lines.append(f"Clip norm: {product['clip_norm']:.2f}" if product['clip_norm'] else "Clip norm: N/A")
        lines.append(f"Reviews used: {product['num_reviews']}" if product['num_reviews'] else "Reviews used: N/A")
        lines.append("-" * 70)
        
        # Sort summaries by epsilon
        summaries = sorted(product['summaries'], key=lambda x: x['epsilon'])
        
        for summary_data in summaries:
            eps = summary_data['epsilon']
            eps_tok = summary_data['eps_tok']
            summary = summary_data['summary']
            lines.append(f"ε_global = {eps:.2f}, ε_tok = {eps_tok:.6f}: {summary}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract results from output file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output file (e.g., dp_seed1_48288199.out)")
    parser.add_argument("--data_root", type=str, default="summary_data",
                       help="Root directory for data files (to load reference summaries)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path (e.g., results/results_seed1_temp1.txt)")
    parser.add_argument("--delta", type=float, default=1e-6,
                       help="Delta value for header")
    args = parser.parse_args()
    
    print(f"Parsing output file: {args.output_file}")
    products = parse_output_file(args.output_file, data_root=args.data_root)
    print(f"Found {len(products)} products")
    
    # Check how many summaries per product
    if products:
        num_summaries = [len(p['summaries']) for p in products]
        print(f"Summaries per product: min={min(num_summaries)}, max={max(num_summaries)}, avg={sum(num_summaries)/len(num_summaries):.1f}")
    
    # Format and save
    results_text = format_results(products, delta=args.delta)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(results_text)
    
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

