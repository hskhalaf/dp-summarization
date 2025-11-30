import argparse
import glob
import json
import os
import random
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import brentq
from scipy.stats import norm


class DPSummarizer:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device=None, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if device != "cpu" else torch.float32).to(device)
        self.model.eval()
       
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            self.final_norm = self.model.model.norm
        elif hasattr(self.model, "norm"):
            self.final_norm = self.model.norm
        else:
            self.final_norm = None
            print("Warning: No final norm layer found!")

        print(f"Model loaded: name = {model_name}, hidden_size={self.model.config.hidden_size}, vocab_size={len(self.tokenizer)}")

# GDP composition functions
    @staticmethod
    def _delta_gdp(epsilon: float, mu: float) -> float:
        """Compute δ(ε; μ) for a μ-GDP mechanism."""
        if mu <= 0:
            return 1.0
        term1 = norm.cdf(-epsilon / mu + mu / 2)
        term2 = np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2)
        return term1 - term2

    @staticmethod
    def _solve_mu_final(eps_target: float, delta_target: float, mu_min: float = 1e-6, mu_max: float = 100.0) -> float:
        """Solve for μ_final such that δ(eps_target; μ_final) = delta_target."""
        def f(mu):
            return DPSummarizer._delta_gdp(eps_target, mu) - delta_target
        if f(mu_min) > 0:
            raise ValueError(f"f(mu_min) > 0. Try smaller mu_min or check inputs.")
        if f(mu_max) < 0:
            raise ValueError(f"f(mu_max) < 0. Try larger mu_max or check inputs.")
        mu_final = brentq(f, mu_min, mu_max, xtol=1e-10, rtol=1e-10)
        return mu_final

    def _calibrate_gdp_params(self, eps_global: float, delta_global: float,  clip_norm: float, n_reviews: int, max_tokens: int) -> Tuple[float, float]:
        """Calibrate per-token epsilon/delta using GDP to achieve target global (ε, δ)."""
        # Step 1: Solve for μ_final from target (ε, δ)
        mu_final = self._solve_mu_final(eps_global, delta_global)
        # Step 2: Compute per-token noise σ_tok
        sensitivity = 2*clip_norm / n_reviews  # for add/remove adjacency
        mu_tok = mu_final / np.sqrt(max_tokens)
        sigma_tok = sensitivity / mu_tok
        # Step 3: Map to code's epsilon/delta parameters
        delta_tok = delta_global / max_tokens
        eps_tok = sensitivity * np.sqrt(2 * np.log(1.25 / delta_tok)) / sigma_tok
        return eps_tok, delta_tok

    def create_prompt(self, review: str, partial_summary: str = "") -> str:
        return (
            "Write one concise sentence following this pattern:\n"
            '"The product is a [TYPE]. Customers praise its [ASPECT_1] and [ASPECT_2], '
            'but some complain about [ISSUE_1] and [ISSUE_2]."\n'
            "Replace every bracketed token with specific phrases from the review. "
            "Do not invent details or leave placeholders.\n"
            f"Review excerpt: {review[:500]}\n"
            f"Current summary draft: {partial_summary}"
        )

    def _get_post_norm_vectors(self, reviews: List[str], partial_summary: str, batch_size: int) -> np.ndarray:
        """Extract post-norm vectors z_i for all reviews."""
        all_z_states = []
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i : i + batch_size]
            batch_prompts = [self.create_prompt(rev, partial_summary) for rev in batch_reviews]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
                for j in range(len(batch_reviews)):
                    seq_len = int(inputs.attention_mask[j].sum().item())
                    last_hidden = hidden_states[j, seq_len - 1, :].unsqueeze(0)
                    # Apply final norm per-example: z_i = final_norm(last_hidden_i)
                    if self.final_norm is not None:
                        z_i = self.final_norm(last_hidden.to(self.final_norm.weight.dtype))
                    else:
                        z_i = last_hidden
                    all_z_states.append(z_i[0].cpu().float().numpy())
        
        return np.array(all_z_states)  # [n, hidden_size]

    def get_next_token_logits_dp(self, reviews: List[str], partial_summary: str, epsilon: float, clip_norm: float, delta: float = 1e-6, batch_size: int = 8):
        """Get next token logits with differential privacy."""
        # Get post-norm vectors for all reviews
        z_states = self._get_post_norm_vectors(reviews, partial_summary, batch_size)
        n = len(z_states)
        
        # Clip each vector to L2 norm
        norms = np.linalg.norm(z_states, axis=1, keepdims=True)
        scale_factors = np.where(norms > clip_norm, clip_norm / norms, 1.0)
        clipped_z = z_states * scale_factors
        
        # Average in post-norm space
        mu = np.mean(clipped_z, axis=0)
        
        # Add Gaussian noise
        sensitivity = clip_norm / n
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, noise_scale, mu.shape)
        h_noisy = mu + noise
        
        # Map to logits via LM head
        with torch.no_grad():
            h_tensor = torch.tensor(h_noisy, dtype=torch.float32, device=self.device).unsqueeze(0)
            if hasattr(self.model, "lm_head"):
                logits = self.model.lm_head(h_tensor.to(self.model.lm_head.weight.dtype))
            else:
                logits = self.model.get_output_embeddings()(h_tensor.to(self.model.dtype))
            logits = logits[0, :].cpu().numpy()
        
        return logits, noise_scale

    def sample_token_from_logits(self, logits: np.ndarray, temperature: float = 0.7, top_k: int = 50) -> int:
        # Mask special tokens but allow EOS for proper stopping
        special_ids = set(self.tokenizer.all_special_ids)
        if self.tokenizer.eos_token_id is not None:
            # Remove EOS from masking - we need it for clean stopping
            special_ids.discard(self.tokenizer.eos_token_id)
        for token_id in special_ids:
            if token_id < len(logits):
                logits[token_id] = -float("inf")
        if temperature > 0:
            logits = logits / temperature
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
        probs = exp_logits / np.sum(exp_logits)
        sampled_idx = np.random.choice(len(top_k_indices), p=probs)
        return top_k_indices[sampled_idx]

    def _warmup_clip_norm(self, reviews: List[str], partial_summary: str = "", quantile: float = 0.90, batch_size: int = 8) -> float:
        """Compute clip_norm from quantile of per-example norms in post-norm space."""
        # for now, im ignoring the privacy cost of this step
        z_states = self._get_post_norm_vectors(reviews, partial_summary, batch_size)
        norms = np.linalg.norm(z_states, axis=1)
        clip_norm = float(np.quantile(norms, quantile))
        print(f"Warmup: clip_norm={clip_norm:.2f} ({quantile*100:.0f}th percentile, range: {norms.min():.2f}-{norms.max():.2f})")
        return clip_norm

    def generate_summary(self, reviews: List[str], epsilon: float, delta: float, 
                        clip_norm: float = None, max_tokens: int = 50, temperature: float = 0.7, 
                        batch_size: int = 8, warmup_quantile: float = 0.90) -> Tuple[str, float]:

        summary = "The product is "
        n_reviews = len(reviews)
        
        # Compute clip_norm if not provided
        if clip_norm is None:
            clip_norm = self._warmup_clip_norm(reviews, summary, quantile=warmup_quantile, batch_size=batch_size)
        
        # Calibrate per-token parameters from global target
        eps_tok, delta_tok = self._calibrate_gdp_params(epsilon, delta, clip_norm, n_reviews, max_tokens)
        print(f"GDP: (ε={epsilon}, δ={delta}) -> per-token (ε={eps_tok:.6f}, δ={delta_tok:.6e})")
        
        # Generate tokens
        for token_idx in range(max_tokens):
            logits, _ = self.get_next_token_logits_dp(reviews, summary, eps_tok, clip_norm, delta_tok, batch_size)
            token_id = self.sample_token_from_logits(logits, temperature)
            token_text = self.tokenizer.decode([token_id])
            summary += token_text
            
            # Early stopping conditions
            if summary.rstrip().endswith("."):
                if "praise" in summary.lower() and "complain" in summary.lower():
                    break
                if token_idx > max_tokens * 0.7:
                    break
        
        summary = summary.strip()
        if not summary.endswith("."):
            summary += "."
        return summary, eps_tok


def load_products(data_root: str, n: int = 1):
    files = sorted(glob.glob(os.path.join(data_root, "test", "*.json")))[:n]
    products = []
    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)
        reviews = []
        for review in data.get("customer_reviews", []):
            text = f"{review.get('title', '')} {review.get('text', '')}".strip()
            if text:
                reviews.append(text)
        reference = data.get("website_summaries", [{}])[0].get("verdict", "No reference")
        title = data.get("product_meta", {}).get("title", "Unknown")[:60]
        products.append({"title": title, "reviews": reviews, "reference": reference})
    return products


def main():
    parser = argparse.ArgumentParser(description="Differentially Private Review Summarization (GDP composition)")
    parser.add_argument("--data_root", default="summary_data")
    parser.add_argument("--num_products", type=int, default=50, help="Number of products to process")
    parser.add_argument("--epsilon", type=float, default=None, help="Single epsilon value (if provided, overrides range)")
    parser.add_argument("--epsilon_min", type=float, default=10.0, help="Minimum epsilon value")
    parser.add_argument("--epsilon_max", type=float, default=120.0, help="Maximum epsilon value")
    parser.add_argument("--epsilon_steps", type=int, default=20, help="Number of epsilon values to try")
    parser.add_argument("--delta", type=float, default=1e-6, help="Target global delta (δ_global)")
    parser.add_argument("--clip_norm", type=float, default=None, help="Clip norm (if None, computed from warmup quantile per product)")
    parser.add_argument("--warmup_quantile", type=float, default=0.90, help="Quantile for warmup clip_norm (0.85-0.95 recommended)")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results_50products_07temp.txt", help="Output text file path")
    
    args = parser.parse_args()
    
    # Load product data
    products = load_products(args.data_root, n=args.num_products)
    print(f"\nLoaded {len(products)} product(s)")
    
    # Initialize summarizer
    summarizer = DPSummarizer(seed=args.seed)
    print(1)    
    # Determine epsilon values to test
    if args.epsilon is not None:
        epsilon_values = [args.epsilon]
    else:
        if args.epsilon_steps == 1:
            epsilon_values = [args.epsilon_min]
        else:
            epsilon_values = np.linspace(args.epsilon_min, args.epsilon_max, args.epsilon_steps).tolist()
    
    print(f"\nTesting {len(epsilon_values)} epsilon value(s) with GDP composition...")
    print(f"Delta: {args.delta}")
    
    # Process each product
    all_results = []
    for prod_idx, product in enumerate(products, 1):
        print(f"\n{'='*70}")
        print(f"Product {prod_idx}/{len(products)}: {product['title']}")
        print(f"{'='*70}")
        reviews = product["reviews"]
        batch_size = len(reviews)  # Process all reviews in one batch
        print(f"Using all {len(reviews)} reviews for summarization (batch_size={batch_size})")
        
        # Compute clip_norm for this product (if not provided globally)
        if args.clip_norm is None:
            print("\nComputing clip_norm for this product...")
            initial_summary = "The product is "
            clip_norm = summarizer._warmup_clip_norm(reviews, initial_summary, quantile=args.warmup_quantile, batch_size=batch_size)
        else:
            clip_norm = args.clip_norm
            print(f"\nUsing provided clip_norm: {clip_norm:.2f}")
        
        # Generate summaries for each epsilon
        product_results = []
        for eps_idx, epsilon in enumerate(epsilon_values, 1):
            print(f"\n[{eps_idx}/{len(epsilon_values)}] Processing ε={epsilon:.2f}...")
            summary, eps_tok = summarizer.generate_summary(
                reviews=reviews,
                epsilon=epsilon,
                delta=args.delta,
                clip_norm=clip_norm,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                batch_size=batch_size,
                warmup_quantile=args.warmup_quantile,
            )
            product_results.append({
                "epsilon": epsilon,
                "eps_tok": eps_tok,
                "summary": summary
            })
            print(f"Summary: {summary}")
        
        all_results.append({
            "product_title": product['title'],
            "reference": product['reference'],
            "clip_norm": clip_norm,
            "num_reviews": len(reviews),
            "results": product_results
        })
    
    # Save all results
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Delta: {args.delta}\n")
        f.write(f"Number of products: {len(all_results)}\n")
        f.write(f"Number of epsilon values: {len(epsilon_values)}\n")
        f.write("="*70 + "\n\n")
        
        for prod_result in all_results:
            f.write(f"Product: {prod_result['product_title']}\n")
            f.write(f"Reference summary: {prod_result['reference']}\n")
            f.write(f"Clip norm: {prod_result['clip_norm']:.2f}\n")
            f.write(f"Reviews used: {prod_result['num_reviews']}\n")
            f.write("-"*70 + "\n")
            
            for result in prod_result['results']:
                f.write(f"ε_global = {result['epsilon']:.2f}, ε_tok = {result['eps_tok']:.6f}: {result['summary']}\n")
            f.write("\n" + "="*70 + "\n\n")
    
    print(f"\n{'='*70}")
    print(f"Results saved to {args.output}")
    print(f"Processed {len(all_results)} product(s) with {len(epsilon_values)} epsilon value(s) each")

if __name__ == "__main__":
    main()
