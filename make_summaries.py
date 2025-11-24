import argparse
import glob
import json
import os
import random
from typing import List
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        # Get the final layer norm (RMSNorm) for per-example normalization
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            self.final_norm = self.model.model.norm
        elif hasattr(self.model, "norm"):
            self.final_norm = self.model.norm
        else:
            self.final_norm = None
            print("Warning: No final norm layer found!")

        print(f"Model loaded: name = {model_name}, hidden_size={self.model.config.hidden_size}, vocab_size={len(self.tokenizer)}")

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

    def get_next_token_logits_dp(self, reviews: List[str], partial_summary: str, epsilon: float, clip_norm: float, delta: float = 1e-6, batch_size: int = 8):
        all_z_states = []  # z_i = final_norm(last_hidden_i) - post-norm vectors
        # Get hidden states and apply final norm per-example
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i : i + batch_size]
            batch_prompts = [self.create_prompt(rev, partial_summary) for rev in batch_reviews]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
                for j in range(len(batch_reviews)):
                    seq_len = int(inputs.attention_mask[j].sum().item())
                    last_hidden = hidden_states[j, seq_len - 1, :].unsqueeze(0)  # [1, hidden_size]
                    # Apply final RMSNorm per-example: z_i = final_norm(last_hidden_i)
                    if self.final_norm is not None:
                        z_i = self.final_norm(last_hidden.to(self.final_norm.weight.dtype))
                    else:
                        z_i = last_hidden
                    all_z_states.append(z_i[0].cpu().float().numpy())
        
        all_z_states = np.array(all_z_states)  # [n, hidden_size] - post-norm vectors
        n = len(all_z_states)
        # Clip each post-norm vector z_i to L2 norm <= clip_norm, then average
        clipped_z_states = []
        for i in range(n):
            norm = np.linalg.norm(all_z_states[i])
            if norm > clip_norm:
                clipped_z_states.append(all_z_states[i] * (clip_norm / norm))
            else:
                clipped_z_states.append(all_z_states[i])
        clipped_z_states = np.array(clipped_z_states)
        # Average in post-norm space
        mu = np.mean(clipped_z_states, axis=0)
        # Add Gaussian noise in post-norm space
        # Sensitivity is clip_norm / n (max change from one example)
        sensitivity = clip_norm / n
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, noise_scale, mu.shape)
        h_noisy = mu + noise 
        # Map to logits via lm_head only (linear, preserves DP)
        # No norm applied again - we're already in post-norm space
        with torch.no_grad():
            h_noisy_tensor = torch.tensor(h_noisy, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, hidden_size]
            # Get logits from LM head (linear transformation, post-processing preserves DP)
            if hasattr(self.model, "lm_head"):
                logits = self.model.lm_head(h_noisy_tensor.to(self.model.lm_head.weight.dtype))
            else:
                logits = self.model.get_output_embeddings()(h_noisy_tensor.to(self.model.dtype))
            logits = logits[0, :].cpu().numpy()  # [vocab_size]
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
        all_z_states = []  # Collect post-norm vectors z_i
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i : i + batch_size]
            batch_prompts = [self.create_prompt(rev, partial_summary) for rev in batch_reviews]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
                for j in range(len(batch_reviews)):
                    seq_len = int(inputs.attention_mask[j].sum().item())
                    last_hidden = hidden_states[j, seq_len - 1, :].unsqueeze(0)
                    
                    # Apply final norm per-example (same as in get_next_token_logits_dp)
                    if self.final_norm is not None:
                        z_i = self.final_norm(last_hidden.to(self.final_norm.weight.dtype))
                    else:
                        z_i = last_hidden
                    
                    all_z_states.append(z_i[0].cpu().float().numpy())
        # Compute norms in post-norm space
        norms = [np.linalg.norm(z) for z in all_z_states]
        clip_norm = np.quantile(norms, quantile)
        print(f"Warmup: computed clip_norm={clip_norm:.2f} from {quantile*100:.0f}th percentile of post-norm vector norms (range: {min(norms):.2f}-{max(norms):.2f})")
        return float(clip_norm)

    def generate_summary(self, reviews: List[str], epsilon: float = 10.0, clip_norm: float = None, delta: float = 1e-6, max_tokens: int = 50, temperature: float = 0.7, batch_size: int = 8, warmup_quantile: float = 0.90) -> str:
        summary = "The product is "
        
        # Warmup: determine clip_norm from quantile if not provided
        if clip_norm is None:
            clip_norm = self._warmup_clip_norm(reviews, summary, quantile=warmup_quantile, batch_size=batch_size)
        
        for token_idx in range(max_tokens):
            token_eps = epsilon / np.sqrt(max_tokens)
            logits, _ = self.get_next_token_logits_dp(reviews, summary, token_eps, clip_norm, delta, batch_size)
            token_id = self.sample_token_from_logits(logits, temperature)
            token_text = self.tokenizer.decode([token_id])
            summary += token_text
            print(f"\r  {token_idx + 1} tokens: {summary}", end="", flush=True)
            if summary.rstrip().endswith("."):
                if "praise" in summary.lower() and "complain" in summary.lower(): break
                if token_idx > max_tokens * 0.7: break
        summary = summary.strip()
        print()  # ensure progress line ends cleanly
        if not summary.endswith("."):
            summary += "."
        print(f"\nGenerated: {summary}")
        return summary


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
    parser = argparse.ArgumentParser(description="Differentially Private Review Summarization")
    parser.add_argument("--data_root", default="summary_data")
    parser.add_argument("--reviews", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--clip_norm", type=float, default=None, help="Clip norm (if None, computed from warmup quantile)")
    parser.add_argument("--warmup_quantile", type=float, default=0.90, help="Quantile for warmup clip_norm (0.85-0.95 recommended)")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    products = load_products(args.data_root, n=1)
    product = products[0] # just use the first product for now
    print(f"\nProduct: {product['title']}")
    print(f"Total reviews available: {len(product['reviews'])}")
    reviews = product["reviews"][:args.reviews]
    print(f"Using {len(reviews)} reviews for summarization")
    summarizer = DPSummarizer(seed=args.seed)
    summary = summarizer.generate_summary(
        reviews=reviews,
        epsilon=args.epsilon,
        clip_norm=args.clip_norm,
        delta=args.delta,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        warmup_quantile=args.warmup_quantile,
    )
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Generated Summary:\n  {summary}")
    print(f"\nReference Summary (first 200 chars):\n  {product['reference'][:200]}...")


if __name__ == "__main__":
    main()