#!/usr/bin/env python3
import json
import glob
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import List, Dict, Tuple
import random

class DPSummarizerLlama:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda" if torch.cuda.is_available() else "mps", seed=1):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.device = device
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.vocab_size = len(self.tokenizer)
        self.hidden_size = self.model.config.hidden_size
        
        with torch.no_grad():
            self.lm_head_weight = self.model.lm_head.weight.float().cpu()
            self.lm_head_bias = self.model.lm_head.bias.float().cpu() if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None else torch.zeros(self.vocab_size, dtype=torch.float32)
        
        self.instruction_template = """Read this product review and summarize it in one sentence using exactly this format:

        "The product is a [type]. Customers praise its [positive1] and [positive2], but some complain about [negative1] and [negative2]."

        Review: {review}

        Summary: The product is a"""

        self.placeholder_review = "I bought this item recently. It has some good features like the design and ease of use. However, there are some drawbacks including the price and durability."
        
        print(f"Model: hidden_size={self.hidden_size}, vocab_size={self.vocab_size}")
        
        self.base_hidden_cache = {}
        
        self.rdp_orders = [1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]
    
    def create_instruction(self, review: str) -> str:
        return self.instruction_template.format(review=review[:300])
    
    def get_base_hidden(self, context: str) -> np.ndarray:
        """Get hidden state for prompt with NEUTRAL placeholder review."""
        instruction_placeholder = self.create_instruction(self.placeholder_review)
        messages = [
            {"role": "user", "content": instruction_placeholder},
            {"role": "assistant", "content": context}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs_base = self.tokenizer([base_prompt], return_tensors="pt", 
                                    truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs_base = self.model(**inputs_base, output_hidden_states=True)
            hidden_states_base = outputs_base.hidden_states[-1]
            seq_len_base = int((inputs_base.attention_mask[0] == 1).sum().item())
            base_hidden = hidden_states_base[0, seq_len_base - 1, :].float().cpu().numpy().astype(np.float64)
        
        return base_hidden
    
    def get_hidden_states_batch(self, reviews: List[str], context: str, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process reviews in batches to manage memory."""
        base_hidden = self.get_base_hidden(context)
        
        all_full_hidden = []
        
        for batch_start in range(0, len(reviews), batch_size):
            batch_end = min(batch_start + batch_size, len(reviews))
            batch_reviews = reviews[batch_start:batch_end]
            
            prompts = []
            for review in batch_reviews:
                instruction = self.create_instruction(review)
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": context}
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
            
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                for i in range(len(batch_reviews)):
                    seq_len = int((inputs.attention_mask[i] == 1).sum().item())
                    last_hidden = hidden_states[i, seq_len - 1, :].float().cpu().numpy().astype(np.float64)
                    all_full_hidden.append(last_hidden)
            
            if self.device != "cpu":
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        full_hidden = np.array(all_full_hidden, dtype=np.float64)
        review_contributions = full_hidden - base_hidden
        
        return full_hidden, review_contributions, base_hidden
    
    def dp_quantile(self, values: np.ndarray, quantile: float, epsilon: float) -> float:
        """DP quantile selection using the exponential mechanism."""
        n = len(values)
        sorted_values = np.sort(values)
        
        ranks = np.arange(1, n + 1) / n
        scores = -np.abs(ranks - quantile)
        
        sensitivity = 1.0 / n
        
        logits = (epsilon / (2 * sensitivity)) * scores
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        
        selected_idx = np.random.choice(n, p=probs)
        return sorted_values[selected_idx]
    
    def calibrate_clip_norm_dp(self, reviews: List[str], epsilon_R: float = 1.0, 
                               target_quantile: float = 0.50, batch_size: int = 8) -> float:
        print(f"\n{'='*70}")
        print(f"DP CLIP NORM CALIBRATION (ε_R={epsilon_R:.2f})")
        print(f"{'='*70}")
        
        # Calibrate at the start of generation
        context = ""
        print(f"Processing {len(reviews)} reviews in batches of {batch_size}...")
        _, review_contributions, _ = self.get_hidden_states_batch(reviews, context, batch_size)
        norms = np.array([np.linalg.norm(contrib) for contrib in review_contributions])
        n = len(norms)
        
        R_selected = self.dp_quantile(norms, target_quantile, epsilon_R)
        
        print(f"Selected R (DP quantile at {100*target_quantile:.0f}%): {R_selected:.1f}")
        print(f"n={n}, ε_R={epsilon_R:.2f}")
        
        return R_selected
    
    def rdp_gaussian_single(self, alpha: float, noise_multiplier: float) -> float:
        return alpha / (2 * noise_multiplier ** 2)
    
    def rdp_to_eps_delta(self, rdp_eps: float, alpha: float, delta: float) -> float:
        return rdp_eps + np.log(1.0 / delta) / (alpha - 1)
    
    def compute_epsilon_from_noise_multiplier(self, noise_multiplier: float, num_tokens: int, delta: float) -> float:
        min_eps = float('inf')
        
        for alpha in self.rdp_orders:
            rdp_eps_total = num_tokens * self.rdp_gaussian_single(alpha, noise_multiplier)
            eps = self.rdp_to_eps_delta(rdp_eps_total, alpha, delta)
            
            if eps > 0:
                min_eps = min(min_eps, eps)
        
        return min_eps
    
    def calibrate_noise_multiplier(self, target_epsilon: float, max_tokens: int, 
                                   delta: float, tol: float = 0.01) -> float:
        mu_low = 0.01
        mu_high = 100.0
        
        eps_low = self.compute_epsilon_from_noise_multiplier(mu_low, max_tokens, delta)
        if eps_low <= target_epsilon:
            return mu_low
        
        eps_high = self.compute_epsilon_from_noise_multiplier(mu_high, max_tokens, delta)
        while eps_high > target_epsilon:
            mu_high *= 2
            eps_high = self.compute_epsilon_from_noise_multiplier(mu_high, max_tokens, delta)
            if mu_high > 10000:
                print(f"Warning: Required noise multiplier very high (>{mu_high})")
                return mu_high
        
        for _ in range(100):
            mu_mid = (mu_low + mu_high) / 2
            eps_mid = self.compute_epsilon_from_noise_multiplier(mu_mid, max_tokens, delta)
            
            if abs(eps_mid - target_epsilon) < tol:
                return mu_mid
            
            if eps_mid > target_epsilon:
                mu_low = mu_mid
            else:
                mu_high = mu_mid
        
        return mu_mid
    
    def clip_l2(self, x: np.ndarray, R: float) -> Tuple[np.ndarray, bool]:
        norm = np.linalg.norm(x)
        if norm > R:
            return x * (R / norm), True
        return x, False
    
    def sample_token_gaussian_relative(self, review_contributions: np.ndarray, base_hidden: np.ndarray, 
                                        clip_norm: float, sigma: float, debug: bool = False) -> Tuple[int, str, int]:
        """Gaussian mechanism with greedy decoding on RELATIVE logits."""
        n = len(review_contributions)
        
        # Step 1: Clip (float64)
        clipped_contributions = []
        clipped_count = 0
        for i in range(n):
            contribution, was_clipped = self.clip_l2(review_contributions[i], clip_norm)
            clipped_contributions.append(contribution)
            if was_clipped:
                clipped_count += 1
        
        # Step 2: Aggregate (float64)
        avg_contribution = np.mean(clipped_contributions, axis=0)
        h_avg = base_hidden + avg_contribution
        
        # Step 3: Add Gaussian noise (float64)
        noise = np.random.normal(0, sigma, size=h_avg.shape)
        h_priv = h_avg + noise
        
        if debug:
            print(f"    Clipped {clipped_count}/{n} contributions")
            print(f"    Avg contribution norm: {np.linalg.norm(avg_contribution):.1f}")
            print(f"    Noise norm: {np.linalg.norm(noise):.1f}")
            print(f"    Signal/Noise ratio: {np.linalg.norm(avg_contribution) / np.linalg.norm(noise):.2f}")
            print(f"    Sigma: {sigma:.4f}")
        
        # Step 4: Compute logits for BOTH base and private
        h_priv_tensor = torch.tensor(h_priv, dtype=torch.float32)
        h_base_tensor = torch.tensor(base_hidden, dtype=torch.float32)
        
        with torch.no_grad():
            logits_priv = torch.matmul(self.lm_head_weight, h_priv_tensor) + self.lm_head_bias
            logits_base = torch.matmul(self.lm_head_weight, h_base_tensor) + self.lm_head_bias
            
            logits_relative = (logits_priv - logits_base).numpy().astype(np.float64)
        
        # Step 5: Greedy decode on RELATIVE logits
        token_id = int(np.argmax(logits_relative))
        token_str = self.tokenizer.decode([token_id])
        
        if debug:
            print(f"    Top 5 tokens (relative logits):")
            top5 = np.argsort(logits_relative)[-5:][::-1]
            for i in top5:
                print(f"      '{self.tokenizer.decode([i])}': rel={logits_relative[i]:.2f}")
            print(f"    Selected: '{token_str}'")
        
        return token_id, token_str, clipped_count
    
    def generate_tokens(self, reviews: List[str], context: str, 
                       clip_norm: float, sigma: float,
                       noise_multiplier: float, delta: float, epsilon_budget: float,
                       tokens_so_far: int, max_tokens: int, 
                       stop_tokens: List[str], batch_size: int = 8,
                       debug: bool = False) -> Tuple[str, int]:
        """Generate tokens until stop token or max reached."""
        generated = ""
        tokens_generated = 0
        
        for i in range(max_tokens):
            eps_if_emit = self.compute_epsilon_from_noise_multiplier(
                noise_multiplier, tokens_so_far + tokens_generated + 1, delta
            )
            if eps_if_emit > epsilon_budget:
                print(f"    Stopping early: next token would exceed ε budget ({eps_if_emit:.2f} > {epsilon_budget:.2f})")
                break
            
            _, review_contributions, base_hidden = self.get_hidden_states_batch(reviews, context, batch_size)
            
            token_id, token_str, clipped = self.sample_token_gaussian_relative(
                review_contributions, base_hidden, clip_norm, sigma, debug=(debug and i==0)
            )
            
            if i == 0:
                clip_rate = 100 * clipped / len(reviews)
                print(f"    Clipped: {clipped}/{len(reviews)} ({clip_rate:.1f}%)")
            
            # Check for stop tokens
            should_stop = False
            for stop_tok in stop_tokens:
                if stop_tok in token_str:
                    # Include up to the stop token
                    idx = token_str.find(stop_tok)
                    if idx > 0:
                        generated += token_str[:idx]
                    should_stop = True
                    break
            
            if should_stop:
                break
            
            generated += token_str
            context += token_str
            tokens_generated += 1
        
        return generated.strip(), tokens_generated, context
    
    def generate_summary(self, reviews: List[str], epsilon: float = 20.0,
                        clip_norm: float = None, epsilon_R: float = None,
                        delta: float = 1e-6, target_quantile: float = 0.50,
                        batch_size: int = 8, debug: bool = False) -> Dict[str, str]:
        n = len(reviews)
        
        if clip_norm is None:
            if epsilon_R is None:
                epsilon_R = 0.05 * epsilon
            clip_norm = self.calibrate_clip_norm_dp(reviews, epsilon_R=epsilon_R, 
                                                    target_quantile=target_quantile,
                                                    batch_size=batch_size)
            epsilon_gen = epsilon - epsilon_R
        else:
            epsilon_gen = epsilon
            epsilon_R = 0.0
        
        # The sentence structure we're generating:
        # "The product is a [description]. Customers praise its [pos1] and [pos2], but some complain about [neg1] and [neg2]."
        #
        # We generate in segments, stopping at the appropriate punctuation/words
        
        segments = [
            ("description", 5, ["."]),  # Stop at period
            ("positive1", 4, [" and"]),  # Stop at " and"
            ("positive2", 4, [","]),  # Stop at comma
            ("negative1", 4, [" and"]),  # Stop at " and"
            ("negative2", 4, ["."]),  # Stop at period
        ]
        
        total_max_tokens = sum(max_tok for _, max_tok, _ in segments)
        
        sensitivity = (2 * clip_norm) / n
        
        noise_multiplier = self.calibrate_noise_multiplier(
            target_epsilon=epsilon_gen,
            max_tokens=total_max_tokens,
            delta=delta
        )
        
        sigma = noise_multiplier * sensitivity
        
        actual_eps_at_max = self.compute_epsilon_from_noise_multiplier(
            noise_multiplier, total_max_tokens, delta
        )
        
        expected_noise_norm = sigma * np.sqrt(self.hidden_size)
        
        print(f"\n{'='*70}")
        print(f"DP SUMMARIZATION (Gaussian + RDP + Relative Logits + Greedy)")
        print(f"{'='*70}")
        print(f"Privacy: n={n}, ε_gen_target={epsilon_gen:.1f}, δ={delta:.0e}")
        print(f"  Clip norm R: {clip_norm:.1f}")
        print(f"  Sensitivity Δ = 2R/n = {sensitivity:.4f}")
        print(f"  Noise multiplier μ = σ/Δ = {noise_multiplier:.4f}")
        print(f"  Gaussian σ = {sigma:.4f}")
        print(f"  Expected noise norm ≈ σ√d = {expected_noise_norm:.1f}")
        print(f"  Max tokens: {total_max_tokens}")
        print(f"  ε at max tokens (RDP): {actual_eps_at_max:.2f}")
        print(f"  Batch size: {batch_size}")
        
        result = {}
        tokens_used_total = 0
        
        # Start with the beginning of the sentence (already in template)
        context = ""
        
        # Generate description
        print(f"\n--- Generating: description (max 5 tokens) ---")
        desc, tokens_used, context = self.generate_tokens(
            reviews, context, clip_norm, sigma,
            noise_multiplier, delta, epsilon_gen,
            tokens_used_total, 5, ["."], batch_size, debug
        )
        result["description"] = desc
        tokens_used_total += tokens_used
        print(f"    Generated: '{desc}' ({tokens_used} tokens)")
        
        # Add transition to positives
        context += ". Customers praise its "
        
        # Generate positive1
        print(f"\n--- Generating: positive1 (max 4 tokens) ---")
        pos1, tokens_used, context = self.generate_tokens(
            reviews, context, clip_norm, sigma,
            noise_multiplier, delta, epsilon_gen,
            tokens_used_total, 4, [" and", ","], batch_size, debug
        )
        result["positive1"] = pos1
        tokens_used_total += tokens_used
        print(f"    Generated: '{pos1}' ({tokens_used} tokens)")
        
        # Add transition
        context += " and "
        
        # Generate positive2
        print(f"\n--- Generating: positive2 (max 4 tokens) ---")
        pos2, tokens_used, context = self.generate_tokens(
            reviews, context, clip_norm, sigma,
            noise_multiplier, delta, epsilon_gen,
            tokens_used_total, 4, [",", "."], batch_size, debug
        )
        result["positive2"] = pos2
        tokens_used_total += tokens_used
        print(f"    Generated: '{pos2}' ({tokens_used} tokens)")
        
        # Add transition to negatives
        context += ", but some complain about "
        
        # Generate negative1
        print(f"\n--- Generating: negative1 (max 4 tokens) ---")
        neg1, tokens_used, context = self.generate_tokens(
            reviews, context, clip_norm, sigma,
            noise_multiplier, delta, epsilon_gen,
            tokens_used_total, 4, [" and", ","], batch_size, debug
        )
        result["negative1"] = neg1
        tokens_used_total += tokens_used
        print(f"    Generated: '{neg1}' ({tokens_used} tokens)")
        
        # Add transition
        context += " and "
        
        # Generate negative2
        print(f"\n--- Generating: negative2 (max 4 tokens) ---")
        neg2, tokens_used, context = self.generate_tokens(
            reviews, context, clip_norm, sigma,
            noise_multiplier, delta, epsilon_gen,
            tokens_used_total, 4, ["."], batch_size, debug
        )
        result["negative2"] = neg2
        tokens_used_total += tokens_used
        print(f"    Generated: '{neg2}' ({tokens_used} tokens)")
        
        context += "."
        
        actual_eps_gen = self.compute_epsilon_from_noise_multiplier(
            noise_multiplier, tokens_used_total, delta
        )
        actual_eps_total = actual_eps_gen + epsilon_R
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {tokens_used_total}/{total_max_tokens} tokens")
        print(f"  ε_R (calibration) = {epsilon_R:.2f}")
        print(f"  ε_gen (used, RDP) = {actual_eps_gen:.2f}")
        print(f"  ε_total = {actual_eps_total:.2f}")
        print(f"  δ = {delta:.1e}")
        print(f"{'='*70}")
        
        return result, context
    
    def format_summary(self, data: Dict[str, str]) -> str:
        return (
            f"The product is a {data['description']}. "
            f"Customers praise its {data['positive1']} and {data['positive2']}, "
            f"but some complain about {data['negative1']} and {data['negative2']}."
        )


def load_products(data_root: str, n: int = 1):
    files = sorted(glob.glob(os.path.join(data_root, "test", "*.json")))[:n]
    products = []
    for f in files:
        with open(f, "r") as file:
            data = json.load(file)
        reviews = []
        for r in data.get("customer_reviews", []):
            text = f"{r.get('title', '')} {r.get('text', '')}".strip()
            if text:
                reviews.append(text)
        ref = data.get("website_summaries", [{}])[0].get("verdict", "No reference")
        title = data.get("product_meta", {}).get("title", "Unknown")[:60]
        products.append({"title": title, "reviews": reviews, "reference": ref})
    return products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="summary_data")
    parser.add_argument("--reviews", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=20.0)
    parser.add_argument("--epsilon_R", type=float, default=None)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--target_quantile", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    summarizer = DPSummarizerLlama(seed=args.seed)
    products = load_products(args.data_root, n=1)
    
    p = products[0]
    print(f"\nProduct: {p['title']}")
    print(f"Total reviews available: {len(p['reviews'])}")
    reviews = p["reviews"][:args.reviews]
    print(f"Using: {len(reviews)} reviews")
    print(f"Reference: {p['reference'][:150]}...")
    
    result, generated_text = summarizer.generate_summary(
        reviews, args.epsilon, args.clip_norm, args.epsilon_R, 
        args.delta, args.target_quantile, args.batch_size, args.debug
    )
    
    formatted = summarizer.format_summary(result)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nGenerated context:\n  {generated_text}")
    print(f"\nFormatted summary:\n  {formatted}")
    print(f"\nExtracted fields:")
    for k, v in result.items():
        print(f"  {k}: '{v}'")
    print(f"\nReference:\n  {p['reference']}")


if __name__ == "__main__":
    main()