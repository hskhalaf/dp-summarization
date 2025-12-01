import torch
import numpy as np  
import random
from pathlib import Path
import json

from .log import logging
from .frozenllm import FrozenLLM
from .adapter import SoftPromptAdapter
from . import utils

class DPSummarizer:
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def read_documents(self, file_path: str, max_docs: int | None = None) -> list[tuple[list[str], str]]:
        """
        Read all .json files under the given path and return a list of
        (public_reviews, public_summary) pairs suitable for training.

        - public_reviews: list[str] of review texts (title + text).
        - public_summary: one concise reference summary (verdict if present, else compact pros/cons).

        :param file_path: Directory containing .json files (scans recursively).
        :type file_path: str
        :param max_docs: Maximum number of documents to read. If None, read all.
        :type max_docs: int | None
        
        :return: List of (reviews, summary) pairs.
        """
        root = Path(file_path)
        if not root.exists():
            logging.warning(f"Path not found: {file_path}")
            return []
        
        logging.info(f"Reading documents from: {file_path}")

        pairs = []
        count_files = 0
        count_used = 0

        for fp in root.rglob("*.json"):
            count_files += 1
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read {fp}: {e}")
                continue

            # Collect reviews
            reviews = []
            for r in data.get("customer_reviews", []):
                title = r.get("title") or ""
                text = r.get("text") or ""
                txt = f"{title} {text}".strip()
                if txt:
                    reviews.append(txt)

            # Build a single summary string
            summary = None
            ws = data.get("website_summaries", [])
            if ws and isinstance(ws, list):
                # Prefer the first verdict if present
                verdict = ws[0].get("verdict") if isinstance(ws[0], dict) else None
                if verdict and isinstance(verdict, str) and verdict.strip():
                    summary = verdict.strip()
                else:
                    # Compact representation from pros/cons
                    pros = []
                    cons = []
                    for s in ws:
                        if not isinstance(s, dict):
                            continue
                        if s.get("pros"):
                            pros.extend([str(p) for p in s.get("pros", []) if p])
                        if s.get("cons"):
                            cons.extend([str(c) for c in s.get("cons", []) if c])
                    # Assemble concise summary if no verdict
                    if pros or cons:
                        pros_str = "; ".join(pros[:6]) if pros else "N/A"
                        cons_str = "; ".join(cons[:6]) if cons else "N/A"
                        summary = f"Pros: {pros_str}. Cons: {cons_str}."

            # Only keep pairs with both parts
            if reviews and summary:
                pairs.append((reviews, summary))
                count_used += 1
                if max_docs is not None and count_used >= max_docs:
                    logging.info(f"Reached max_docs={max_docs}; stopping early.")
                    break

        logging.info(f"Scanned {count_files} JSON file(s); using {count_used} with reviews+summary.")
        return pairs

    def train(
        self, 
        llm: FrozenLLM, 
        public_dataset: list[tuple[list[str], str]],
        instruction_template: str,
        m: int,
        lr: float = 1e-3,
        num_epochs: int = 5,
        max_reviews_per_product: int = 100,
    ):
        """
        :param llm: Frozen LLM instance.
        :type llm: FrozenLLM

        :param public_dataset: A list of (public_reviews, public_summary) pairs.
        :type public_dataset: list[tuple[list[str], str]]
        
        :param instruction_template: Instructions for the LLM to generate summaries.
        :type instruction_template: str

        :param m: Number of soft prompt tokens.
        :type m: int

        :param lr: Learning rate for adapter.
        :type lr: float

        :param num_epochs: Number of training epochs.
        :type num_epochs: int

        :return: A trained SoftPromptAdapter instance.
        :rtype: SoftPromptAdapter
        """
        logging.info("Starting adapter training...")
        adapter = SoftPromptAdapter(d_model=llm.d_model, m=m).to(llm.device)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=1e-2)

        for epoch in range(1, num_epochs+1):
            logging.info(f"Epoch {epoch}/{num_epochs}...")
            for i, (reviews, summary) in enumerate(public_dataset):
                logging.debug(f"[E{epoch}] Processing product {i+1}/{len(public_dataset)}")
                # encode public reviews into hidden vectors h_i
                h_list = []
                for j, r in enumerate(reviews):
                    logging.debug(f"[E{epoch}] [P{i+1}] Encoding review {j+1}/{max_reviews_per_product}")
                    prompt_text   = utils.create_prompt(r)
                    with torch.no_grad():
                        hidden_states = llm.encode(prompt_text)
                    h_list.append(hidden_states[-1])
                    
                    if j + 1 >= max_reviews_per_product:
                        break
                    
                # aggregate to mean representation 
                h_mean = torch.stack(h_list, dim=0).mean(dim=0)  # (d_model,)

                target_dtype = next(adapter.parameters()).dtype
                h_mean = h_mean.to(target_dtype)

                # map to soft prompt
                soft_prompt = adapter(h_mean).to(llm.device)     # (m, d_model)

                # prepare instruction + target
                instruction_ids = llm.tokenizer.encode(
                    instruction_template,
                    add_special_tokens=False,
                )

                target_ids = llm.tokenizer.encode(
                    summary,
                    add_special_tokens=False,
                )

                # compute loss
                optimizer.zero_grad()
                loss = llm.lm_loss(
                    soft_prefix=soft_prompt,
                    instruction_ids=instruction_ids,
                    target_ids=target_ids,
                )
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    logging.debug(f"[E{epoch}] [P{i+1}] Loss: {loss.item():.4f}")

        logging.info("Adapter training complete.")
        return adapter
                
    def summarize(
        self,
        llm: FrozenLLM,
        adapter: SoftPromptAdapter,
        private_reviews: list[str],
        instruction_template: str,
        C: float,
        epsilon: float,
        delta: float,
        max_new_tokens: int = 64,
    ):
        """
        One-shot DP summarization of private reviews. Uses add/remove
        adjacency.

        :param llm: Frozen LLM instance.
        :type llm: FrozenLLM

        :param adapter: Trained SoftPromptAdapter instance.
        :type adapter: SoftPromptAdapter

        :param private_reviews: List of private reviews to summarize.
        :type private_reviews: list[str]

        :param instruction_template: Instructions for the LLM to generate summaries.
        :type instruction_template: str

        :param C: Clipping norm for DP-SGD.
        :type C: float

        :param epsilon: Privacy budget epsilon.
        :type epsilon: float

        :param delta: Privacy budget delta.
        :type delta: float

        :param max_new_tokens: Maximum number of new tokens to generate.
        :type max_new_tokens: int
        """
        # encode private reviews into clipped hidden vectors h_i
        h_list = []
        for r in private_reviews:
            prompt_text   = utils.create_prompt(r)
            hidden_states = llm.encode(prompt_text)
            h_i           = hidden_states[-1]

            # L2 clipping
            norm = torch.norm(h_i, p=2)
            if norm > C:
                h_i = h_i * (C / norm)

            h_list.append(h_i)

        n     = len(h_list)
        h_hat = torch.stack(h_list, dim=0).mean(dim=0) # (d_model,)

        # add Gaussian noise
        sensitivity = 2.0 * C / n
        sigma  = utils.calibrate_gaussian_sigma(epsilon, delta, sensitivity)
        noise  = np.random.normal(loc=0.0, scale=sigma, size=h_hat.shape)
        noise  = torch.tensor(noise, dtype=h_hat.dtype, device=h_hat.device)
        h_priv = h_hat + noise

        # h_priv = h_hat

        # map to soft prompt
        target_dtype = next(adapter.parameters()).dtype
        h_priv = h_priv.to(target_dtype)
        soft_prompt = adapter(h_priv)                       # (m, d_model)
        soft_prompt = soft_prompt.to(llm.model.dtype)

        # decode summary
        instruction_ids = llm.tokenizer.encode(
            instruction_template, 
            add_special_tokens=False
        )
        instruction_embeddings = llm.token_embed(instruction_ids)

        # convert soft_prompt to numpy
        soft_prompt_np = soft_prompt.detach().cpu().numpy()
        init_embeddings = np.concatenate([soft_prompt_np, instruction_embeddings], axis=0)

        summary = llm.decode(
            init_embeddings,
            max_new_tokens
        )

        return summary
 

