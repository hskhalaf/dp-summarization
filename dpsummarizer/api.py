import torch
import numpy as np  
import random
from tqdm import tqdm

from .log import logging
from .frozenllm import FrozenLLM
from .adapter import SoftPromptAdapter
from . import utils

class DPSummarizer:
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(
        self, 
        llm: FrozenLLM, 
        public_dataset: list[tuple[list[str], str, dict]],
        instruction_template: str,
        m: int,
        lr: float = 1e-3,
        num_epochs: int = 5,
        max_reviews_per_product: int = 100,
        weight_decay: float = 5e-2,
        batch_size: int = 8,
        docs_per_epoch: int | None = None,
    ):
        """
        :param llm: Frozen LLM instance.
        :type llm: FrozenLLM

        :param public_dataset: A list of (public_reviews, public_summary, metadata) pairs.
        :type public_dataset: list[tuple[list[str], str, dict]]
        
        :param instruction_template: Instructions for the LLM to generate summaries.
        :type instruction_template: str

        :param m: Number of soft prompt tokens.
        :type m: int

        :param lr: Learning rate for adapter.
        :type lr: float

        :param num_epochs: Number of training epochs.
        :type num_epochs: int

        :param max_reviews_per_product: Maximum number of reviews to use per product.
        :type max_reviews_per_product: int

        :param weight_decay: Weight decay for optimizer.
        :type weight_decay: float

        :param batch_size: Number of reviews to process in parallel.
        :type batch_size: int

        :param docs_per_epoch: Number of documents to sample per epoch. If None, uses all documents.
        :type docs_per_epoch: int | None

        :return: A trained SoftPromptAdapter instance.
        :rtype: SoftPromptAdapter
        """
        logging.info("Starting adapter training...")

        adapter = SoftPromptAdapter(
            d_in=llm.d_model, 
            d_model=llm.d_model, 
            m=m
        ).to(llm.device).float()

        optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)

        # caching hidden states to speed up training on multiple epochs
        hidden_cache = {}

        for epoch in range(1, num_epochs+1):
            logging.info(f"Epoch {epoch}/{num_epochs}...")
            epoch_losses = []

            # Sample subset of documents if docs_per_epoch is specified
            if docs_per_epoch is not None and docs_per_epoch < len(public_dataset):
                epoch_dataset = random.sample(public_dataset, docs_per_epoch)
                logging.debug(f"Sampled {docs_per_epoch} documents for epoch {epoch}")
            else:
                epoch_dataset = public_dataset

            # progress bar for running epochs
            pbar = tqdm(
                enumerate(epoch_dataset), 
                total=len(epoch_dataset), 
                desc=f"Epoch {epoch}/{num_epochs}"
            )
            for i, (reviews, summary, metadata) in pbar:
                # encode public reviews into embeddings e_i (direct from LLM hidden states)
                reviews_to_process = reviews[:max_reviews_per_product]
                e_list = []
                
                # Process reviews in batches
                for batch_start in range(0, len(reviews_to_process), batch_size):
                    batch_reviews = reviews_to_process[batch_start:batch_start + batch_size]
                    batch_prompts = [
                        utils.create_prompt(
                            r, 
                            title=metadata.get('title'), 
                            categories=metadata.get('categories')
                        ) 
                        for r in batch_reviews
                    ]
                    
                    # Check cache for uncached prompts
                    uncached_prompts = []
                    uncached_indices = []
                    batch_vecs = [None] * len(batch_prompts)

                    for idx, prompt_text in enumerate(batch_prompts):
                        if prompt_text in hidden_cache:
                            batch_vecs[idx] = hidden_cache[prompt_text].to(llm.device)
                        else:
                            uncached_prompts.append(prompt_text)
                            uncached_indices.append(idx)

                    # Batch encode uncached prompts
                    if uncached_prompts:
                        with torch.no_grad():
                            batch_hidden_vecs = llm.encode_batch(uncached_prompts)  # (u, d_model) 

                        for j, prompt_text in enumerate(uncached_prompts):
                            vec = batch_hidden_vecs[j].float().cpu()  # (d_model,)
                            hidden_cache[prompt_text] = vec  # store on CPU
                            batch_vecs[uncached_indices[j]] = vec.to(llm.device)

                    # Now batch_vecs is a list of (d_model,) tensors in original order
                    for vec in batch_vecs:
                        e_list.append(vec.float())

                if not e_list:
                    continue
                    
                # aggregate to mean representation 
                e_stack = torch.stack(e_list, dim=0)  # (n, d_model)
                e_mean = e_stack.mean(dim=0)          # (d_model,)

                # map to soft prompt (add batch dimension)
                soft_prompt = adapter(e_mean.unsqueeze(0))  # (1, m, d_model)
                soft_prompt = soft_prompt.squeeze(0)  # (m, d_model)

                # prepare instruction + target
                instruction_ids = llm.tokenizer.encode(
                    instruction_template.format(metadata["title"], metadata["categories"]),
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

                epoch_losses.append(loss.item())
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if i % 10 == 0:
                    logging.debug(f"[E{epoch}] [P{i+1}] Loss: {loss.item():.4f}")

            if epoch_losses:
                mean_loss = float(np.mean(epoch_losses))
                logging.info(f"[E{epoch}] Mean training loss: {mean_loss:.4f}")

        logging.info("Adapter training complete.")
        return adapter
                
    def summarize(
        self,
        llm: FrozenLLM,
        adapter: SoftPromptAdapter,
        private_reviews: list[str],
        metadata: dict,
        instruction_template: str,
        C: float,
        epsilon: float,
        delta: float,
        k: int = 1,
        no_dp: bool = False,
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ):
        """
        K-shot DP summarization of private reviews using basic composition.
        Uses add/remove adjacency.

        :param llm: Frozen LLM instance.
        :type llm: FrozenLLM

        :param adapter: Trained SoftPromptAdapter instance.
        :type adapter: SoftPromptAdapter

        :param private_reviews: List of private reviews to summarize.
        :type private_reviews: list[str]
        
        :param metadata: Product metadata dict with price_bucket, rating.
        :type metadata: dict

        :param instruction_template: Instructions for the LLM to generate summaries.
        :type instruction_template: str

        :param C: Clipping norm.
        :type C: float

        :param epsilon: Privacy budget epsilon (divided by k for basic composition).
        :type epsilon: float

        :param delta: Privacy budget delta (divided by k for basic composition).
        :type delta: float

        :param k: Number of independent queries (shots). Default 1 is standard 1-shot.
        :type k: int

        :param no_dp: If True, do not apply differential privacy.
        :type no_dp: bool

        :param max_new_tokens: Maximum number of new tokens to generate.
        :type max_new_tokens: int

        :param batch_size: Number of reviews to process in parallel.
        :type batch_size: int
        """
        adapter = adapter.to(llm.device)  # Keep adapter in float32

        n = len(private_reviews)
        soft_prompts = []

        for shot in range(k):
            # encode private reviews into clipped hidden vectors h_i
            e_list = []
            
            # Process reviews in batches
            for batch_start in range(0, len(private_reviews), batch_size):
                batch_reviews = private_reviews[batch_start:batch_start + batch_size]
                batch_prompts = [
                    utils.create_prompt(
                        r, 
                        title=metadata.get('title'), 
                        categories=metadata.get('categories')
                    ) 
                    for r in batch_reviews
                ]
                
                # Batch encode all prompts
                with torch.no_grad():
                    batch_hidden = llm.encode_batch(batch_prompts)  # (batch_size, d_model) - already pooled
                
                # Process each embedding in the batch
                batch_embeddings = []
                for e_j in batch_hidden:
                    e_j = e_j.float()  # (d_model,)
                    
                    # L2 clipping
                    norm = torch.norm(e_j, p=2)
                    if norm > C:
                        e_j = e_j * (C / norm)
                    
                    batch_embeddings.append(e_j)
                
                e_list.extend(batch_embeddings)

            e_stack = torch.stack(e_list, dim=0)             # (n, proj_dim)
            e_hat = e_stack.mean(dim=0)                      # (proj_dim,)

            if no_dp:
                if shot == 0:
                    logging.warning("DP is disabled. No noise will be added.")
                e_priv = e_hat
            else:
                # add Gaussian noise with per-shot budget
                epsilon_shot = epsilon / k
                delta_shot = delta / k
                sensitivity = 2.0 * C / n
                sigma = utils.calibrate_gaussian_sigma(epsilon_shot, delta_shot, sensitivity)
                noise = torch.normal(
                    mean=0.0,
                    std=sigma,
                    size=e_hat.shape,
                    device=e_hat.device,
                    dtype=e_hat.dtype,
                )
                e_priv = e_hat + noise
            # map to soft prompt (adapter stays float32, add batch dimension)
            soft_prompt = adapter(e_priv.unsqueeze(0))  # (1, m, d_model)
            soft_prompt = soft_prompt.squeeze(0)  # (m, d_model)
            soft_prompt = soft_prompt.to(llm.model.dtype)
            
            soft_prompts.append(soft_prompt)

        # Average soft prompts from k shots
        soft_prompt_final = torch.stack(soft_prompts, dim=0).mean(dim=0)
        soft_prompt_final = torch.clamp(soft_prompt_final, min=-20.0, max=20.0)
        
        logging.debug(f"Soft prompt stats (k={k}): mean={soft_prompt_final.mean():.4f}, std={soft_prompt_final.std():.4f}, min={soft_prompt_final.min():.4f}, max={soft_prompt_final.max():.4f}")

        # decode summary using metadata-aware instruction text
        title = metadata.get("title", "")
        categories = metadata.get("categories") or []
        categories_str = ", ".join(categories) if isinstance(categories, list) else str(categories)
        instruction_text = instruction_template.format(title, categories_str)

        instruction_ids = llm.tokenizer.encode(
            instruction_text,
            add_special_tokens=False
        )
        instruction_embeddings = llm.token_embed(instruction_ids)

        # convert soft_prompt to numpy
        soft_prompt_np = soft_prompt_final.detach().cpu().numpy()
        init_embeddings = np.concatenate([soft_prompt_np, instruction_embeddings], axis=0)

        summary = llm.decode(
            init_embeddings,
            max_new_tokens
        )

        return summary

 

