import numpy as np
from .log import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class FrozenLLM:
    def __init__(self, model_name: str, device: str | None = None):
        """
        :param model_name: Name of the pre-trained LLM model.
        :type model_name: str
        
        :param device: Device to run the model on. (e.g., "cpu", "cuda")
        :type device: str | None
        """
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.device = torch.device(device)
        logging.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.d_model = self.model.config.hidden_size

        # freeze weights
        for p in self.model.parameters():
            p.requires_grad = False

        logging.info(f"Model loaded: {model_name}")
        logging.debug(f"hidden_size={self.d_model}")
        logging.debug(f"vocab_size={len(self.tokenizer)}")


    def encode(self, text: str) -> torch.Tensor:
        """
        Run a forward pass through the frozen LLM and return the hidden states.

        :param text: Input text string.
        :type text: str
        
        :return: Hidden states of shape (seq_len, d_model)
        :rtype: torch.Tensor
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            # last hidden layer, remove batch dim
            hidden = outputs.hidden_states[-1][0] # (seq_len, d_model)

        return hidden
    
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Run a forward pass and return mean-pooled hidden states for each text.

        :param texts: List of input text strings.
        :return: Tensor of shape (batch_size, d_model)
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]              # (batch, seq_len, d_model)

            # Use attention_mask to compute a masked mean over non-padding tokens
            attn_mask = inputs.attention_mask               # (batch, seq_len)
            mask = attn_mask.unsqueeze(-1)                  # (batch, seq_len, 1)

            # Zero-out padded positions, then sum and divide by lengths
            hidden_masked = hidden * mask                   # (batch, seq_len, d_model)
            lengths = mask.sum(dim=1).clamp(min=1)          # (batch, 1)
            pooled = hidden_masked.sum(dim=1) / lengths     # (batch, d_model)

        return pooled  # already on self.device

    def decode(
        self, 
        init_embeddings: np.ndarray, 
        max_new_tokens: int = 64, 
        **decode_kwargs
    ) -> str:
        """
        Generate a string of text given initial embeddings.
        
        :param init_embeddings: Initial embeddings to start generation from.
        :type init_embeddings: np.ndarray

        :param max_new_tokens: Maximum number of new tokens to generate.
        :type max_new_tokens: int

        :return: Generated text.
        :rtype: str
        """
        inputs_embeds = torch.tensor(
            init_embeddings,
            dtype=self.model.dtype,
            device=self.device,
        ).unsqueeze(0)  # (1, seq_len, d_model)

        attention_mask = torch.ones(
            inputs_embeds.size()[:2],
            dtype=torch.long,
            device=self.device
        )

        # prompt_len = inputs_embeds.size(1)

        with torch.no_grad():
            generated = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **decode_kwargs
            )

        gen_ids = generated[0]
        decoded = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True
        )
        return decoded
    
    def token_embed(self, token_ids: list[int]) -> np.ndarray:
        """
        Map token IDs to embeddings.
        
        :param token_ids: Input token IDs.
        :type token_ids: list[int]
        
        :return: Token embeddings of shape (seq_len, d_model)
        :rtype: np.ndarray
        """
        input_ids = torch.tensor(
            token_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0) 

        with torch.no_grad():
            embeddings_layer = self.model.get_input_embeddings()
            embeddings = embeddings_layer(input_ids)[0] # (seq_len, d_model)

        return embeddings.detach().cpu().numpy()
    
    def lm_loss(
        self, 
        soft_prefix: torch.Tensor, 
        instruction_ids: list[int], 
        target_ids: list[int]
    ) -> torch.Tensor:
        """
        Computes the language modeling loss for target_ids.

        We can't wrap this in torch.no_grad() since we need gradients to flow
        through the soft prompt adapter.

        :param soft_prefix: Soft prompt embeddings of shape (m, d_model) with requires_grad=True.
        :type soft_prefix: torch.Tensor

        :param instruction_ids: Instruction token IDs.
        :type instruction_ids: list[int]

        :param target_ids: Target token IDs.
        :type target_ids: list[int]

        :return: Language modeling loss.
        :rtype: torch.Tensor
        """
        assert isinstance(soft_prefix, torch.Tensor)

        soft_prefix = soft_prefix.unsqueeze(0)  # (1, m, d_model)
        soft_prefix = soft_prefix.to(self.model.dtype)

        # convert inputs to embeddings
        instr_ids = torch.tensor(
            instruction_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        trgt_ids  = torch.tensor(
            target_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        
        embeddings_layer = self.model.get_input_embeddings()
        token_embeddings = embeddings_layer(torch.cat([
            instr_ids, 
            trgt_ids
        ], dim=1))  # (1, L, d_model)

        # concatenate soft prompt embeddings
        input_embeds = torch.cat([
            soft_prefix,         
            token_embeddings         
        ], dim=1)   # (1, m + L, d_model)
        seq_len = input_embeds.size(1)

        # build labels, same length as inputs but only the last |target_ids| 
        # positions get real labels, rest are -100 to ignore
        labels = torch.full(
            (1, seq_len),
            -100,
            dtype=torch.long,
            device=self.device
        )
        labels[0, -trgt_ids.size(1):] = trgt_ids

        # forward pass
        outputs = self.model(
            inputs_embeds=input_embeds,
            labels=labels,
            use_cache=False,
        )
        
        return outputs.loss