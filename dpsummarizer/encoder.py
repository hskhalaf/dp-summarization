"""Learnable encoder that pools review hidden states into a fixed-dimensional vector."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReviewEncoder(nn.Module):
    def __init__(self, d_model: int, proj_dim: int = 256, num_categories: int = 10):
        """
        Transformer-based encoder that preserves sequence information.
        Uses self-attention to adaptively aggregate information rather than simple pooling.
        
        This design avoids aggressive information bottlenecks by:
        1. Using multi-head self-attention over the full sequence
        2. Preserving position-specific information
        3. Learning what information to aggregate for each product type
        
        :param d_model: Dimension of the LLM hidden states.
        :type d_model: int
        
        :param proj_dim: Dimension of the projection layer output.
        :type proj_dim: int
        """
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.num_categories = num_categories
        
        # Self-attention layers to process sequence structure
        # This allows the model to learn which tokens are important per domain
        self.attention_layer1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Second attention layer for deeper processing
        self.attention_layer2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.layer_norm4 = nn.LayerNorm(d_model)
        
        # Learn a learnable token for aggregation (like [CLS] token in BERT)
        self.aggregation_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Metadata embeddings (price, rating) - stronger influence retained
        metadata_dim = d_model // 2  # Larger metadata embeddings for differentiation
        self.price_embedding = nn.Embedding(3, metadata_dim)  # 0=budget, 1=mid, 2=premium
        self.rating_proj = nn.Linear(1, metadata_dim)
        
        # Metadata fusion to d_model dimensions
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Projection to output dimension
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        
        # Concat fusion layers (review + metadata) - much stronger fusion
        # Key insight: use larger hidden dimension to preserve metadata signal
        self.fusion_layers = nn.Sequential(
            nn.Linear(proj_dim + d_model, proj_dim * 4),  # Larger hidden layer
            nn.ReLU(),
            nn.LayerNorm(proj_dim * 4),
            nn.Dropout(0.2),
            nn.Linear(proj_dim * 4, proj_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
        )

    def forward(self, hidden_states: torch.Tensor,
                price_bucket: Optional[int] = None, rating: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass using transformer attention to preserve sequence information.

        :param hidden_states: Hidden states of shape (seq_len, d_model).
        :type hidden_states: torch.Tensor
        :param price_bucket: Price bucket (0=budget, 1=mid, 2=premium).
        :param rating: Product rating (0.0-5.0).

        :return: Encoded review representation of shape (proj_dim,).
        :rtype: torch.Tensor
        """
        # Add batch dimension for attention layers
        x = hidden_states.unsqueeze(0)  # (1, seq_len, d_model)
        
        # First attention layer with residual connection
        attn_out, _ = self.attention_layer1(x, x, x)
        x = self.layer_norm1(x + attn_out)
        ffn_out = self.ffn1(x)
        x = self.layer_norm2(x + ffn_out)
        
        # Second attention layer
        attn_out, _ = self.attention_layer2(x, x, x)
        x = self.layer_norm3(x + attn_out)
        ffn_out = self.ffn2(x)
        x = self.layer_norm4(x + ffn_out)  # (1, seq_len, d_model)
        
        # Use learnable aggregation token that attends to all positions
        batch_size = x.size(0)
        agg_token = self.aggregation_token.expand(batch_size, -1, -1)  # (1, 1, d_model)
        
        # Aggregate by cross-attention of aggregation token to sequence
        aggregated, _ = self.attention_layer1(agg_token, x, x)  # (1, 1, d_model)
        aggregated = aggregated.squeeze(1)  # (1, d_model)
        
        # Project review
        review_emb = self.proj(aggregated).squeeze(0)  # (proj_dim,)
        
        # If metadata provided, use concat fusion
        if price_bucket is not None and rating is not None:
            device = hidden_states.device
            
            # Embed metadata
            price_emb = self.price_embedding(torch.tensor(price_bucket, device=device))
            rating_tensor = torch.tensor([[rating]], device=device, dtype=torch.float32)
            rating_emb = self.rating_proj(rating_tensor).squeeze(0)
            
            # Combine metadata
            metadata_combined = torch.cat([price_emb, rating_emb], dim=-1)
            metadata_emb = self.metadata_proj(metadata_combined)  # (d_model,)
            
            # Concat fusion
            combined = torch.cat([review_emb, metadata_emb], dim=-1)
            output = self.fusion_layers(combined)
            return output
        
        return review_emb

