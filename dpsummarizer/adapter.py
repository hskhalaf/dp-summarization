import numpy as np
import torch
import torch.nn as nn

class SoftPromptAdapter(nn.Module):
    def __init__(self, d_model: int, m: int, hidden_dim: int = 128):
        """
        :param d_model: Dimension of the LLM hidden states.
        :type d_model: int
        
        :param m: Number of soft prompt tokens.
        :type m: int
        
        :param hidden_dim: Dimension of the hidden layer in the adapter MLP.
        :type hidden_dim: int
        """
        super().__init__()
        self.m = m
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m * d_model)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate soft prompt embeddings.
        
        :param h: Input hidden representation of shape (d_model,) or (batch_size, d_model).
        :type h: torch.Tensor

        :return: Soft prompt embeddings of shape (m, d_model) or (batch_size, m, d_model).
        :rtype: torch.Tensor
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)  # (1, d_model)
        out = self.net(h)       # (batch_size, m * d_model)
        out = out.view(-1, self.m, self.d_model)  # (batch_size, m, d_model)
        return out.squeeze(0)  # (m, d_model) if batch_size == 1 else (batch_size, m, d_model)

            
    def compute_grads(
        self, 
        loss: torch.Tensor, 
        soft_prompt: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients of loss w.r.t. adapter parameters via backpropagation.

        :param loss: Scalar loss tensor.
        :type loss: torch.Tensor

        :param soft_prompt: Soft prompt embeddings tensor of shape (m, d_model).
        :type soft_prompt: torch.Tensor

        :return: Gradients for W1, b1, W2, b2
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if soft_prompt.grad is not None:
            soft_prompt.grad.zero_()

        loss.backward()

        dsoft = soft_prompt.grad.detach().cpu().numpy()    # (m, d_model)

        return self.backward(dsoft)
