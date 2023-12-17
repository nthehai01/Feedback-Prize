import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


class FeedbackLoss(nn.Module):
    def __init__(self, 
                 weights: Optional[Tensor] = None) -> None:
        super().__init__()
        self.eps = 1e-9
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weights = torch.ones_like(input) if self.weights is None else self.weights
        weights = weights.to(input.device)
        
        loss = F.mse_loss(input, target, reduction="none")
        loss = torch.sqrt(loss + self.eps)
        loss = loss * weights

        loss = loss.sum() / weights.sum()
        
        return loss
