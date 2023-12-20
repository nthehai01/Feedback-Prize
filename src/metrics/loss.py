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

    def forward(self, preds: Tensor, refs: Tensor) -> Tensor:
        weights = torch.ones_like(preds) if self.weights is None else self.weights
        weights = weights.to(preds.device)
        
        loss = F.mse_loss(preds, refs, reduction="none")
        loss = torch.sqrt(loss + self.eps)
        loss = loss * weights

        loss = loss.sum() / weights.sum()
        
        return loss
