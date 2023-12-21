import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MeanPooling(nn.Module):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class AttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_dim, **kwargs):
        super().__init__()
        
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, last_hidden_state, attention_mask):
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(last_hidden_state))
        score = self.V(att)
        score[attention_mask==0]=-1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights*weights_mask*last_hidden_state, dim=1)
        return context_vector
    

POOLING_MAPPING = {
    "mean": MeanPooling,
    "attention": AttentionPooling,
}
