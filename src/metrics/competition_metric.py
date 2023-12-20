import torch


def mcrmse(preds, refs):
    colwise_mse = torch.mean(torch.square(refs - preds), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse + 1e-9), dim=0)
    
    return loss
