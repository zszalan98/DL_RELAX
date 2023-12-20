import torch

def update_importance(importance_mx, masks, s, batch_idx):
    
    # Apply similarity value to each masked element (higher similarity = lower importnace masked out)
    masked_similarity = s.view(-1, 1, 1) * masks
    # Updated importance matrix
    importance_mx = (batch_idx * importance_mx + torch.sum(masked_similarity, dim=0)) / (batch_idx + 1)
    # Return updated importance
    return importance_mx
    

def update_uncertainty(uncertainty_mx, masks, s, importance_mx, prev_importance_mx, batch_idx):
    # Calculate difference from mean importance values for each batch element's similarity
    broadcast_s = s.view(-1, 1, 1)
    # Approximation of the squared differences
    squared_diff = torch.abs((broadcast_s - importance_mx[None])) * torch.abs((broadcast_s - prev_importance_mx[None]))
    # Apply difference value to each masked element
    masked_diff = masks * squared_diff
    # Updated uncertainty matrix (increasing weights!)
    old_weigth = batch_idx * (batch_idx + 1) / 2
    new_weight = batch_idx + 1
    uncertainty_mx = (old_weigth * uncertainty_mx + new_weight * torch.mean(masked_diff, dim=0)) / (old_weigth + new_weight)
    # Return updated uncertainty
    return uncertainty_mx


def update_importance_mask_weighted(importance_mx, masks, s, batch_idx):
    
    # Apply similarity value to each masked element (higher similarity = lower importnace masked out)
    masked_similarity = s.view(-1, 1, 1) * masks
    new_similarity = torch.sum(masked_similarity, dim=0) / torch.sum(masks, dim=0)
    # Updated importance matrix
    importance_mx = (batch_idx * importance_mx + new_similarity) / (batch_idx + 1)
    # Return updated importance
    return importance_mx


def update_uncertainty_mask_weighted(uncertainty_mx, masks, s, importance_mx, prev_importance_mx, batch_idx):
    # Calculate difference from mean importance values for each batch element's similarity
    broadcast_s = s.view(-1, 1, 1)
    # Approximation of the squared differences
    squared_diff = torch.abs((broadcast_s - importance_mx[None])) * torch.abs((broadcast_s - prev_importance_mx[None]))
    # Apply difference value to each masked element
    masked_diff = masks * squared_diff
    new_uncertainty = torch.sum(masked_diff, dim=0) / torch.sum(masks, dim=0)
    # Updated uncertainty matrix (increasing weights!)
    old_weigth = batch_idx * (batch_idx + 1) / 2
    new_weight = batch_idx + 1
    uncertainty_mx = (old_weigth * uncertainty_mx + new_weight * new_uncertainty) / (old_weigth + new_weight)
    # Return updated uncertainty
    return uncertainty_mx


def manhattan_similarity(h_star, h_masked, p=1):
    sim =  torch.cdist(h_star, h_masked, p).squeeze()
    sim = torch.where(sim == 0, torch.tensor(1.0), sim)
    return (1 / sim).clamp(max=1.0)

