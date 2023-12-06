import torch

def update_importance(importance_mx, masks, s):
    # Apply similarity value to each masked element (higher similarity = lower importnace masked out)
    masked_similarity = masks * (1 - s.view(-1, 1, 1))
    # Add batch mean to importance matrix
    return importance_mx + torch.mean(masked_similarity, dim=0)

    

def update_uncertainty(uncertainty_mx, masks, s, importance_mx, prev_importance_mx):
    # Calculate difference from mean importance values for each batch element's similarity
    broadcast_s = (1 - s.view(-1, 1, 1))
    diff = (broadcast_s - importance_mx[None]) * (broadcast_s - prev_importance_mx[None])
    # Apply difference value to each masked element
    masked_diff = masks * diff
    # Add batch mean to uncertainty matrix
    return uncertainty_mx + torch.mean(masked_diff, dim=0)

def manhattan_similarity(h_star, h_masked, p=1):
    sim =  torch.cdist(h_star, h_masked, p).squeeze()
    sim = torch.where(sim == 0, torch.tensor(1.0), sim)
    return (1 / sim).clamp(max=1.0)

