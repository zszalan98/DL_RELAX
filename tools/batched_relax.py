import torch

def update_importance(importance_mx, masks, s):
    # Apply similarity value to each masked element (higher similarity = lower importnace masked out)
    masked_similarity = masks * (1 - s.view(-1, 1, 1))
    # Add batch mean to importance matrix
    return importance_mx + torch.mean(masked_similarity, dim=0)

    

def update_uncertainty(uncertainty_mx, masks, importance_mx, s):
    # Calculate difference from mean importance values for each batch element's similarity
    diff = (s.view(-1, 1, 1) - importance_mx[None]) ** 2
    # Apply difference value to each masked element
    masked_diff = masks * diff
    # Add batch mean to uncertainty matrix
    return uncertainty_mx + torch.mean(masked_diff, dim=0)

def manhattan_similarity(h_star, h_masked, p=1):
    return (1 / torch.cdist(h_star, h_masked, p).squeeze()).clamp(max=1.0)

