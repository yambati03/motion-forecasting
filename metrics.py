import torch

def compute_predicted_positions(model, x, y, dt=0.1):
    outputs = model(x)
    cum_disp = torch.cumsum(outputs * dt, dim=1)
    return cum_disp + y[:, 0, :2].unsqueeze(1)

def compute_mean_ade(pred_position, y):
    distances = torch.norm(pred_position - y[:, :, :2], dim=2)
    return distances.mean().item()

def compute_fde(pred_position, y):
    return torch.norm(pred_position[:, -1, :] - y[:, -1, :2], dim=1).mean().item()

def compute_directional_accuracy(pred_position, y):
    pred_directions = pred_position[:, 1:, :] - pred_position[:, :-1, :]
    target_directions = y[:, 1:, :2] - y[:, :-1, :2]

    pred_norm = torch.norm(pred_directions, dim=2, keepdim=True).clamp(min=1e-8) # clamp to prevent divide by zero
    target_norm = torch.norm(target_directions, dim=2, keepdim=True).clamp(min=1e-8)

    pred_unit = pred_directions / pred_norm
    target_unit = target_directions / target_norm

    cosine_similarity = (pred_unit * target_unit).sum(dim=2)
    return cosine_similarity.mean().item()

def compute_linear_wade(pred_position, y):
    distances = torch.norm(pred_position - y[:, :, :2], dim=2)

    # Linear time weights (higher weights for earlier timesteps)
    time_weights = torch.linspace(pred_position.shape[1], 1, pred_position.shape[1]).float().to(pred_position.device)
    time_weights = time_weights / time_weights.sum() 

    return (distances * time_weights).mean().item()

def compute_exponential_wade(pred_position, y, alpha=0.1):

    distances = torch.norm(pred_position - y[:, :, :2], dim=2)

    # Exponential time weights (higher weights for earlier timesteps)
    time_weights = torch.exp(torch.linspace(alpha * (pred_position.shape[1] - 1), 0, pred_position.shape[1])).float()
    time_weights = time_weights.to(pred_position.device)
    time_weights = time_weights / time_weights.sum()

    return (distances * time_weights).mean().item()