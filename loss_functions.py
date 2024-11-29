import torch
import torch.nn as nn
import torch.nn.functional as F

# Base class for loss functions
class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

# Position loss
class PositionLoss(BaseLoss):
    def __init__(self):
        super(PositionLoss, self).__init__()

    def forward(self, pred_position, target_position):
        return F.mse_loss(pred_position, target_position)

# Velocity loss
class VelocityLoss(BaseLoss):
    def __init__(self):
        super(VelocityLoss, self).__init__()

    def forward(self, pred_velocity, target_velocity):
        return F.mse_loss(pred_velocity, target_velocity)

# Smoothness loss
class SmoothnessLoss(BaseLoss):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, pred_position):
        diff = pred_position[:, :-1, :] - pred_position[:, 1:, :]
        smoothness = torch.norm(diff, dim=2)
        return torch.mean(smoothness)

# Terminal position loss
class TerminalPositionLoss(BaseLoss):
    def __init__(self):
        super(TerminalPositionLoss, self).__init__()

    def forward(self, pred_position, target_position):
        return F.mse_loss(pred_position[:, -1, :], target_position[:, -1, :])
    
class DeltaLoss(BaseLoss):
    def __init__(self, base_loss_fn=nn.MSELoss()):

        super(DeltaLoss, self).__init__()
        self.base_loss_fn = base_loss_fn

    def forward(self, predictions, targets):

        pred_deltas = predictions[:, 1:, :] - predictions[:, :-1, :]  
        target_deltas = targets[:, 1:, :] - targets[:, :-1, :]

        # Compute the loss on the deltas
        delta_loss = self.base_loss_fn(pred_deltas, target_deltas)
        return delta_loss
    
# Trajectory loss (combination of different components)
class TrajectoryLoss(nn.Module):
    def __init__(self, 
                 use_velocity_loss=False, 
                 use_smoothness_loss=False, 
                 use_terminal_loss=False, 
                 time_weighting_scheme=None,
                 use_delta_loss=False, 
                 prediction_horizon=10,
                 loss_weights=None):
        super(TrajectoryLoss, self).__init__()
        self.position_loss = PositionLoss()
        self.velocity_loss = VelocityLoss() if use_velocity_loss else None
        self.smoothness_loss = SmoothnessLoss() if use_smoothness_loss else None
        self.terminal_loss = TerminalPositionLoss() if use_terminal_loss else None
        self.delta_loss_fn = DeltaLoss(base_loss_fn=nn.MSELoss()) if use_delta_loss else None
        self.time_weighting_scheme = time_weighting_scheme
        self.prediction_horizon = prediction_horizon

        #Initialize loss weights - defaults to all weights = 1.0
        self.loss_weights = loss_weights if loss_weights else {
            "position_loss": 1.0,
            "velocity_loss": 1.0,
            "smoothness_loss": 1.0,
            "terminal_loss": 1.0,
            "delta_loss": 1.0
        }
        # Precompute time weights if applicable
        self.time_weights = None
        if time_weighting_scheme == "Linear":
            self.time_weights = torch.linspace(prediction_horizon, 1, prediction_horizon).float() / prediction_horizon # [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        elif time_weighting_scheme == "Exponential":
            alpha = 0.5  # Adjust alpha to control exponential growth
            self.time_weights = torch.exp(
                torch.linspace(alpha * (prediction_horizon - 1), 0, prediction_horizon) # [2.459, 2.225, 2.013, 1.822, 1.648, 1.491, 1.349, 1.221, 1.105, 1.0] (alpha 0.1)
            )

    def apply_time_weights(self, loss, device):
        if self.time_weights is None:
            return loss
        weights = self.time_weights.to(device).unsqueeze(0).unsqueeze(2)
        return loss * weights

    def forward(self, predictions, targets, last_input_state, dt=0.1):

        # Reconstruct trajectory from velocities
        pred_velocity = predictions
        cum_disp = torch.cumsum(pred_velocity * dt, dim=1)
        pred_position = cum_disp + last_input_state.unsqueeze(1)

        target_position = targets[:, :, :2]
        target_velocity = targets[:, :, 2:4]

        # Calculate individual loss components
        position_loss = F.mse_loss(pred_position, target_position, reduction='none')
        velocity_loss = F.mse_loss(pred_velocity, target_velocity, reduction='none') if self.velocity_loss else 0

        # Apply time weights if applicable
        device = predictions.device
        position_loss = self.apply_time_weights(position_loss, device)
        if self.velocity_loss:
            velocity_loss = self.apply_time_weights(velocity_loss, device)

        total_position_loss = position_loss.mean() * self.loss_weights["position_loss"]
        total_velocity_loss = velocity_loss.mean() * self.loss_weights["velocity_loss"] if self.velocity_loss else 0

        smoothness_loss = (self.smoothness_loss(pred_position) * self.loss_weights["smoothness_loss"]) if self.smoothness_loss else 0
        terminal_loss = (self.terminal_loss(pred_position, target_position) * self.loss_weights["terminal_loss"]) if self.terminal_loss else 0
        delta_loss = (self.delta_loss_fn(pred_position, target_position) * self.loss_weights["delta_loss"]) if self.delta_loss_fn else 0
        
        # print(f"Position Loss: {total_position_loss:.4f}")
        # print(f"Velocity Loss: {total_velocity_loss:.4f}")
        # print(f"Smoothness Loss: {smoothness_loss:.4f}")
        # print(f"Terminal Loss: {terminal_loss:.4f}")
        # print(f"Delta Loss: {delta_loss:.4f}")

        # Combine all losses
        total_loss = (
            total_position_loss 
            + total_velocity_loss 
            + smoothness_loss 
            + terminal_loss 
            + delta_loss
        )
        return total_loss

# Loss registry for dynamic selection (Add new loss functions here as created)
LOSS_REGISTRY = {
    "position_loss": PositionLoss,
    "velocity_loss": VelocityLoss,
    "smoothness_loss": SmoothnessLoss,
    "terminal_position_loss": TerminalPositionLoss,
    "delta_loss": DeltaLoss,
    "trajectory_loss": TrajectoryLoss,
}

def get_loss_function(loss_name, **kwargs):
    if loss_name in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_name](**kwargs)
    else:
        raise ValueError(f"Loss function '{loss_name}' not found in registry.")