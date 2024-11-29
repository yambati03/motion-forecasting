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
        return F.mse_loss(pred_position[:, -1, :], target_position[:, -1, :]) + \
               F.mse_loss(pred_position[:, 0, :], target_position[:, 0, :])

# Trajectory loss (combination of different components)
class TrajectoryLoss(BaseLoss):
    def __init__(self, use_velocity_loss=True, use_smoothness_loss=True, use_terminal_loss=True):
        super(TrajectoryLoss, self).__init__()
        self.position_loss = PositionLoss()
        self.velocity_loss = VelocityLoss() if use_velocity_loss else None
        self.smoothness_loss = SmoothnessLoss() if use_smoothness_loss else None
        self.terminal_loss = TerminalPositionLoss() if use_terminal_loss else None

    def forward(self, predictions, targets, last_input_state, dt=0.1):
        # Reconstruct trajectory from velocities
        pred_velocity = predictions
        cum_disp = torch.cumsum(pred_velocity * dt, dim=1)
        pred_position = cum_disp + last_input_state.unsqueeze(1)

        target_position = targets[:, :, :2]
        target_velocity = targets[:, :, 2:4]

        # Calculate losses
        position_loss = self.position_loss(pred_position, target_position)
        velocity_loss = self.velocity_loss(pred_velocity, target_velocity) if self.velocity_loss else 0
        smoothness_loss = self.smoothness_loss(pred_position) if self.smoothness_loss else 0
        terminal_loss = self.terminal_loss(pred_position, target_position) if self.terminal_loss else 0

        # Combine losses
        total_loss = position_loss + velocity_loss + smoothness_loss + terminal_loss
        return total_loss

# Loss registry for dynamic selection
LOSS_REGISTRY = {
    "position_loss": PositionLoss,
    "velocity_loss": VelocityLoss,
    "smoothness_loss": SmoothnessLoss,
    "terminal_position_loss": TerminalPositionLoss,
    "trajectory_loss": TrajectoryLoss,
}

def get_loss_function(loss_name, **kwargs):
    if loss_name in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_name](**kwargs)
    else:
        raise ValueError(f"Loss function '{loss_name}' not found in registry.")
