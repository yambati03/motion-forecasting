import unittest
import torch
from loss_functions import (
    PositionLoss,
    VelocityLoss,
    SmoothnessLoss,
    TerminalPositionLoss,
    DeltaLoss,
    TrajectoryLoss
)

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.timesteps = 10
        self.features = 2

        self.pred_position = torch.zeros(self.batch_size, self.timesteps, self.features)
        self.target_position = torch.ones(self.batch_size, self.timesteps, self.features)
        self.pred_velocity = torch.zeros(self.batch_size, self.timesteps, self.features)
        self.target_velocity = torch.ones(self.batch_size, self.timesteps, self.features)

    def test_position_loss(self):
        loss_fn = PositionLoss()
        loss = loss_fn(self.pred_position, self.target_position)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)

    def test_velocity_loss(self):
        loss_fn = VelocityLoss()
        loss = loss_fn(self.pred_velocity, self.target_velocity)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)

    def test_smoothness_loss(self):
        loss_fn = SmoothnessLoss()
        loss = loss_fn(self.pred_position)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_terminal_position_loss(self):
        loss_fn = TerminalPositionLoss()
        loss = loss_fn(self.pred_position, self.target_position)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)

    def test_delta_loss(self):
        loss_fn = DeltaLoss(base_loss_fn=torch.nn.MSELoss())
        loss = loss_fn(self.pred_position, self.target_position)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_trajectory_loss(self):
        loss_fn = TrajectoryLoss(
            use_velocity_loss=True,
            use_smoothness_loss=True,
            use_terminal_loss=True,
            use_delta_loss=True,
            loss_weights={
                "position_loss": 1.0,
                "velocity_loss": 0.5,
                "smoothness_loss": 0.2,
                "terminal_loss": 0.7,
                "delta_loss": 1.5
            }
        )
        predictions = torch.zeros(self.batch_size, self.timesteps, self.features)
        targets = torch.ones(self.batch_size, self.timesteps, self.features + 2)
        last_input_state = torch.zeros(self.batch_size, self.features)
        loss = loss_fn(predictions, targets, last_input_state)
        self.assertTrue(loss.item() >= 0)

if __name__ == '__main__':
    unittest.main()
