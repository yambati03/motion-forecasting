import unittest
import torch
from metrics import (
    compute_predicted_positions,
    compute_mean_ade,
    compute_fde,
    compute_directional_accuracy,
    compute_linear_wade,
    compute_exponential_wade
)

class DummyModel:
    def __call__(self, x):
        return torch.ones(x.shape[0], 10, 2) # all ones

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.timesteps = 10
        self.features = 2

        self.targets = torch.tensor([
            [[i, i] for i in range(self.timesteps)],
            [[i * 2, i * 2] for i in range(self.timesteps)]
        ], dtype=torch.float32)

        self.predictions = self.targets + 0.1
        self.x = torch.zeros(self.batch_size, 80, 5)
        self.y = self.targets

    def test_compute_mean_ade(self):
        ade = compute_mean_ade(self.predictions, self.targets)
        self.assertAlmostEqual(ade, 0.141421, places=3)

    def test_compute_fde(self):
        fde = compute_fde(self.predictions, self.targets)
        self.assertAlmostEqual(fde, 0.141421, places=3)

    def test_compute_directional_accuracy(self):
        directional_accuracy = compute_directional_accuracy(self.predictions, self.targets)
        self.assertAlmostEqual(directional_accuracy, 1.0, places=3)

    def test_compute_linear_wade(self):
        wade = compute_linear_wade(self.predictions, self.targets)
        weights = torch.linspace(self.timesteps, 1, self.timesteps).float()
        weights /= weights.sum()
        expected_wade = (weights * 0.141421).mean().item()
        self.assertAlmostEqual(wade, expected_wade, places=3)

    def test_compute_exponential_wade(self):
        wade = compute_exponential_wade(self.predictions, self.targets, alpha=0.1)
        weights = torch.exp(torch.linspace(0, -0.1 * (self.timesteps - 1), self.timesteps))
        weights /= weights.sum()
        expected_wade = (weights * 0.141421).mean().item()
        self.assertAlmostEqual(wade, expected_wade, places=3)

    def test_compute_predicted_positions(self):
        dummy_model = DummyModel()
        pred_position = compute_predicted_positions(dummy_model, self.x, self.y)
        self.assertEqual(pred_position.shape, (self.batch_size, self.timesteps, self.features))

if __name__ == '__main__':
    unittest.main()
