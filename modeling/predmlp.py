import torch
import torch.nn as nn
from torch import Tensor


class TrajPredMLP(nn.Module):
    """
    A simple MLP head to predict future trajectory offsets from the agent (ego) embedding.

    param in_channels: Dimension of input embedding (global_graph_width).
    param out_channels: Dimension of output (2 * future_steps).
    param hidden_unit: Hidden layer size.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_unit: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        param x: Agent embedding tensor of shape [B, in_channels] or [1, in_channels]
        return: Predicted offsets tensor of shape [B, out_channels]
        """
        return self.mlp(x)


if __name__ == "__main__":
    # Quick sanity check
    # Suppose we want to predict 12 future time steps (x,y), so out_channels = 24
    in_dim = 128
    future_steps = 12
    hidden = 64

    # Create dummy agent embedding for a batch of 2
    dummy_input = torch.randn(2, in_dim)
    model = TrajPredMLP(in_channels=in_dim, out_channels=2 * future_steps, hidden_unit=hidden)
    pred = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {pred.shape} (should be [2, {2 * future_steps}])")