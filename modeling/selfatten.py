import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(scores: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    """
    Perform softmax on the last dimension of `scores`, masking out positions
    beyond the sequence length specified by `valid_len`.

    param scores: Tensor of shape [B, P, P]
    param valid_len: 1D tensor of shape [batch_size] indicating the valid length for each sequence (<= seq_len)
    return: Tensor of same shape as `scores` with softmax applied
             over the last dim, respecting the mask.
    """
    batch_size, seq_len, _ = scores.size()
    # Ensure valid_len is a tensor of shape [batch_size]
    if valid_len.dim() == 0:
        valid_len = valid_len.repeat(batch_size)
    # Create mask: True for valid positions, False for padded
    mask = torch.arange(seq_len, device=scores.device)[None, :] < valid_len[:, None]
    # Expand mask to [batch_size, 1, seq_len] for broadcasting over queries
    mask = mask.unsqueeze(1)  # shape [batch_size, 1, seq_len]
    # Fill masked positions with -inf before softmax
    scores = scores.masked_fill(~mask, float('-inf'))
    # print(scores)
    # Compute softmax
    return F.softmax(scores, dim=-1)


class SelfAttentionLayer(nn.Module):
    """
    Single-head self-attention layer for fusing polyline embeddings.

    Given input x of shape [B, P, in_channels] (P polylines), and a valid_len
    indicating how many of those P are real (rest are padding), this layer
    computes:
        Q = x W_q, K = x W_k, V = x W_v
    then attention weights = softmax(Q K^T / scale) with masking,
    and output = attention_weights @ V.

    param in_channels: Dimension of input embeddings.
    param out_channels: Dimension of attention outputs (per head).
    param need_scale: Whether to scale scores by sqrt(d_k).
    """
    def __init__(self, in_channels: int, out_channels: int, need_scale: bool = True):
        super().__init__()
        self.q_lin = nn.Linear(in_channels, out_channels)
        self.k_lin = nn.Linear(in_channels, out_channels)
        self.v_lin = nn.Linear(in_channels, out_channels)
        self.scale = (out_channels ** 0.5) if need_scale else 1.0

    def forward(self, x: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        """
        param x: Tensor of shape [P, in_channels]
        param valid_len: Tensor of shape [1] or [batch_size]
        return: Tensor of shape [P, out_channels]
        """
        # Add batch dimension if missing
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # -> [1, P, in_channels]
            single_batch = True
        # Project to queries, keys, values
        Q = self.q_lin(x)  # [B, P, D]
        K = self.k_lin(x)  # [B, P, D]
        V = self.v_lin(x)  # [B, P, D]
        # Compute scaled dot-product attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, P, P]
        # Mask and normalize
        attn = masked_softmax(scores, valid_len)              # [B, P, P]
        # Weighted sum of values
        out = torch.bmm(attn, V)  # [B, P, D]
        # Remove batch dim if originally single
        if single_batch:
            out = out.squeeze(0)   # -> [P, D]
        return out


if __name__ == "__main__":
    # Sanity check for a tiny example
    P, C = 5, 8
    x = torch.randn(P, C)
    valid_len = torch.tensor([3])  # only first 3 polylines are real
    layer = SelfAttentionLayer(in_channels=C, out_channels=16)
    out = layer(x, valid_len)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
