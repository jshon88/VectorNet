from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, max_pool

def normalize_per_feature(tensor: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize a tensor per feature (column-wise).
    """
    norm = tensor.norm(p=2, dim=0, keepdim=True)
    # Avoid division by zero
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return tensor / norm


class GraphLayerProp(MessagePassing):
    """
    A single message-passing layer that updates node features by:
      1. Applying an MLP to each node's own features.
      2. Max-pooling over its neighbors' features.
      3. Concatenating the self and aggregated features.
    Each forward pass increases the feature dimension by a factor of 2.
    """

    def __init__(self, in_channels: int, hidden_unit: int = 64, verbose: bool = False):
        super().__init__(aggr='max')  # use max aggregation for neighbor messages
        self.verbose = verbose
        # MLP to transform node's own features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        param x: Node feature matrix [N, in_channels]
        param edge_index: Edge indices [2, E]
        return: Updated node features [N, in_channels * 2]
        """
        if self.verbose:
            print("[GraphLayerProp] input x shape:", x.shape)
        # 1) Transform self features
        x_transformed = self.mlp(x)  # [N, in_channels]
        # 2) Propagate to aggregate neighbor features
        aggr_out = self.propagate(edge_index, x=x_transformed)
        if self.verbose:
            print("[GraphLayerProp] after MLP x:", x_transformed.shape)
            print("[GraphLayerProp] aggregated neighbors:", aggr_out.shape)
        # 3) Concatenate self and neighbor features
        out = torch.cat([x_transformed, aggr_out], dim=1)
        if self.verbose:
            print("[GraphLayerProp] output shape:", out.shape)
        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """
        param x_j: Features of neighbor nodes
        return: Messages to aggregate (identity here)
        """
        return x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        This function is automatically called from self.propagate
        param aggr_out: Aggregated neighbor features [N, in_channels]
        return: aggr_out
        """
        return aggr_out


class SubGraph(nn.Module):
    """
    Encodes each polyline (subgraph) by stacking multiple
    GraphLayerProp layers and pooling node features into
    a single polyline embedding.
    """

    def __init__(self, in_channels: int, num_subgraph_layers: int = 3, hidden_unit: int = 64, verbose=False):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_channels
        for _ in range(num_subgraph_layers):
            self.layers.append(GraphLayerProp(current_dim, hidden_unit, verbose=verbose))
            current_dim *= 2  # each layer doubles the feature dim

    def forward(self, data: Data) -> Data:
        """
        param data: PyG Data object with fields:
                     - x: [N_nodes, in_channels]
                     - edge_index: [2, E]
                     - cluster: [N_nodes] polyline IDs (0..P-1)
        return: Data with x: [P, feature_dim], where P = # polylines.
        """
        x, edge_index, cluster = data.x, data.edge_index, data.cluster
        # 1) Sequentially apply each GraphLayerProp
        for layer in self.layers:
            x = layer(x, edge_index)
        # 2) Pool nodes per polyline
        data.x = x
        pooled = max_pool(cluster, data)
        # 3) Normalize per feature to unit norm
        pooled.x = normalize_per_feature(pooled.x)
        return pooled


if __name__ == "__main__":
    # Quick sanity check
    from torch_geometric.data import Data
    x = torch.tensor([[1.0], [7.0], [3.0]])
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    cluster = torch.tensor([0, 0, 0])  # one polyline
    data = Data(x=x, edge_index=edge_index, cluster=cluster)
    net = SubGraph(in_channels=1, num_subgraph_layers=2, hidden_unit=4, verbose=True)
    out = net(data)
    print("Output embedding shape:", out.x.shape)