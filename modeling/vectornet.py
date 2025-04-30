import os
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from modeling.subgraph import SubGraph
from modeling.selfatten import SelfAttentionLayer
from modeling.predmlp import TrajPredMLP


class HGNN(nn.Module):
    """
    Hierarchical GNN encoder with a prediction MLP head.

    Stages:
      1) SubGraph: per-polyline GNN encoding
      2) SelfAttentionLayer: global fusion across polylines
      3) TrajPredMLP: final MLP to predict future offsets

    param in_channels: feature dimension of each node in subgraph
    param future_steps: number of future time-steps to predict (e.g. 30 for 3s @10Hz)
    param num_subgraph_layers: number of GNN layers in SubGraph
    param subgraph_hidden: hidden size for each subgraph MLP
    param global_hidden: output dimension of self-attention layer
    param mlp_hidden: hidden size of final MLP
    param need_scale: whether to scale dot-product scores in attention
    """
    def __init__(
        self,
        in_channels: int,
        future_steps: int,
        num_subgraph_layers: int = 3,
        subgraph_hidden: int = 64,
        global_hidden: int = 64,
        mlp_hidden: int = 64,
        need_scale: bool = True
    ):
        super().__init__()
        # dimension after SubGraph (doubling each layer)
        self.polyline_dim = in_channels * (2 ** num_subgraph_layers)
        # 1) local polyline encoder
        self.subgraph = SubGraph(
            in_channels=in_channels,
            num_subgraph_layers=num_subgraph_layers,
            hidden_unit=subgraph_hidden
        )
        # 2) global self-attention across polylines
        self.global_net = SelfAttentionLayer(
            in_channels=self.polyline_dim,
            out_channels=global_hidden,
            need_scale=need_scale
        )
        # 3) final trajectory predictor
        self.traj_pred = TrajPredMLP(
            in_channels=global_hidden,
            out_channels=2 * future_steps,
            hidden_unit=mlp_hidden
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        param data: PyG Data or Batch with fields:
                     - x: [N_total_nodes, in_channels]
                     - edge_index: [2, E]
                     - cluster: [N_total_nodes] global cluster IDs
                     - batch: [N_total_nodes] graph ID per node
                     - valid_len: [batch_size] # of clusters per scenario
        return: Tensor of shape [batch_size, 2*future_steps]
        """
        # # 1) Encode subgraphs (polylines) to get one vector per cluster
        # sub_out = self.subgraph(data)   # Data.x: [P, polyline_dim]
        # poly_feats = sub_out.x          # [P, polyline_dim]

        # # 2) Fuse via self-attention (ignores batch if batch_size=1)
        # #    Output shape: [P, global_hidden]
        # attn_feats = self.global_net(poly_feats, data.valid_len)

        # # 3) Extract the agent (cluster 0 of each scenario)
        # #    Here we assume batch_size=1; for multi-batch, would need to split by valid_len
        # agent_feat = attn_feats[0].unsqueeze(0)  # [1, global_hidden]

        # # 4) Predict future (x,y) offsets
        # out = self.traj_pred(agent_feat)         # [1, 2*future_steps]
        # return out
    
        # 1) Offset cluster IDs to make them unique across the batch
        #    So cluster_id = graph_id * P_max + original_cluster
        P_max = int(data.cluster.max().item()) + 1
        # print(P_max)
        data.cluster = data.batch * P_max + data.cluster

        # 2) Encode all subgraphs in one pass
        sub_out = self.subgraph(data)
        feats = sub_out.x            # [C_total, D]
        # print(feats.shape)
        batch_poly = sub_out.batch  # [C_total]
        # print(batch_poly)

        # 3) Pack into dense tensor [B, P_batch_max, D]
        #    mask (unused) simply indicates presence of each slot
        X_padded, mask = to_dense_batch(feats, batch_poly)

        # 4) Global self-attention per graph
        #    valid_len tells how many of the first slots are real
        valid = data.valid_len.view(-1)  # [B]
        attn_out = self.global_net(X_padded, valid)
        # attn_out: [B, P_batch_max, global_hidden]

        # 5) Extract the agent embedding (cluster 0) for each graph
        agent_feats = attn_out[:, 0, :]   # [B, global_hidden]

        # 6) MLP prediction for all agents in parallel
        preds = self.traj_pred(agent_feats)  # [B, 2*future_steps]
        return preds

if __name__ == "__main__":
    # # Simple test with batch_size=1
    # from torch_geometric.data import Data
    # # Example subgraph: 3 nodes, feature dim 8
    # x = torch.randn(3, 8)
    # edge_index = torch.tensor([[0,1,2,0],[1,2,0,2]])
    # cluster = torch.tensor([0,0,0])  # single polyline
    # valid_len = torch.tensor([1])    # one cluster
    # data = Data(x=x, edge_index=edge_index, cluster=cluster, valid_len=valid_len)

    # # Instantiate model: in_channels=8, future_steps=30
    # model = HGNN(8, future_steps=30)
    # pred = model(data)
    # print("Pred shape:", pred.shape)  # should print [1, 60]


    from torch_geometric.data import Data, Batch

    # Scenario 1: 3 nodes, single cluster
    x1 = torch.randn(4, 8)
    edge_index1 = torch.tensor([[0,1,2,3,0],[1,2,3,0,2]])
    cluster1 = torch.tensor([0,0,1,1])
    valid_len1 = torch.tensor([1])
    data1 = Data(x=x1, edge_index=edge_index1,
                 cluster=cluster1, valid_len=valid_len1) 

    # Scenario 2: 4 nodes, two clusters
    x2 = torch.randn(4, 8)
    edge_index2 = torch.tensor([[0,1,2,3,0],[1,2,3,0,2]])
    cluster2 = torch.tensor([0,0,1,1])
    valid_len2 = torch.tensor([1])
    data2 = Data(x=x2, edge_index=edge_index2,
                 cluster=cluster2, valid_len=valid_len2)

    # Batch them and run
    batch = Batch.from_data_list([data1, data2])
    model = HGNN(in_channels=8, future_steps=30)
    out = model(batch)
    print(f"Batch pred shape: {out.shape}  (expected [2, 60])")
