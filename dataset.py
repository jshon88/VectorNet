# %%

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR
from tqdm import tqdm


# %%
def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((np.hstack([from_[:i], from_[i+1:]]), np.hstack([to_[:i], to_[i+1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start
# %%


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# %%


class GraphDataset(InMemoryDataset):
    """
    dataset object similar to `torchvision`.
    Creates Graph Dataset
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # return ['dataset.pt']
        if not os.path.isdir(self.processed_dir):
            return []
        
        files = sorted(os.listdir(self.processed_dir))
        return [f for f in files if f.startswith('dataset') and f.endswith('.pt')] 

    def download(self):
        pass

    def process(self):

        def get_data_path_ls(dir_):
            return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]
        
        # make sure deterministic results
        data_path_ls = sorted(get_data_path_ls(self.root))

        valid_len_ls = []
        data_ls = []
        for data_p in tqdm(data_path_ls):
            if not data_p.endswith('pkl'):
                continue
            x_ls = []
            y = None
            cluster = None
            edge_index_ls = []
            data = pd.read_pickle(data_p)
            all_in_features = data['POLYLINE_FEATURES'].values[0]
            add_len = data['TARJ_LEN'].values[0]
            cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
            valid_len_ls.append(cluster.max()) # cluster.max() is basically # of unique polyline_id
            y = data['GT'].values[0].reshape(-1).astype(np.float32)

            traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
            agent_id = 0
            edge_index_start = 0
            assert all_in_features[agent_id][
                -1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

            for id_, mask_ in traj_mask.items():
                data_ = all_in_features[mask_[0]:mask_[1]]
                edge_index_, edge_index_start = get_fc_edge_index(
                    data_.shape[0], start=edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)

            for id_, mask_ in lane_mask.items():
                data_ = all_in_features[mask_[0]+add_len: mask_[1]+add_len]
                edge_index_, edge_index_start = get_fc_edge_index(
                    data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)
            edge_index = np.hstack(edge_index_ls)
            x = np.vstack(x_ls)
            data_ls.append([x, y, cluster, edge_index])

        # # [x, y, cluster, edge_index, valid_len]
        # g_ls = []
        # padd_to_index = np.max(valid_len_ls)
        # feature_len = data_ls[0][0].shape[1]
        # for ind, tup in enumerate(data_ls): # padding max # objects among all scenarios to each scenario
        #     tup[0] = np.vstack(
        #         [tup[0], np.zeros((padd_to_index - tup[-2].max(), feature_len), dtype=tup[0].dtype)])
        #     tup[-2] = np.hstack(
        #         [tup[2], np.arange(tup[-2].max()+1, padd_to_index+1)])
        #     g_data = GraphData(
        #         x=torch.from_numpy(tup[0]),
        #         y=torch.from_numpy(tup[1]),
        #         cluster=torch.from_numpy(tup[2]),
        #         edge_index=torch.from_numpy(tup[3]),
        #         valid_len=torch.tensor([valid_len_ls[ind]]),
        #         time_step_len=torch.tensor([padd_to_index + 1])
        #     )
        #     g_ls.append(g_data)
        # data, slices = self.collate(g_ls)
        # torch.save((data, slices), self.processed_paths[0])

        # [x, y, cluster, edge_index, valid_len]
        shard_size = 10000
        padd_to_index = np.max(valid_len_ls)
        feature_len = data_ls[0][0].shape[1]
        for shard_id in range(0, len(data_ls), shard_size):
            chunk_data   = data_ls[shard_id : shard_id + shard_size]
            chunk_valid  = valid_len_ls[shard_id : shard_id + shard_size]
            g_ls = []
            for (x_np, y_np, cl_np, ei_np), valid_len in zip(chunk_data, chunk_valid):
                pad_count = padd_to_index - cl_np.max()
                if pad_count > 0:
                    x_np = np.vstack([x_np, np.zeros((pad_count, feature_len), dtype=x_np.dtype)])
                    cl_np = np.hstack([ cl_np, np.arange(cl_np.max()+  1, padd_to_index + 1)])
                g_data = GraphData(
                    x=torch.from_numpy(x_np),
                    y=torch.from_numpy(y_np),
                    cluster=torch.from_numpy(cl_np),
                    edge_index=torch.from_numpy(ei_np),
                    valid_len=torch.tensor([valid_len]),
                    time_step_len=torch.tensor([padd_to_index + 1])
                )
                g_ls.append(g_data)
            data, slices = self.collate(g_ls)
            out_file = os.path.join(self.processed_dir, f'dataset_part_{shard_id//shard_size}.pt')
            torch.save((data, slices), out_file)

            del g_ls, chunk_data, chunk_valid # free memory before next shard

    def len(self):
        total = 0
        for fname in self.processed_file_names:
            _, slices = torch.load(os.path.join(self.processed_dir, fname))
            # each shard’s slices['x'] has length = #scenarios + 1
            num_scen = slices['x'].size(0) - 1
            total += num_scen
        return total

    def get(self, idx: int) -> Data:
        cumulative = 0
        for fname in self.processed_file_names:
            data, slices = torch.load(os.path.join(self.processed_dir, fname))
            # number of scenarios in this shard
            num_scen = slices['x'].size(0) - 1
            if idx < cumulative + num_scen:
                local_i = idx - cumulative

                # --- slice nodes & clusters via slices['x'] & slices['cluster'] ---
                x_start, x_end = slices['x'][local_i].item(), slices['x'][local_i+1].item()
                x = data.x[x_start : x_end]
                cluster = data.cluster[x_start : x_end]

                # --- slice edges via slices['edge_index'] ---
                e_start, e_end = slices['edge_index'][local_i].item(), slices['edge_index'][local_i+1].item()
                edge_index = data.edge_index[:, e_start : e_end]

                # --- slice targets via slices['y'] ---
                y_start, y_end = slices['y'][local_i].item(), slices['y'][local_i+1].item()
                y = data.y[y_start : y_end]

                # --- valid_len & time_step_len are 1D per scenario ---
                valid_len      = data.valid_len[local_i].unsqueeze(0)
                time_step_len  = data.time_step_len[local_i].unsqueeze(0)

                return Data(
                    x=x,
                    edge_index=edge_index,
                    cluster=cluster,
                    y=y,
                    valid_len=valid_len,
                    time_step_len=time_step_len
                )

            cumulative += num_scen

        raise IndexError(f"Index {idx} out of range")

# %%
if __name__ == "__main__":
    for folder in os.listdir(DATA_DIR):
        if folder.startswith('.'):
            continue
        dataset_input_path = os.path.join(
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
        GraphDataset(dataset_input_path)
        print(f'Completed sharding for {folder}')
        # dataset = GraphDataset(dataset_input_path)
        # batch_iter = DataLoader(dataset, batch_size=256)
        # batch = next(iter(batch_iter))


# %%
