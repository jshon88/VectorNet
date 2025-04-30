# VectorNet reimplementation: Graph-Based Vehicle Trajectory Forecasting

This repository implements **VectorNet**, a two-stasge Graph Neural Network for motion forecasting in autonomous driving. It covers:

- Hand-crafted feature extraction from raw Argoverse CSVs  
- Construction of per-scenario graph datasets  
- Local polyline GNN (“SubGraph”) + global self-attention  
- End-to-end training & evaluation on Argoverse  
- Visualization utilities for maps, graphs, and predictions  

## Repository Structure
```bash
.
├── preprocess_data.py           # orchestrates feature extraction → .pkl
├── dataset.py                   # builds PyG GraphDataset → .pt shards
├── modeling/
│   ├── vectornet.py             # HGNN model (SubGraph + Attention + MLP)
│   ├── subgraph.py              # local GNN layers & pooling
│   ├── selfatten.py             # global self-attention modules
│   ├── predmlp.py               # prediction MLP head
├── train.py          # training & validation loop
├── utils/
│   ├── feature_utils.py         # raw-to-feature functions
│   ├── lane_utils.py            # lane boundary computations & featurization
│   ├── object_utils.py          # moving-object feature extraction
│   ├── agent_utils.py          # ego vehicle feature extraction
│   └── config.py                # global settings (paths, thresholds)
├── graphs/                      # example, loss, and evaluation visualization
└── README.md                    # this file
```

## Environment & Installation
1. **Clone** this repo and `cd` into it:
   ```bash
   git clone https://github.com/jshon88/VectorNet.git
   cd VectorNet

2. Create environment
    ```bash
    conda env create -f environment.yml

3. Dwonload Argoverse Motion Forecasting CSVs:
    ```bash
    # Place them under data/train/ and data/val/ (or adjust paths in config.py)

4. Instsall argoverse-api

## Data Preprocessing
1. Extract hand-crafted features -> .pkl
    ```bash
    python preprocess_data.py
    ```
    - Reads raw CSV and builds per-scenario features (agent, lanes, objects)
    - Saves one .pkl per scenario for fast downstream loading

2. Build GraphDataset -> shards of dataset.pt
    ```bash
    python dataset.py
    ```
    - Loads each .pkl in manageable batches
    - Wraps scenario as torch_geometric.data.GraphData
    - collates into dataset_part_*.pt

## Training & Evaluation
```bash
python dataset.py
```

## Visualization
- Graph_viz.ipynb
- Contains per scenario graph dataset visualization, sub-graph level visualization
- Plots vectorized representation of a sample scenario
- Plots vectorized representation of predicted vs ground truth of ego vehicle

## Model Architecture
1. Feature extractor -> node faetures
2. SubGraph (local GNN):
    - 3x GraphLayerProp for Message Passing
3. Gloabl self-attention (Global GNN):
4. Prediction head:
    - Extract ego vehicle embedding -> MLP -> prediction

See modeling/ for full implementation

## Other Notes
- With limited computing power of my local machine, I was able to use only 0.3% of training dataset for training for 12 epochs

## References
Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)  
Reference Github repo source: [yet-another-vectornet](https://github.com/xk-huang/yet-another-vectornet)

