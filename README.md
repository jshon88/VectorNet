# VectorNet reimplementation: Graph-Based Vehicle Trajectory Forecasting

This repository implements **VectorNet**, a two-stasge Graph Neural Network for motion forecasting in autonomous driving. It covers:

- Hand-crafted feature extraction from raw Argoverse CSVs  
- Construction of per-scenario graph datasets  
- Local polyline GNN (“SubGraph”) + global self-attention  
- End-to-end training & evaluation on Argoverse  
- Visualization utilities for maps, graphs, and predictions  

## Repository Structure
. ├── preprocess_data.py # orchestrates feature extraction → .pkl ├── dataset.py # builds PyG GraphDataset → .pt shards ├── modeling/ │ ├── vectornet.py # HGNN model (SubGraph + Attention + MLP) │ ├── subgraph.py # local GNN layers & pooling │ ├── selfatten.py # global self-attention modules │ ├── predmlp.py # prediction MLP head │ └── … # other helper scripts ├── single_gpu_train.py # training & validation loop ├── utils/ │ ├── feature_utils.py # raw‐to‐feature functions │ ├── lane_utils.py # lane boundary computations & feature extraction │ ├── object_utils.py # moving‐object feature extraction │ ├── viz_utils.py # plotting helpers │ └── config.py # global settings (paths, thresholds) ├── interm_data/ │ ├── train_intermediate/ # per-scenario .pkl files │ ├── val_intermediate/ │ └── test_intermediate/ ├── trained_params/ │ └── loss_history.json # logged train/val losses & metrics │ └── best_epoch_*.pth # best‐checkpointed models ├── graphs/ # example visualizations ├── data/ # raw Argoverse CSVs (not checked in) └── README.md # you are here

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
- Reads raw CSV and builds per-scenario features (agent, lanes, objects)
- Saves one .pkl per scenario for fast downstream loading

2. Build GraphDataset -> shards of dataset.pt
    ```bash
    python dataset.py
- Loads each .pkl in manageable batches
- Wraps scenario as torch_geometric.data.GraphData
- collates into dataset_part_*.pt

## Training & Evaluation
    ```bash
    python dataset.py

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

## References
Paper: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)  
Reference Github repo source: [yet-another-vectornet](https://github.com/xk-huang/yet-another-vectornet)

