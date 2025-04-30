import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import logging
import time
import glob
import re
from datetime import datetime
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import dataset
from dataset import GraphData
import __main__
__main__.GraphData = GraphData

from modeling.vectornet import HGNN
from dataset import GraphDataset
from utils.eval import get_eval_metric_results

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
SEED = 42
TRAIN_DIR       = os.path.join('interm_data', 'train_intermediate')
VAL_DIR         = os.path.join('interm_data', 'val_intermediate')
SAVE_DIR        = 'trained_params'
HISTORY_PATH    = os.path.join(SAVE_DIR, 'loss_history.json')
os.makedirs(SAVE_DIR, exist_ok=True)

# Model params
IN_CHANNELS    = 8      # feature dim per node
FUTURE_STEPS   = 30     # predict 3s @ 10Hz → 30 steps
NUM_SUB_LAYERS = 3
SUB_HIDDEN     = 64
GLOBAL_HIDDEN  = 64
MLP_HIDDEN     = 64
NEED_SCALE     = True

# Training hyperparams
BATCH_SIZE     = 16
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
EPOCHS         = 12
SMALL_DATA_PCT = 0.003    # if <1, use subset for quick tests
VAL_EVERY      = 1      # validate every epoch
SHOW_EVERY     = 1    # print batch‐level loss every N steps

# Evaluation settings
MAX_N_GUESSES  = 6
HORIZON        = FUTURE_STEPS
MISS_THRESHOLD = 5.0

# Reproducibility
torch.manual_seed(SEED)


def main():

    # timestamped logfile
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fname = os.path.join(SAVE_DIR, f'train_{now}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_fname),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Load datasets
    train_data = GraphDataset(TRAIN_DIR)
    val_data   = GraphDataset(VAL_DIR)

    # Optional small‐subset mode
    if SMALL_DATA_PCT < 1.0:
        n_train = len(train_data)
        n_val   = len(val_data)
        keep_train = int(n_train * SMALL_DATA_PCT)
        keep_val   = int(n_val   * SMALL_DATA_PCT)
        train_data, _ = random_split(train_data, [keep_train, n_train - keep_train])
        val_data,   _ = random_split(val_data,   [keep_val,   n_val   - keep_val])

    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # device = 'cpu'
    model = HGNN(
        in_channels=IN_CHANNELS,
        future_steps=FUTURE_STEPS,
        num_subgraph_layers=NUM_SUB_LAYERS,
        subgraph_hidden=SUB_HIDDEN,
        global_hidden=GLOBAL_HIDDEN,
        mlp_hidden=MLP_HIDDEN,
        need_scale=NEED_SCALE
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ─── Resume training from checkpoint ───────────────────────────────────
    ckpt_files = glob.glob(os.path.join(SAVE_DIR, "best_epoch_*.pth"))
    if ckpt_files:
        # pick the checkpoint with the largest epoch number
        def _epoch(fn):
            m = re.search(r"best_epoch_(\d+)_ADE", fn)
            return int(m.group(1)) if m else -1
        latest = max(ckpt_files, key=_epoch)
        logging.info(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_minade = ckpt.get('minADE', float('inf'))
    else:
        logging.info("No checkpoint found; training from scratch")
        start_epoch = 1
        best_minade = float('inf')

    # ─── Load existing loss history if present ───────────────────────────────────
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            old_hist = json.load(f)
        # pull out the lists we care about
        train_mse_history = old_hist.get('train_mse', [])
        val_mse_history   = old_hist.get('val_mse', [])
        # metrics lists (minADE, minFDE, MR) must be zipped back into dicts
        val_minADE = old_hist.get('val_minADE', [])
        val_minFDE = old_hist.get('val_minFDE', [])
        val_MR     = old_hist.get('val_MR', [])
        epoch_val_metrics = [
            {'minADE': a, 'minFDE': f, 'MR': m}
            for a, f, m in zip(val_minADE, val_minFDE, val_MR)
        ]

        # If you want start_epoch to match history length+1:
        # (but you’re already computing start_epoch from the checkpoint)
        # start_epoch = len(train_mse_history) + 1
        print(f"Loaded history for {len(train_mse_history)} epochs; resuming at epoch {start_epoch}")
    else:
        # no prior history
        train_mse_history  = []
        val_mse_history    = []
        epoch_val_metrics  = []
        print("No prior loss_history.json found; starting fresh")

    # # Histories
    # epoch_train_loss  = []
    # epoch_val_loss    = []
    # epoch_val_metrics = []
    # best_minade = float('inf')

    # ─────────────────────────────────────────────────────────────────────────────
    # Training loop
    logging.info('Start Training')
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps    = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            preds   = model(batch)            # [B, 2*FUTURE_STEPS]
            targets = batch.y.to(device).view_as(preds)      # same shape
            loss     = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps    += 1

            if step % SHOW_EVERY == 0:
                avg_chunk = train_loss_sum / train_steps
                elapsed   = time.time() - t0
                logging.info(f"Epoch {epoch} | Step {step:04d}/{len(train_loader)} | "
                    f"Train MSE (so far): {avg_chunk:.6f} | {elapsed:.2f}s")
                # continue accumulating for full‐epoch average

        # Average training loss over all batches
        avg_train_loss = train_loss_sum / max(train_steps, 1)
        train_mse_history.append(avg_train_loss)

        # **STEP THE SCHEDULER HERE** (after training, before validation)
        scheduler.step()

        # Validation
        if epoch % VAL_EVERY == 0:
            model.eval()
            # 1) Compute eval metrics (minADE, minFDE, MR)
            with torch.no_grad():
                metrics = get_eval_metric_results(
                    model, val_loader, device,
                    out_channels=2 * FUTURE_STEPS,
                    max_n_guesses=MAX_N_GUESSES,
                    horizon=HORIZON,
                    miss_threshold=MISS_THRESHOLD
                )
            epoch_val_metrics.append(metrics)

            # 2) Compute validation MSE
            val_loss_sum = 0.0
            val_steps    = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds   = model(batch)
                    targets = batch.y.to(device).view_as(preds)
                    val_loss_sum += F.mse_loss(preds, targets).item()
                    val_steps    += 1
            avg_val_loss = val_loss_sum / max(val_steps, 1)
            val_mse_history.append(avg_val_loss)

            # Logging
            minade = metrics['minADE']
            logging.info(f"Epoch {epoch} validation → "
                f"Val MSE: {avg_val_loss:.6f} | "
                f"minADE: {minade:.6f}, minFDE: {metrics['minFDE']:.6f}, "
                f"MR: {metrics['MR']:.6f}")

            # Checkpoint best
            if minade < best_minade:
                best_minade = minade
                ckpt = {
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch':     epoch,
                    'minADE':    minade
                }
                fname = f"best_epoch_{epoch:02d}_ADE_{minade:.4f}.pth"
                torch.save(ckpt, os.path.join(SAVE_DIR, fname))

        # ─────────────────────────────────────────────────────────────────────────────
        # Save JSON history
        history = {
            'train_mse':  train_mse_history,
            'val_mse':    val_mse_history,
            'val_minADE': [m['minADE']  for m in epoch_val_metrics],
            'val_minFDE': [m['minFDE'] for m in epoch_val_metrics],
            'val_MR':     [m['MR']      for m in epoch_val_metrics],
        }
        with open(HISTORY_PATH, 'w') as fp:
            json.dump(history, fp, indent=2)
        print(f"Saved loss history to {HISTORY_PATH}")

    # Final evaluation on full val set
    model.eval()
    with torch.no_grad():
        final_metrics = get_eval_metric_results(
            model, val_loader, device,
            out_channels=2 * FUTURE_STEPS,
            max_n_guesses=MAX_N_GUESSES,
            horizon=HORIZON,
            miss_threshold=MISS_THRESHOLD
        )
    logging.info("Final validation: \n", final_metrics)
    print("Final validation →", final_metrics)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()