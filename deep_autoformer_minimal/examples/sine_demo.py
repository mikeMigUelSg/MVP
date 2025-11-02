
"""
Quick demo that synthesizes a seasonal trend + noise signal,
writes to CSV, and trains DeepAutoformer.
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import subprocess, sys

# Generate a toy dataset
n = 5000
t0 = datetime(2024, 1, 1)
ts = [t0 + timedelta(minutes=15*i) for i in range(n)]
t = np.arange(n)
trend = 0.001 * t
season = 2.0 * np.sin(2 * np.pi * t / 96)  # daily
noise = 0.3 * np.random.randn(n)
y = trend + season + noise
temp = 15 + 10*np.sin(2*np.pi*t/96 + 0.5) + 0.5*np.random.randn(n)  # a covariate

df = pd.DataFrame({"timestamp": ts, "target": y.astype(np.float32), "temp": temp.astype(np.float32)})
os.makedirs("data", exist_ok=True)
csv_path = "data/sine.csv"
df.to_csv(csv_path, index=False)

# Write a minimal config that points to this CSV
cfg = f"""
data:
  csv_path: {csv_path}
  seq_len: 96
  label_len: 48
  pred_len: 24
  target_col: target
  feature_cols: ["temp"]

model:
  d_model: 128
  n_heads: 4
  e_layers: 2
  d_layers: 1
  d_ff: 256
  top_k: 4
  kernel_size: 25
  dropout: 0.1
  add_deep_mlp: true

train:
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 64
  epochs: 5
  train_ratio: 0.7
  val_ratio: 0.15
  patience: 3
  device: "cuda"
"""
open("configs/sine.yaml", "w").write(cfg)

print("Dataset and config prepared. Run:\n")
print("  python train.py --config configs/sine.yaml\n")
