
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from deep_autoformer.model import DeepAutoformer
from deep_autoformer.dataset import SlidingWindowDataset
from deep_autoformer.utils import mae, rmse, mape

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total = 0.0
    for enc, dec, y in loader:
        enc = enc.to(device)
        dec = dec.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(enc, dec)  # [B, pred_len, 1]
        loss = loss_fn(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * enc.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn, scaler=None):
    model.eval()
    total = 0.0
    mae_v = 0.0
    rmse_v = 0.0
    mape_v = 0.0
    n = 0
    for enc, dec, y in loader:
        enc = enc.to(device)
        dec = dec.to(device)
        y = y.to(device)
        out = model(enc, dec)
        loss = loss_fn(out, y)
        total += loss.item() * enc.size(0)

        # Desnormalizar para calcular métricas nos valores reais
        if scaler is not None:
            out_denorm = torch.from_numpy(scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1))).reshape(out.shape).to(device)
            y_denorm = torch.from_numpy(scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1))).reshape(y.shape).to(device)
        else:
            out_denorm = out
            y_denorm = y

        mae_v += mae(out_denorm, y_denorm).item() * enc.size(0)
        rmse_v += rmse(out_denorm, y_denorm).item() * enc.size(0)
        mape_v += mape(out_denorm, y_denorm).item() * enc.size(0)
        n += enc.size(0)
    return total / n, mae_v / n, rmse_v / n, mape_v / n

def main(cfg_path):
    cfg = load_config(cfg_path)

    # Auto-detect best available device
    device_name = cfg["train"].get("device", "auto")
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")
    ds = SlidingWindowDataset(
        csv_path=cfg["data"]["csv_path"],
        seq_len=int(cfg["data"]["seq_len"]),
        label_len=int(cfg["data"]["label_len"]),
        pred_len=int(cfg["data"]["pred_len"]),
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"].get("feature_cols")
    )

    N = len(ds)
    train_ratio = float(cfg["train"]["train_ratio"])
    val_ratio = float(cfg["train"]["val_ratio"])
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = DeepAutoformer(
        enc_in=1 + (len(cfg["data"].get("feature_cols", []))),
        dec_in=1 + (len(cfg["data"].get("feature_cols", []))),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        e_layers=int(cfg["model"]["e_layers"]),
        d_layers=int(cfg["model"]["d_layers"]),
        d_ff=int(cfg["model"]["d_ff"]),
        top_k=int(cfg["model"]["top_k"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        pred_len=int(cfg["data"]["pred_len"]),
        label_len=int(cfg["data"]["label_len"]),
        output_dim=1,
        add_deep_mlp=bool(cfg["model"]["add_deep_mlp"])
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = int(cfg["train"]["patience"])
    wait = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_mae, val_rmse, val_mape = evaluate(model, val_loader, device, loss_fn, ds.scaler_y)
        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} (MAE {val_mae:.4f} RMSE {val_rmse:.4f} MAPE {val_mape:.2f}%)")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(cfg_path), "best_model.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # Final test
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(cfg_path), "best_model.pt"), map_location=device))
    test_loss, test_mae, test_rmse, test_mape = evaluate(model, test_loader, device, loss_fn, ds.scaler_y)
    print(f"TEST | loss {test_loss:.4f} | MAE {test_mae:.4f} | RMSE {test_rmse:.4f} | MAPE {test_mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
