
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

from deep_autoformer.model import DeepAutoformer
from deep_autoformer.dataset import SlidingWindowDataset
from deep_autoformer.utils import mae, rmse, mape

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc='Training', leave=False)
    for enc, dec, y in pbar:
        enc = enc.to(device)
        dec = dec.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(enc, dec)  # [B, pred_len, 1]
        loss = loss_fn(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_loss = loss.item()
        total += batch_loss * enc.size(0)

        # Atualizar progress bar com loss atual
        pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn, scaler=None, desc='Validation'):
    model.eval()
    total = 0.0
    mae_v = 0.0
    rmse_v = 0.0
    mape_v = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for enc, dec, y in pbar:
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

    print("="*80)
    print("DEEP AUTOFORMER - DAY-AHEAD FORECASTING")
    print("="*80)
    print(f"Device: {device}")
    print()

    print("Loading dataset...")
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

    print("\nDATASET INFO:")
    print(f"  Total samples: {N}")
    print(f"  Train: {n_train} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {n_val} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {n_test} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print()

    print("MODEL CONFIG:")
    print(f"  Input sequence length: {cfg['data']['seq_len']} steps ({cfg['data']['seq_len']/96:.1f} dias)")
    print(f"  Prediction length: {cfg['data']['pred_len']} steps ({cfg['data']['pred_len']/96:.1f} dias)")
    print(f"  Model dimension: {cfg['model']['d_model']}")
    print(f"  Attention heads: {cfg['model']['n_heads']}")
    print(f"  Encoder layers: {cfg['model']['e_layers']}")
    print(f"  Decoder layers: {cfg['model']['d_layers']}")
    print()

    print("TRAINING CONFIG:")
    print(f"  Learning rate: {cfg['train']['lr']}")
    print(f"  Weight decay: {cfg['train']['weight_decay']}")
    print(f"  Max epochs: {cfg['train']['epochs']}")
    print(f"  Early stopping patience: {cfg['train']['patience']}")
    print("="*80)
    print()

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
    total_epochs = int(cfg["train"]["epochs"])

    print("Starting training...\n")

    epoch_times = []
    epoch_pbar = tqdm(range(1, total_epochs + 1), desc='Epochs', position=0)

    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Training
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validation
        val_loss, val_mae, val_rmse, val_mape = evaluate(model, val_loader, device, loss_fn, ds.scaler_y, desc='Validation')

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Calcular tempo estimado restante
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = total_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60

        # Atualizar progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{tr_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_MAE': f'{val_mae:.2f}',
            'time': f'{epoch_time:.1f}s',
            'ETA': f'{eta_minutes:.1f}m'
        })

        # Log detalhado
        status = "✓" if val_loss < best_val - 1e-6 else " "
        print(f"{status} Epoch {epoch:03d}/{total_epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} (MAE {val_mae:.2f} RMSE {val_rmse:.2f} MAPE {val_mape:.2f}%) | {epoch_time:.1f}s")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            model_path = os.path.join(os.path.dirname(cfg_path), "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  → Model saved! (val_loss improved to {val_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"\n⚠ Early stopping triggered (patience={patience})")
                break

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total training time: {sum(epoch_times)/60:.1f} minutes")
    print(f"Best validation loss: {best_val:.4f}")
    print()

    # Final test
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(cfg_path), "best_model.pt"), map_location=device))
    test_loss, test_mae, test_rmse, test_mape = evaluate(model, test_loader, device, loss_fn, ds.scaler_y, desc='Testing')

    print("\nFINAL TEST RESULTS:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE:  {test_mae:.2f} kW")
    print(f"  RMSE: {test_rmse:.2f} kW")
    print(f"  MAPE: {test_mape:.2f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
