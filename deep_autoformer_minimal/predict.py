import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deep_autoformer.model import DeepAutoformer
from deep_autoformer.dataset import SlidingWindowDataset

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def predict(cfg_path, model_path, num_samples=10, plot=True):
    """
    Faz previsões usando o modelo treinado

    Args:
        cfg_path: caminho para o ficheiro de configuração
        model_path: caminho para o modelo treinado (.pt)
        num_samples: número de amostras para prever
        plot: se True, plota os resultados
    """
    cfg = load_config(cfg_path)

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Carregar dataset
    ds = SlidingWindowDataset(
        csv_path=cfg["data"]["csv_path"],
        seq_len=int(cfg["data"]["seq_len"]),
        label_len=int(cfg["data"]["label_len"]),
        pred_len=int(cfg["data"]["pred_len"]),
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"].get("feature_cols")
    )

    # Criar modelo
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

    # Carregar pesos do modelo treinado
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nModelo carregado de: {model_path}")
    print(f"Dataset: {len(ds)} amostras")
    print(f"Seq length: {cfg['data']['seq_len']}, Pred length: {cfg['data']['pred_len']}")

    # Selecionar amostras aleatórias do dataset
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)

    all_predictions = []
    all_targets = []
    all_contexts = []

    with torch.no_grad():
        for idx in indices:
            enc, dec, y = ds[idx]

            # Adicionar batch dimension
            enc = enc.unsqueeze(0).to(device)
            dec = dec.unsqueeze(0).to(device)

            # Fazer previsão
            pred = model(enc, dec)

            # Desnormalizar
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            y_np = y.numpy().reshape(-1, 1)
            enc_np = enc.cpu().numpy()[0, :, 0].reshape(-1, 1)

            pred_denorm = ds.inverse_transform_y(pred_np).flatten()
            y_denorm = ds.inverse_transform_y(y_np).flatten()
            context_denorm = ds.inverse_transform_y(enc_np).flatten()

            all_predictions.append(pred_denorm)
            all_targets.append(y_denorm)
            all_contexts.append(context_denorm)

            # Calcular métricas para esta amostra
            mae = np.mean(np.abs(pred_denorm - y_denorm))
            rmse = np.sqrt(np.mean((pred_denorm - y_denorm) ** 2))

            print(f"\nAmostra {idx}:")
            print(f"  MAE: {mae:.4f} kW")
            print(f"  RMSE: {rmse:.4f} kW")
            print(f"  Contexto (últimos 5): {context_denorm[-5:]}")
            print(f"  Previsão (primeiros 5): {pred_denorm[:5]}")
            print(f"  Real (primeiros 5): {y_denorm[:5]}")

    # Plotar resultados
    if plot:
        n_plots = min(4, len(indices))
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))
        if n_plots == 1:
            axes = [axes]

        for i in range(n_plots):
            context = all_contexts[i]
            pred = all_predictions[i]
            target = all_targets[i]

            # Criar eixo temporal
            context_time = np.arange(len(context))
            pred_time = np.arange(len(context), len(context) + len(pred))

            axes[i].plot(context_time, context, label='Contexto (histórico)',
                        color='blue', linewidth=2)
            axes[i].plot(pred_time, pred, label='Previsão',
                        color='red', linewidth=2, linestyle='--')
            axes[i].plot(pred_time, target, label='Real',
                        color='green', linewidth=2, alpha=0.7)

            axes[i].axvline(x=len(context), color='gray', linestyle=':', alpha=0.5)
            axes[i].set_xlabel('Time steps (15 min)')
            axes[i].set_ylabel('Consumo (kW)')
            axes[i].set_title(f'Amostra {indices[i]}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('predictions.png', dpi=150)
        print(f"\nGráfico guardado em: predictions.png")
        plt.show()

    return all_predictions, all_targets, all_contexts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fazer previsões com modelo treinado')
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Caminho para o ficheiro de configuração")
    parser.add_argument("--model", type=str, default="configs/best_model.pt",
                       help="Caminho para o modelo treinado")
    parser.add_argument("--samples", type=int, default=4,
                       help="Número de amostras para prever")
    parser.add_argument("--no-plot", action="store_true",
                       help="Não plotar os resultados")

    args = parser.parse_args()

    predict(args.config, args.model, args.samples, not args.no_plot)
