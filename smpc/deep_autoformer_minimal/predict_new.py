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

def predict_day_ahead(cfg_path, model_path, new_csv_path, plot=True, n_days=10):
    """
    Faz previsões day-ahead: usa dados de um dia para prever o dia seguinte completo

    Args:
        cfg_path: caminho para o ficheiro de configuração
        model_path: caminho para o modelo treinado (.pt)
        new_csv_path: caminho para CSV com novos dados (formato: date, target)
        plot: se True, plota os resultados
        n_days: número de dias a prever
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

    # Carregar dataset ORIGINAL (para obter o scaler)
    ds_original = SlidingWindowDataset(
        csv_path=cfg["data"]["csv_path"],
        seq_len=int(cfg["data"]["seq_len"]),
        label_len=int(cfg["data"]["label_len"]),
        pred_len=int(cfg["data"]["pred_len"]),
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"].get("feature_cols")
    )

    # Carregar dados completos do novo CSV
    df = pd.read_csv(new_csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\nDados carregados: {len(df)} time steps")
    print(f"Período: {df['date'].min()} a {df['date'].max()}")

    # Assumir que temos dados de 15 em 15 minutos (96 time steps por dia)
    steps_per_day = 96
    seq_len = int(cfg["data"]["seq_len"])
    pred_len = int(cfg["data"]["pred_len"])

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
        pred_len=pred_len,
        label_len=int(cfg["data"]["label_len"]),
        output_dim=1,
        add_deep_mlp=bool(cfg["model"]["add_deep_mlp"])
    ).to(device)

    # Carregar pesos do modelo treinado
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preparar dados
    target_col = cfg["data"]["target_col"]
    feature_cols = cfg["data"].get("feature_cols", [])

    print(f"\nModelo carregado de: {model_path}")
    print(f"Features: {feature_cols if feature_cols else 'Nenhuma (apenas target)'}")
    print(f"Fazendo previsões day-ahead para {n_days} dias...")
    print(f"Usando últimos {seq_len} time steps como contexto")
    print(f"Prevendo {pred_len} time steps à frente por iteração")

    values = df[target_col].values

    # Normalizar target usando o scaler do dataset original
    from sklearn.preprocessing import StandardScaler
    scaler_y = ds_original.scaler_y
    values_scaled = scaler_y.transform(values.reshape(-1, 1)).flatten()

    # Preparar features se existirem
    if feature_cols:
        features = df[feature_cols].values  # [N, num_features]
        scaler_x = ds_original.scaler_x
        if scaler_x is not None:
            features_scaled = scaler_x.transform(features)
        else:
            features_scaled = features
    else:
        features_scaled = None

    all_results = []

    with torch.no_grad():
        # Para cada dia, usar o dia anterior como contexto
        for day_idx in range(n_days):
            # Calcular índices: precisamos de seq_len para contexto + steps_per_day para previsão
            start_idx = day_idx * steps_per_day
            context_start = start_idx
            context_end = start_idx + seq_len
            target_start = context_end
            target_end = target_start + steps_per_day

            # Verificar se temos dados suficientes
            if target_end > len(values):
                print(f"  Dia {day_idx + 1}: Não há dados suficientes, parando.")
                break

            # Obter contexto (dia anterior ou últimos seq_len steps)
            context_target = values_scaled[context_start:context_end]
            target_real = values[target_start:target_end]

            # Obter features para o contexto
            if features_scaled is not None:
                context_features = features_scaled[context_start:context_end]
                # Concatenar target + features: [seq_len, 1 + num_features]
                context = np.column_stack([context_target, context_features])
            else:
                context = context_target.reshape(-1, 1)

            # Fazer previsão iterativa para cobrir todo o dia seguinte
            predictions = []
            current_context = context.copy()

            # Quantas iterações precisamos para cobrir o dia?
            n_iterations = (steps_per_day + pred_len - 1) // pred_len

            for iter_idx in range(n_iterations):
                # Preparar encoder input (contexto: target + features)
                enc_input = torch.FloatTensor(current_context[-seq_len:]).to(device)  # [seq_len, features]

                # Preparar decoder input
                label_len = int(cfg["data"]["label_len"])
                num_features = enc_input.shape[1]
                dec_input = torch.zeros(label_len + pred_len, num_features).to(device)

                # Preencher label_len com o final do contexto
                dec_input[:label_len, :] = enc_input[-label_len:, :]

                # Adicionar batch dimension
                enc_batch = enc_input.unsqueeze(0)
                dec_batch = dec_input.unsqueeze(0)

                # Fazer previsão
                pred = model(enc_batch, dec_batch)
                pred_np = pred.cpu().numpy().flatten()

                # Adicionar às previsões
                predictions.extend(pred_np)

                # Atualizar contexto com as previsões + features
                if features_scaled is not None:
                    # Adicionar features dos próximos pred_len steps
                    next_features = features_scaled[context_end:context_end + pred_len]
                    # Se não houver features suficientes no dataset, truncar
                    if len(next_features) < pred_len:
                        # Não podemos continuar sem features futuras
                        break

                    pred_with_features = np.column_stack([pred_np[:pred_len], next_features[:pred_len]])
                    current_context = np.concatenate([current_context, pred_with_features])
                else:
                    current_context = np.concatenate([current_context, pred_np[:pred_len].reshape(-1, 1)])

            # Pegar apenas os primeiros steps_per_day previstos
            predictions = np.array(predictions[:steps_per_day])

            # Desnormalizar
            pred_denorm = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            # Contexto: extrair apenas o target (primeira coluna) para desnormalizar
            context_target_only = context[:, 0].reshape(-1, 1) if context.ndim > 1 else context.reshape(-1, 1)
            context_denorm = scaler_y.inverse_transform(context_target_only).flatten()

            # Calcular métricas
            mae = np.mean(np.abs(pred_denorm - target_real))
            rmse = np.sqrt(np.mean((pred_denorm - target_real) ** 2))

            # Informações do dia
            day_start_date = df['date'].iloc[target_start]
            day_end_date = df['date'].iloc[target_end - 1]

            all_results.append({
                'day': day_idx + 1,
                'date_start': day_start_date,
                'date_end': day_end_date,
                'context': context_denorm,
                'prediction': pred_denorm,
                'real': target_real,
                'mae': mae,
                'rmse': rmse
            })

            print(f"  Dia {day_idx + 1} ({day_start_date.strftime('%Y-%m-%d')}): MAE={mae:.2f} kW, RMSE={rmse:.2f} kW")

    # Calcular métricas médias
    if all_results:
        avg_mae = np.mean([r['mae'] for r in all_results])
        avg_rmse = np.mean([r['rmse'] for r in all_results])

        print(f"\nMétricas médias:")
        print(f"  MAE médio: {avg_mae:.2f} kW")
        print(f"  RMSE médio: {avg_rmse:.2f} kW")

        # Plotar
        if plot:
            n_cols = 2
            n_rows = (len(all_results) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
            if len(all_results) == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, result in enumerate(all_results):
                ax = axes[i]

                # Time steps
                context_steps = len(result['context'])
                pred_steps = len(result['prediction'])

                context_time = np.arange(context_steps)
                pred_time = np.arange(context_steps, context_steps + pred_steps)

                # Plotar
                ax.plot(context_time, result['context'],
                       label='Contexto (dia anterior)', color='blue', linewidth=1.5, alpha=0.7)
                ax.plot(pred_time, result['prediction'],
                       label='Previsão (day-ahead)', color='red', linewidth=2, linestyle='--')
                ax.plot(pred_time, result['real'],
                       label='Real', color='green', linewidth=2, alpha=0.7)

                ax.axvline(x=context_steps, color='gray', linestyle=':', alpha=0.5, linewidth=2)
                ax.set_xlabel('Time steps (15 min)', fontsize=10)
                ax.set_ylabel('Consumo (kW)', fontsize=10)
                ax.set_title(f'Dia {result["day"]} - {result["date_start"].strftime("%Y-%m-%d")} | MAE={result["mae"]:.2f} kW',
                           fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            # Remover subplots vazios
            for i in range(len(all_results), len(axes)):
                fig.delaxes(axes[i])

            plt.suptitle(f'Previsões Day-Ahead - {len(all_results)} dias | MAE médio: {avg_mae:.2f} kW',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('prediction_day_ahead.png', dpi=150, bbox_inches='tight')
            print(f"\nGráfico guardado em: prediction_day_ahead.png")
            plt.show()

    return all_results

def predict_new_data(cfg_path, model_path, new_csv_path, plot=True, n_predictions=10):
    """
    Faz previsões para dados completamente novos

    Args:
        cfg_path: caminho para o ficheiro de configuração
        model_path: caminho para o modelo treinado (.pt)
        new_csv_path: caminho para CSV com novos dados (formato: date, target)
        plot: se True, plota os resultados
        n_predictions: número de previsões a fazer ao longo do mês
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

    # Carregar dataset ORIGINAL (para obter o scaler)
    ds_original = SlidingWindowDataset(
        csv_path=cfg["data"]["csv_path"],
        seq_len=int(cfg["data"]["seq_len"]),
        label_len=int(cfg["data"]["label_len"]),
        pred_len=int(cfg["data"]["pred_len"]),
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"].get("feature_cols")
    )

    # Carregar NOVOS dados
    ds_new = SlidingWindowDataset(
        csv_path=new_csv_path,
        seq_len=int(cfg["data"]["seq_len"]),
        label_len=int(cfg["data"]["label_len"]),
        pred_len=int(cfg["data"]["pred_len"]),
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"].get("feature_cols")
    )

    # IMPORTANTE: usar o scaler do dataset original!
    ds_new.scaler_y = ds_original.scaler_y

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
    print(f"Novos dados: {len(ds_new)} amostras possíveis")

    # Selecionar índices distribuídos ao longo do dataset
    total_samples = len(ds_new)
    indices = np.linspace(0, total_samples - 1, n_predictions, dtype=int)

    print(f"\nFazendo {n_predictions} previsões distribuídas ao longo do mês...")

    all_results = []
    total_mae = 0
    total_rmse = 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            enc, dec, y = ds_new[idx]

            # Adicionar batch dimension
            enc_batch = enc.unsqueeze(0).to(device)
            dec_batch = dec.unsqueeze(0).to(device)

            # Fazer previsão
            pred = model(enc_batch, dec_batch)

            # Desnormalizar
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            y_np = y.numpy().reshape(-1, 1)
            enc_np = enc.numpy()[:, 0].reshape(-1, 1)  # enc já é 2D (seq_len, features)

            pred_denorm = ds_new.inverse_transform_y(pred_np).flatten()
            y_denorm = ds_new.inverse_transform_y(y_np).flatten()
            context_denorm = ds_new.inverse_transform_y(enc_np).flatten()

            # Calcular métricas
            mae = np.mean(np.abs(pred_denorm - y_denorm))
            rmse = np.sqrt(np.mean((pred_denorm - y_denorm) ** 2))

            total_mae += mae
            total_rmse += rmse

            all_results.append({
                'idx': idx,
                'context': context_denorm,
                'prediction': pred_denorm,
                'real': y_denorm,
                'mae': mae,
                'rmse': rmse
            })

            print(f"  Previsão {i+1}/{n_predictions} (idx={idx}): MAE={mae:.4f} kW, RMSE={rmse:.4f} kW")

    avg_mae = total_mae / n_predictions
    avg_rmse = total_rmse / n_predictions

    print(f"\nMétricas médias:")
    print(f"  MAE médio: {avg_mae:.4f} kW")
    print(f"  RMSE médio: {avg_rmse:.4f} kW")

    # Plotar todas as previsões numa só janela
    if plot:
        # Calcular o layout dos subplots
        n_cols = 2
        n_rows = (n_predictions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_predictions > 1 else [axes]

        for i, result in enumerate(all_results):
            ax = axes[i]

            context = result['context']
            pred = result['prediction']
            real = result['real']

            context_time = np.arange(len(context))
            pred_time = np.arange(len(context), len(context) + len(pred))

            ax.plot(context_time, context, label='Contexto',
                   color='blue', linewidth=1.5, alpha=0.7)
            ax.plot(pred_time, pred, label='Previsão',
                   color='red', linewidth=2, linestyle='--', marker='s', markersize=3)
            ax.plot(pred_time, real, label='Real',
                   color='green', linewidth=2, alpha=0.7, marker='^', markersize=3)

            ax.axvline(x=len(context), color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Time steps (15 min)', fontsize=9)
            ax.set_ylabel('Consumo (kW)', fontsize=9)
            ax.set_title(f'Previsão {i+1} (idx={result["idx"]}) | MAE={result["mae"]:.2f} kW',
                        fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remover subplots vazios
        for i in range(n_predictions, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'{n_predictions} Previsões ao longo do mês | MAE médio: {avg_mae:.2f} kW',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prediction_new_data.png', dpi=150, bbox_inches='tight')
        print(f"\nGráfico guardado em: prediction_new_data.png")
        plt.show()

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fazer previsões em dados novos')
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Caminho para o ficheiro de configuração")
    parser.add_argument("--model", type=str, default="configs/best_model.pt",
                       help="Caminho para o modelo treinado")
    parser.add_argument("--data", type=str, required=True,
                       help="Caminho para CSV com novos dados")
    parser.add_argument("--no-plot", action="store_true",
                       help="Não plotar os resultados")
    parser.add_argument("--n-predictions", type=int, default=10,
                       help="Número de previsões a fazer ao longo do mês (default: 10)")

    args = parser.parse_args()

    predict_new_data(args.config, args.model, args.data, not args.no_plot, args.n_predictions)
