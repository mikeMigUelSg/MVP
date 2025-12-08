"""
Compara o modelo Deep Autoformer com métodos baseline
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from deep_autoformer.model import DeepAutoformer
from deep_autoformer.dataset import SlidingWindowDataset
from baseline import BaselineForecaster


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compare_model_with_baselines(cfg_path, model_path, num_samples=20, plot=True, test_csv=None, use_test_set=True):
    """
    Compara o modelo treinado com todos os métodos baseline

    Args:
        cfg_path: caminho para o ficheiro de configuração
        model_path: caminho para o modelo treinado
        num_samples: número de amostras para avaliar
        plot: se True, gera gráficos
        test_csv: caminho para CSV de teste (opcional). Se fornecido, usa este em vez do dataset original
        use_test_set: se True e test_csv não fornecido, usa apenas o conjunto de teste do split (últimas amostras)
    """
    cfg = load_config(cfg_path)

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("="*80)
    print("COMPARAÇÃO: DEEP AUTOFORMER vs BASELINES")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {model_path}")

    # Determinar qual dataset usar
    csv_for_eval = test_csv if test_csv else cfg['data']['csv_path']
    print(f"Dataset: {csv_for_eval}")
    if test_csv:
        print(f"  → Usando CSV de teste específico")
    elif use_test_set:
        print(f"  → Usando conjunto de TESTE (split: train={cfg['train']['train_ratio']}, val={cfg['train']['val_ratio']})")
    else:
        print(f"  → Usando amostras aleatórias de todo o dataset")

    print(f"Sequence length: {cfg['data']['seq_len']} steps ({cfg['data']['seq_len']/96:.1f} dias)")
    print(f"Prediction length: {cfg['data']['pred_len']} steps ({cfg['data']['pred_len']/96:.1f} dias)")
    print(f"Number of samples: {num_samples}")
    print("="*80)
    print()

    # Carregar dataset para o modelo (sempre usa o dataset original para manter a normalização)
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

    # Carregar pesos
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Modelo carregado com sucesso!")
    print()

    # Criar baseline forecaster (usa csv_for_eval)
    forecaster = BaselineForecaster(
        csv_path=csv_for_eval,
        target_col=cfg["data"]["target_col"],
        seq_len=int(cfg["data"]["seq_len"]),
        pred_len=int(cfg["data"]["pred_len"])
    )

    # Selecionar amostras
    np.random.seed(42)

    if test_csv:
        # Usando CSV de teste específico - usar todas as amostras disponíveis
        print(f"Usando CSV de teste: {test_csv}")
        ds_test = SlidingWindowDataset(
            csv_path=test_csv,
            seq_len=int(cfg["data"]["seq_len"]),
            label_len=int(cfg["data"]["label_len"]),
            pred_len=int(cfg["data"]["pred_len"]),
            target_col=cfg["data"]["target_col"],
            feature_cols=cfg["data"].get("feature_cols")
        )
        # Usar o dataset de teste
        ds = ds_test
        indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
        print(f"Dataset de teste tem {len(ds)} amostras disponíveis")

    elif use_test_set:
        # Usar conjunto de teste do split
        N = len(ds)
        train_ratio = float(cfg["train"]["train_ratio"])
        val_ratio = float(cfg["train"]["val_ratio"])
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        test_start_idx = n_train + n_val
        test_size = N - test_start_idx

        print(f"Usando conjunto de TESTE do split:")
        print(f"  Total: {N} amostras")
        print(f"  Train: {n_train} amostras (0 até {n_train})")
        print(f"  Val: {n_val} amostras ({n_train} até {n_train + n_val})")
        print(f"  Test: {test_size} amostras ({test_start_idx} até {N})")

        # Selecionar apenas do conjunto de teste
        test_indices = np.arange(test_start_idx, N)
        indices = np.random.choice(test_indices, min(num_samples, len(test_indices)), replace=False)

    else:
        # Usar amostras aleatórias de todo o dataset
        print(f"Usando amostras aleatórias de todo o dataset ({len(ds)} amostras)")
        indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)

    print(f"Selecionadas {len(indices)} amostras para avaliação")
    print()

    # Armazenar resultados
    results = {
        'Deep Autoformer': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Naive (último valor)': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Média do contexto': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Média móvel (12 steps)': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Seasonal Naive (1 semana)': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Perfil sazonal': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Média dias similares (4 dias)': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []},
        'Morning-of adjustment': {'predictions': [], 'mae': [], 'rmse': [], 'mape': []}
    }

    all_targets = []
    all_contexts = []

    print("Processando amostras...\n")

    with torch.no_grad():
        for idx in indices:
            # Obter dados do dataset
            enc, dec, y = ds[idx]

            # Adicionar batch dimension
            enc_batch = enc.unsqueeze(0).to(device)
            dec_batch = dec.unsqueeze(0).to(device)

            # Fazer previsão com o modelo
            pred_model = model(enc_batch, dec_batch)

            # Desnormalizar
            pred_model_np = pred_model.cpu().numpy().reshape(-1, 1)
            y_np = y.numpy().reshape(-1, 1)
            enc_np = enc.numpy()[:, 0].reshape(-1, 1)

            pred_model_denorm = ds.inverse_transform_y(pred_model_np).flatten()
            y_denorm = ds.inverse_transform_y(y_np).flatten()
            context_denorm = ds.inverse_transform_y(enc_np).flatten()

            all_contexts.append(context_denorm)
            all_targets.append(y_denorm)

            # Calcular métricas do modelo
            mae_model = mean_absolute_error(y_denorm, pred_model_denorm)
            rmse_model = np.sqrt(mean_squared_error(y_denorm, pred_model_denorm))
            mape_model = np.mean(np.abs((y_denorm - pred_model_denorm) / (y_denorm + 1e-8))) * 100

            results['Deep Autoformer']['predictions'].append(pred_model_denorm)
            results['Deep Autoformer']['mae'].append(mae_model)
            results['Deep Autoformer']['rmse'].append(rmse_model)
            results['Deep Autoformer']['mape'].append(mape_model)

            # Construir perfil sazonal
            profile = forecaster.build_seasonal_profile(idx + int(cfg["data"]["seq_len"]))

            # Gerar previsões com baselines
            baseline_methods = {
                'Naive (último valor)': forecaster.naive_persistence(context_denorm),
                'Média do contexto': forecaster.mean_forecast(context_denorm),
                'Média móvel (12 steps)': forecaster.moving_average(context_denorm, window=12),
                'Seasonal Naive (1 semana)': forecaster.seasonal_naive(idx),
                'Perfil sazonal': forecaster.seasonal_profile(idx, profile),
                'Média dias similares (4 dias)': forecaster.similar_days_average(idx, num_similar_days=4),
                'Morning-of adjustment': forecaster.morning_of_adjustment(idx, num_similar_days=4, morning_window_hours=(10, 13))
            }

            # Calcular métricas dos baselines
            for name, pred in baseline_methods.items():
                mae = mean_absolute_error(y_denorm, pred)
                rmse = np.sqrt(mean_squared_error(y_denorm, pred))
                mape = np.mean(np.abs((y_denorm - pred) / (y_denorm + 1e-8))) * 100

                results[name]['predictions'].append(pred)
                results[name]['mae'].append(mae)
                results[name]['rmse'].append(rmse)
                results[name]['mape'].append(mape)

    # Calcular médias e ordenar por MAE
    print("\nRESULTADOS MÉDIOS:")
    print("-" * 95)
    print(f"{'Rank':<6} {'Método':<30} {'MAE (kW)':<15} {'RMSE (kW)':<15} {'MAPE (%)':<15} {'vs Melhor':<15}")
    print("-" * 95)

    # Calcular médias
    avg_results = []
    for name, data in results.items():
        avg_mae = np.mean(data['mae'])
        avg_rmse = np.mean(data['rmse'])
        avg_mape = np.mean(data['mape'])
        avg_results.append((name, avg_mae, avg_rmse, avg_mape))

    # Ordenar por MAE
    avg_results.sort(key=lambda x: x[1])

    best_mae = avg_results[0][1]

    for rank, (name, avg_mae, avg_rmse, avg_mape) in enumerate(avg_results, 1):
        improvement = ((avg_mae - best_mae) / best_mae) * 100
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        vs_best = f"+{improvement:.1f}%" if rank > 1 else "BEST"

        print(f"{medal} {rank:<4} {name:<30} {avg_mae:<15.4f} {avg_rmse:<15.4f} {avg_mape:<15.2f} {vs_best:<15}")

    print("-" * 95)
    print()

    # Calcular improvement do modelo sobre o melhor baseline
    model_mae = np.mean(results['Deep Autoformer']['mae'])
    baseline_maes = [np.mean(results[name]['mae']) for name in results.keys() if name != 'Deep Autoformer']
    best_baseline_mae = min(baseline_maes)

    improvement_pct = ((best_baseline_mae - model_mae) / best_baseline_mae) * 100

    print(f"ANÁLISE:")
    if improvement_pct > 0:
        print(f"✓ O modelo Deep Autoformer é {improvement_pct:.1f}% melhor que o melhor baseline")
        if improvement_pct > 20:
            print(f"  → Excelente! Melhoria significativa, modelo claramente superior")
        elif improvement_pct > 10:
            print(f"  → Bom! Melhoria considerável, modelo justifica a complexidade")
        elif improvement_pct > 5:
            print(f"  → Razoável. Modelo tem vantagem mas margem é pequena")
        else:
            print(f"  → Marginal. Considere se a complexidade do modelo vale a pena")
    else:
        print(f"✗ O modelo Deep Autoformer é {abs(improvement_pct):.1f}% PIOR que o melhor baseline")
        print(f"  → Atenção! O modelo pode estar com overfitting ou mal configurado")

    print()
    print("="*95)
    print()

    # Plotar comparações detalhadas
    if plot:
        n_plots = min(4, len(indices))
        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4*n_plots))
        if n_plots == 1:
            axes = [axes]

        colors = {
            'Deep Autoformer': 'red',
            'Naive (último valor)': 'orange',
            'Média do contexto': 'purple',
            'Média móvel (12 steps)': 'brown',
            'Seasonal Naive (1 semana)': 'pink',
            'Perfil sazonal': 'cyan',
            'Média dias similares (4 dias)': 'darkblue',
            'Morning-of adjustment': 'darkgreen'
        }

        for i in range(n_plots):
            context = all_contexts[i]
            target = all_targets[i]

            # Eixo temporal
            context_time = np.arange(len(context))
            pred_time = np.arange(len(context), len(context) + len(target))

            # Plotar contexto e target real
            axes[i].plot(context_time, context, label='Contexto',
                        color='blue', linewidth=2, alpha=0.7)
            axes[i].plot(pred_time, target, label='Real',
                        color='green', linewidth=3, alpha=0.9, zorder=10)

            # Plotar previsões (modelo em destaque)
            for name, data in results.items():
                pred = data['predictions'][i]
                mae = data['mae'][i]

                if name == 'Deep Autoformer':
                    # Modelo em destaque
                    axes[i].plot(pred_time, pred,
                               label=f'{name} (MAE: {mae:.2f})',
                               color=colors[name], linewidth=2.5, linestyle='-', alpha=0.9, zorder=9)
                else:
                    # Baselines mais discretos
                    axes[i].plot(pred_time, pred,
                               label=f'{name} (MAE: {mae:.2f})',
                               color=colors.get(name, 'gray'), linewidth=1.2, linestyle='--', alpha=0.6, zorder=5)

            axes[i].axvline(x=len(context), color='gray', linestyle=':', alpha=0.5, linewidth=2)
            axes[i].set_xlabel('Time steps (15 min)')
            axes[i].set_ylabel('Consumo (kW)')
            axes[i].set_title(f'Amostra {i+1} - Deep Autoformer vs Baselines')
            axes[i].legend(loc='best', fontsize=8, ncol=2)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_vs_baselines.png', dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado em: model_vs_baselines.png")

        # Criar gráfico de barras comparativo
        fig2, ax = plt.subplots(figsize=(12, 6))

        methods = [name for name, _, _, _ in avg_results]
        maes = [mae for _, mae, _, _ in avg_results]

        bars = ax.bar(range(len(methods)), maes, color=['red' if m == 'Deep Autoformer' else 'steelblue' for m in methods])

        # Destacar o modelo
        for i, method in enumerate(methods):
            if method == 'Deep Autoformer':
                bars[i].set_color('crimson')
                bars[i].set_edgecolor('darkred')
                bars[i].set_linewidth(2)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('MAE (kW)')
        ax.set_title('Comparação de Performance: Deep Autoformer vs Baselines')
        ax.grid(axis='y', alpha=0.3)

        # Adicionar valores nas barras
        for i, v in enumerate(maes):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig('mae_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Gráfico de barras guardado em: mae_comparison.png")

        plt.show()

    return results, avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparar Deep Autoformer com baselines')
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Caminho para o ficheiro de configuração")
    parser.add_argument("--model", type=str, default="configs/best_model.pt",
                       help="Caminho para o modelo treinado")
    parser.add_argument("--samples", type=int, default=20,
                       help="Número de amostras para avaliar")
    parser.add_argument("--test-csv", type=str, default=None,
                       help="Caminho para CSV de teste específico (opcional)")
    parser.add_argument("--use-all-data", action="store_true",
                       help="Usar amostras aleatórias de todo o dataset em vez de apenas o conjunto de teste")
    parser.add_argument("--no-plot", action="store_true",
                       help="Não plotar os resultados")

    args = parser.parse_args()

    compare_model_with_baselines(
        args.config,
        args.model,
        args.samples,
        not args.no_plot,
        test_csv=args.test_csv,
        use_test_set=not args.use_all_data
    )
