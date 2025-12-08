"""
Métodos baseline simples para previsão de consumo.
Servem como comparação para avaliar o desempenho do modelo Deep Autoformer.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from deep_autoformer.dataset import SlidingWindowDataset


class BaselineForecaster:
    """
    Conjunto de métodos baseline para previsão de séries temporais
    """

    def __init__(self, csv_path, target_col="target", seq_len=96, pred_len=96):
        """
        Args:
            csv_path: caminho para o ficheiro CSV com os dados
            target_col: nome da coluna target
            seq_len: comprimento da sequência de entrada (contexto)
            pred_len: comprimento da previsão
        """
        self.df = pd.read_csv(csv_path, parse_dates=[0])
        self.df = self.df.sort_values(self.df.columns[0])
        self.target_col = target_col
        self.target = self.df[target_col].values
        self.timestamps = self.df[self.df.columns[0]]
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Criar features temporais para baseline sazonal
        self.df['hour'] = self.timestamps.dt.hour
        self.df['minute'] = self.timestamps.dt.minute
        self.df['dayofweek'] = self.timestamps.dt.dayofweek
        self.df['time_of_day'] = self.df['hour'] * 60 + self.df['minute']  # minutos desde meia-noite

    def naive_persistence(self, context):
        """
        Baseline 1: Naive/Persistence
        Repete o último valor conhecido para toda a previsão
        """
        last_value = context[-1]
        return np.full(self.pred_len, last_value)

    def mean_forecast(self, context):
        """
        Baseline 2: Média do contexto
        Usa a média dos valores de contexto como previsão constante
        """
        mean_value = np.mean(context)
        return np.full(self.pred_len, mean_value)

    def moving_average(self, context, window=12):
        """
        Baseline 3: Média móvel
        Usa a média dos últimos N valores como previsão constante
        """
        window = min(window, len(context))
        ma_value = np.mean(context[-window:])
        return np.full(self.pred_len, ma_value)

    def seasonal_naive(self, idx):
        """
        Baseline 4: Seasonal Naive
        Repete o padrão do mesmo período sazonal (mesmo dia da semana, mesma hora)
        Usa dados de 1 semana atrás (96 steps/dia * 7 dias = 672 steps)
        """
        seasonal_lag = 96 * 7  # 1 semana atrás (assumindo dados de 15 em 15 min)
        start_idx = idx + self.seq_len
        prediction = []

        for i in range(self.pred_len):
            hist_idx = start_idx + i - seasonal_lag
            if hist_idx >= 0 and hist_idx < len(self.target):
                prediction.append(self.target[hist_idx])
            else:
                # Se não houver dados históricos, usar o último valor conhecido
                if len(prediction) > 0:
                    prediction.append(prediction[-1])
                else:
                    prediction.append(self.target[idx + self.seq_len - 1])

        return np.array(prediction)

    def seasonal_profile(self, idx, profile_dict):
        """
        Baseline 5: Perfil sazonal médio
        Usa um perfil médio baseado no dia da semana e hora do dia
        """
        start_idx = idx + self.seq_len
        prediction = []

        for i in range(self.pred_len):
            future_idx = start_idx + i
            if future_idx < len(self.df):
                dow = self.df.iloc[future_idx]['dayofweek']
                tod = self.df.iloc[future_idx]['time_of_day']
                key = (dow, tod)

                if key in profile_dict:
                    prediction.append(profile_dict[key])
                else:
                    # Fallback: usar média global
                    prediction.append(np.mean(self.target[:idx + self.seq_len]))
            else:
                prediction.append(prediction[-1] if prediction else self.target[idx + self.seq_len - 1])

        return np.array(prediction)

    def build_seasonal_profile(self, end_idx):
        """
        Constrói um perfil médio para cada combinação de (dia da semana, hora do dia)
        usando apenas dados até end_idx (para evitar data leakage)
        """
        profile = {}
        df_train = self.df.iloc[:end_idx]

        grouped = df_train.groupby(['dayofweek', 'time_of_day'])[self.target_col].mean()

        for (dow, tod), value in grouped.items():
            profile[(dow, tod)] = value

        return profile

    def similar_days_average(self, idx, num_similar_days=4):
        """
        Baseline 6: Média de dias semelhantes
        Pega nos últimos M dias com o mesmo dia-da-semana e faz média hora a hora

        Args:
            idx: índice atual no dataset
            num_similar_days: número de dias similares a usar (default=4)
        """
        start_idx = idx + self.seq_len

        # Determinar o dia da semana que queremos prever
        if start_idx >= len(self.df):
            return np.full(self.pred_len, self.target[idx + self.seq_len - 1])

        target_dow = self.df.iloc[start_idx]['dayofweek']

        # Encontrar os últimos M dias com o mesmo dia-da-semana (antes de start_idx)
        similar_day_indices = []
        days_back = 0
        max_lookback = min(start_idx, 96 * 365)  # Máximo 1 ano atrás

        for i in range(start_idx - 96, max(0, start_idx - max_lookback), -96):  # Voltar dia a dia (96 steps = 1 dia)
            if i < 0:
                break
            if self.df.iloc[i]['dayofweek'] == target_dow:
                similar_day_indices.append(i)
                days_back += 1
                if days_back >= num_similar_days:
                    break

        if len(similar_day_indices) == 0:
            # Fallback: usar seasonal naive
            return self.seasonal_naive(idx)

        # Fazer média hora a hora dos dias similares
        prediction = []
        for i in range(self.pred_len):
            future_idx = start_idx + i
            values = []

            for similar_idx in similar_day_indices:
                offset_idx = similar_idx + i
                if offset_idx < len(self.target):
                    values.append(self.target[offset_idx])

            if len(values) > 0:
                prediction.append(np.mean(values))
            else:
                prediction.append(prediction[-1] if prediction else self.target[idx + self.seq_len - 1])

        return np.array(prediction)

    def morning_of_adjustment(self, idx, num_similar_days=4, morning_window_hours=(10, 13)):
        """
        Baseline 7: Ajuste "morning-of"
        Usa previsão de dias similares mas ajusta com base nas primeiras horas do dia atual

        Args:
            idx: índice atual no dataset
            num_similar_days: número de dias similares para baseline
            morning_window_hours: janela temporal para calcular fator de ajuste (tupla com hora início e fim)

        Fórmula:
            A = média(y_medido[morning]) / média(y_base[morning])
            y_ajustado = clip(A, 0.8, 1.2) * y_base
        """
        # Obter previsão base (média de dias similares)
        base_prediction = self.similar_days_average(idx, num_similar_days)

        start_idx = idx + self.seq_len

        # Verificar se temos dados do "morning window" disponíveis
        # O morning window deve estar dentro do contexto disponível (antes de start_idx)
        morning_start_hour, morning_end_hour = morning_window_hours

        # Converter horas para steps (assumindo 96 steps por dia, 1 step = 15 min)
        steps_per_hour = 4
        morning_start_step = morning_start_hour * steps_per_hour
        morning_end_step = morning_end_hour * steps_per_hour

        # Encontrar o início do dia atual (última meia-noite)
        # Assumindo que temos dados de 15 em 15 minutos
        current_time_of_day = self.df.iloc[start_idx - 1]['time_of_day'] if start_idx > 0 else 0
        current_step_in_day = int(current_time_of_day / 15)  # minutos -> steps

        # Calcular onde está a última meia-noite
        day_start_idx = start_idx - 1 - current_step_in_day

        # Verificar se o morning window está disponível (já passou)
        if day_start_idx + morning_end_step > start_idx:
            # Morning window ainda não passou completamente, não podemos usar
            # Retornar apenas a previsão base
            return base_prediction

        # Extrair valores reais do morning window
        morning_real = []
        morning_base = []

        for step in range(morning_start_step, morning_end_step):
            real_idx = day_start_idx + step
            if 0 <= real_idx < len(self.target) and real_idx < start_idx:
                morning_real.append(self.target[real_idx])

                # Para a previsão base no mesmo horário, usar o perfil de dias similares
                # Reconstruir a previsão base para esse horário
                dow = self.df.iloc[real_idx]['dayofweek']
                tod = self.df.iloc[real_idx]['time_of_day']

                # Procurar dias similares para esse step
                similar_values = []
                for similar_day in range(1, num_similar_days + 1):
                    lookback_idx = real_idx - (similar_day * 7 * 96)  # Dias da semana anteriores
                    if 0 <= lookback_idx < len(self.target):
                        similar_values.append(self.target[lookback_idx])

                if len(similar_values) > 0:
                    morning_base.append(np.mean(similar_values))

        # Calcular fator de ajuste
        if len(morning_real) > 0 and len(morning_base) > 0 and np.mean(morning_base) > 0:
            adjustment_factor = np.mean(morning_real) / np.mean(morning_base)
            # Aplicar clip de ±20%
            adjustment_factor = np.clip(adjustment_factor, 0.8, 1.2)
        else:
            adjustment_factor = 1.0

        # Aplicar ajuste à previsão base
        adjusted_prediction = adjustment_factor * base_prediction

        return adjusted_prediction


def evaluate_baselines(csv_path, config_path, num_samples=20, plot=True):
    """
    Avalia todos os métodos baseline no dataset
    """
    # Carregar configuração
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seq_len = int(cfg["data"]["seq_len"])
    pred_len = int(cfg["data"]["pred_len"])
    target_col = cfg["data"]["target_col"]

    print("="*80)
    print("BASELINE FORECASTING EVALUATION")
    print("="*80)
    print(f"Dataset: {csv_path}")
    print(f"Sequence length: {seq_len} steps ({seq_len/96:.1f} dias)")
    print(f"Prediction length: {pred_len} steps ({pred_len/96:.1f} dias)")
    print(f"Number of samples: {num_samples}")
    print("="*80)
    print()

    # Criar baseline forecaster
    forecaster = BaselineForecaster(csv_path, target_col, seq_len, pred_len)

    # Selecionar amostras aleatórias
    max_idx = len(forecaster.target) - seq_len - pred_len
    np.random.seed(42)
    indices = np.random.choice(max_idx, min(num_samples, max_idx), replace=False)

    # Armazenar resultados
    results = {
        'Naive (último valor)': {'predictions': [], 'mae': [], 'rmse': []},
        'Média do contexto': {'predictions': [], 'mae': [], 'rmse': []},
        'Média móvel (12 steps)': {'predictions': [], 'mae': [], 'rmse': []},
        'Seasonal Naive (1 semana)': {'predictions': [], 'mae': [], 'rmse': []},
        'Perfil sazonal': {'predictions': [], 'mae': [], 'rmse': []},
        'Média dias similares (4 dias)': {'predictions': [], 'mae': [], 'rmse': []},
        'Morning-of adjustment': {'predictions': [], 'mae': [], 'rmse': []}
    }

    all_targets = []
    all_contexts = []

    print("Processando amostras...\n")

    for idx in indices:
        # Extrair contexto e target
        context = forecaster.target[idx:idx + seq_len]
        target = forecaster.target[idx + seq_len:idx + seq_len + pred_len]

        all_contexts.append(context)
        all_targets.append(target)

        # Construir perfil sazonal apenas com dados até idx + seq_len
        profile = forecaster.build_seasonal_profile(idx + seq_len)

        # Gerar previsões com cada método
        methods = {
            'Naive (último valor)': forecaster.naive_persistence(context),
            'Média do contexto': forecaster.mean_forecast(context),
            'Média móvel (12 steps)': forecaster.moving_average(context, window=12),
            'Seasonal Naive (1 semana)': forecaster.seasonal_naive(idx),
            'Perfil sazonal': forecaster.seasonal_profile(idx, profile),
            'Média dias similares (4 dias)': forecaster.similar_days_average(idx, num_similar_days=4),
            'Morning-of adjustment': forecaster.morning_of_adjustment(idx, num_similar_days=4, morning_window_hours=(10, 13))
        }

        # Calcular métricas
        for name, pred in methods.items():
            mae = mean_absolute_error(target, pred)
            rmse = np.sqrt(mean_squared_error(target, pred))

            results[name]['predictions'].append(pred)
            results[name]['mae'].append(mae)
            results[name]['rmse'].append(rmse)

    # Mostrar resultados agregados
    print("\nRESULTADOS MÉDIOS:")
    print("-" * 80)
    print(f"{'Método':<30} {'MAE (kW)':<15} {'RMSE (kW)':<15}")
    print("-" * 80)

    for name, data in results.items():
        avg_mae = np.mean(data['mae'])
        avg_rmse = np.mean(data['rmse'])
        print(f"{name:<30} {avg_mae:<15.2f} {avg_rmse:<15.2f}")

    print("-" * 80)
    print()

    # Plotar alguns exemplos
    if plot:
        n_plots = min(3, len(indices))
        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4*n_plots))
        if n_plots == 1:
            axes = [axes]

        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'darkblue', 'darkgreen']

        for i in range(n_plots):
            context = all_contexts[i]
            target = all_targets[i]

            # Eixo temporal
            context_time = np.arange(len(context))
            pred_time = np.arange(len(context), len(context) + len(target))

            # Plotar contexto e target real
            axes[i].plot(context_time, context, label='Contexto (histórico)',
                        color='blue', linewidth=2)
            axes[i].plot(pred_time, target, label='Real',
                        color='green', linewidth=2.5, alpha=0.8)

            # Plotar previsões dos baselines
            for j, (name, data) in enumerate(results.items()):
                pred = data['predictions'][i]
                mae = data['mae'][i]
                axes[i].plot(pred_time, pred,
                           label=f'{name} (MAE: {mae:.2f})',
                           color=colors[j], linewidth=1.5, linestyle='--', alpha=0.7)

            axes[i].axvline(x=len(context), color='gray', linestyle=':', alpha=0.5, linewidth=2)
            axes[i].set_xlabel('Time steps (15 min)')
            axes[i].set_ylabel('Consumo (kW)')
            axes[i].set_title(f'Amostra {i+1} - Comparação Baselines')
            axes[i].legend(loc='best', fontsize=9)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('baseline_predictions.png', dpi=150)
        print(f"Gráfico guardado em: baseline_predictions.png")
        plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Avaliar métodos baseline para previsão')
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Caminho para o ficheiro de configuração")
    parser.add_argument("--samples", type=int, default=20,
                       help="Número de amostras para avaliar")
    parser.add_argument("--no-plot", action="store_true",
                       help="Não plotar os resultados")

    args = parser.parse_args()

    # Carregar path do CSV da config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg["data"]["csv_path"]

    evaluate_baselines(csv_path, args.config, args.samples, not args.no_plot)
