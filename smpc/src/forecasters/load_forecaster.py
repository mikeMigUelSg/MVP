"""
Load Forecaster - Previsão de carga elétrica

Implementa métodos de previsão para consumo de energia elétrica.
Baseado nos métodos baseline do deep_autoformer_minimal.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os
import sys


class LoadForecaster:
    """
    Forecaster para previsão de carga elétrica.

    Este módulo implementa métodos simples de previsão baseados em
    dados históricos de consumo.
    """

    def __init__(self, house=None, deep_autoformer_config=None, profile_persistence_config=None):
        """
        Args:
            house: Instância da classe House com dados históricos
            deep_autoformer_config: Configuração para o Deep Autoformer
                                   {'model_path': ..., 'config_path': ..., 'data_path': ...}
            profile_persistence_config: Configuração para Profile Persistence
                                       {'n_weeks': 3}
        """
        self.house = house
        self.deep_autoformer_config = deep_autoformer_config
        self.profile_persistence_config = profile_persistence_config or {'n_weeks': 3}
        self._deep_model = None
        self._deep_scaler_y = None
        self._deep_scaler_x = None
        self._deep_cfg = None
        self._profile_forecaster = None

    def mean_forecast(self,
                     context: np.ndarray,
                     n_steps: int) -> np.ndarray:
        """
        Baseline: Média do contexto

        Usa a média dos valores de contexto histórico como previsão constante.
        Este é um método simples mas eficaz para séries temporais estáveis.

        Args:
            context: Array com valores históricos (ex: últimas 96 observações)
            n_steps: Número de steps a prever

        Returns:
            Array com previsão constante baseada na média do contexto

        Exemplo:
            >>> context = np.array([1.0, 1.2, 1.1, 1.3])  # últimas 4 horas
            >>> forecast = forecaster.mean_forecast(context, n_steps=8)
            >>> # Retorna array com 8 valores iguais à média de context
        """
        if len(context) == 0:
            raise ValueError("Context array cannot be empty")

        mean_value = np.mean(context)
        return np.full(n_steps, mean_value)

    def naive_persistence(self,
                         context: np.ndarray,
                         n_steps: int) -> np.ndarray:
        """
        Baseline: Naive/Persistence

        Repete o último valor conhecido para toda a previsão.
        Útil para séries temporais com pouca variação.

        Args:
            context: Array com valores históricos
            n_steps: Número de steps a prever

        Returns:
            Array com previsão constante igual ao último valor
        """
        if len(context) == 0:
            raise ValueError("Context array cannot be empty")

        last_value = context[-1]
        return np.full(n_steps, last_value)

    def moving_average(self,
                      context: np.ndarray,
                      n_steps: int,
                      window: int = 12) -> np.ndarray:
        """
        Baseline: Média móvel

        Usa a média dos últimos N valores como previsão constante.

        Args:
            context: Array com valores históricos
            n_steps: Número de steps a prever
            window: Tamanho da janela para média móvel (default=12, i.e., 3 horas)

        Returns:
            Array com previsão constante baseada na média móvel
        """
        if len(context) == 0:
            raise ValueError("Context array cannot be empty")

        window = min(window, len(context))
        ma_value = np.mean(context[-window:])
        return np.full(n_steps, ma_value)

    def _load_deep_autoformer_model(self):
        """
        Carrega o modelo Deep Autoformer, configuração e scalers (lazy loading).
        """
        if self._deep_model is not None:
            return  # Já carregado

        if self.deep_autoformer_config is None:
            raise ValueError("Deep Autoformer config not provided during initialization")

        try:
            import torch
            import yaml
            import pandas as pd
            from sklearn.preprocessing import StandardScaler

            # Adicionar path do deep_autoformer ao sys.path
            deep_autoformer_path = self.deep_autoformer_config.get('base_path')
            if deep_autoformer_path is None:
                raise ValueError("base_path must be provided in deep_autoformer_config")
            if deep_autoformer_path not in sys.path:
                sys.path.insert(0, deep_autoformer_path)

            from deep_autoformer.model import DeepAutoformer
            from deep_autoformer.dataset import SlidingWindowDataset

            # Carregar configuração
            config_path = self.deep_autoformer_config.get('config_path')
            with open(config_path, 'r') as f:
                self._deep_cfg = yaml.safe_load(f)

            # Detectar device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            # Carregar dataset original para obter scalers
            original_data_path = self.deep_autoformer_config.get('data_path')
            ds_original = SlidingWindowDataset(
                csv_path=original_data_path,
                seq_len=int(self._deep_cfg["data"]["seq_len"]),
                label_len=int(self._deep_cfg["data"]["label_len"]),
                pred_len=int(self._deep_cfg["data"]["pred_len"]),
                target_col=self._deep_cfg["data"]["target_col"],
                feature_cols=self._deep_cfg["data"].get("feature_cols")
            )

            self._deep_scaler_y = ds_original.scaler_y
            self._deep_scaler_x = ds_original.scaler_x

            # Criar modelo
            feature_cols = self._deep_cfg["data"].get("feature_cols", [])
            self._deep_model = DeepAutoformer(
                enc_in=1 + len(feature_cols),
                dec_in=1 + len(feature_cols),
                d_model=int(self._deep_cfg["model"]["d_model"]),
                n_heads=int(self._deep_cfg["model"]["n_heads"]),
                e_layers=int(self._deep_cfg["model"]["e_layers"]),
                d_layers=int(self._deep_cfg["model"]["d_layers"]),
                d_ff=int(self._deep_cfg["model"]["d_ff"]),
                top_k=int(self._deep_cfg["model"]["top_k"]),
                kernel_size=int(self._deep_cfg["model"]["kernel_size"]),
                dropout=float(self._deep_cfg["model"]["dropout"]),
                pred_len=int(self._deep_cfg["data"]["pred_len"]),
                label_len=int(self._deep_cfg["data"]["label_len"]),
                output_dim=1,
                add_deep_mlp=bool(self._deep_cfg["model"]["add_deep_mlp"])
            ).to(device)

            # Carregar pesos
            model_path = self.deep_autoformer_config.get('model_path')
            self._deep_model.load_state_dict(torch.load(model_path, map_location=device))
            self._deep_model.eval()
            self._device = device

            print(f"✓ Deep Autoformer carregado: {model_path}")
            print(f"  Device: {device}")
            print(f"  Seq_len: {self._deep_cfg['data']['seq_len']}, Pred_len: {self._deep_cfg['data']['pred_len']}")

        except ImportError as e:
            raise ImportError(f"Erro ao importar Deep Autoformer. Instale as dependências: {e}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar Deep Autoformer: {e}")

    def _profile_persistence_forecast(self,
                                      start_time: datetime,
                                      n_steps: int,
                                      dt_minutes: int) -> np.ndarray:
        """
        Previsão usando profile persistence (média por dia da semana).

        Args:
            start_time: Timestamp inicial da previsão
            n_steps: Número de steps a prever
            dt_minutes: Duração de cada step

        Returns:
            Array com previsão de carga em kW
        """
        # Lazy loading do profile forecaster
        if self._profile_forecaster is None:
            from .profile_persistence_forecaster import ProfilePersistenceForecaster
            import pandas as pd

            # Criar historical data from house
            if self.house is None or self.house.consumption_data is None:
                raise ValueError("House data required for profile persistence forecast")

            # Criar dataframe com dados históricos
            # Usar dados de várias semanas antes de start_time
            n_weeks = self.profile_persistence_config.get('n_weeks', 3)
            history_days = n_weeks * 7 + 7  # Extra dias para garantir dados

            historical_start = start_time - timedelta(days=history_days)
            timestamps = pd.date_range(
                start=historical_start,
                end=start_time - timedelta(minutes=dt_minutes),
                freq=f'{dt_minutes}min'
            )

            # Obter valores históricos
            consumption_values = []
            for ts in timestamps:
                consumption_values.append(self.house.get_consumption(ts))

            # Criar DataFrame
            historical_data = pd.DataFrame({
                'timestamp': timestamps,
                'consumption': consumption_values
            })

            # Criar forecaster
            self._profile_forecaster = ProfilePersistenceForecaster(
                n_weeks=n_weeks,
                dt_minutes=dt_minutes
            )
            self._profile_forecaster.set_data(historical_data, value_column='consumption')

        # Fazer previsão (set_data renomeia a coluna para 'value')
        forecast = self._profile_forecaster.get_forecast(
            start_time=start_time,
            n_steps=n_steps,
            value_column='value'
        )

        return forecast

    def _deep_autoformer_forecast(self,
                                   start_time: datetime,
                                   n_steps: int,
                                   dt_minutes: int) -> np.ndarray:
        """
        Faz previsão usando o modelo Deep Autoformer.

        Args:
            start_time: Timestamp inicial da previsão
            n_steps: Número de steps a prever
            dt_minutes: Duração de cada step (deve ser 15 min)

        Returns:
            Array com previsão de carga em kW
        """
        import torch
        import pandas as pd

        # Lazy loading do modelo
        self._load_deep_autoformer_model()

        seq_len = int(self._deep_cfg["data"]["seq_len"])
        pred_len = int(self._deep_cfg["data"]["pred_len"])
        label_len = int(self._deep_cfg["data"]["label_len"])
        feature_cols = self._deep_cfg["data"].get("feature_cols", [])

        # Obter contexto histórico (seq_len steps antes de start_time)
        context_steps = seq_len
        context = self._get_historical_context(start_time, context_steps, dt_minutes)

        # Normalizar contexto
        context_scaled = self._deep_scaler_y.transform(context.reshape(-1, 1)).flatten()

        # Preparar contexto (sem features adicionais)
        current_context = context_scaled.reshape(-1, 1)

        # Fazer previsões iterativas até cobrir n_steps
        predictions = []
        n_iterations = (n_steps + pred_len - 1) // pred_len

        with torch.no_grad():
            for iter_idx in range(n_iterations):
                # Preparar encoder input
                enc_input = torch.FloatTensor(current_context[-seq_len:]).to(self._device)

                # Preparar decoder input
                num_features = enc_input.shape[1]
                dec_input = torch.zeros(label_len + pred_len, num_features).to(self._device)

                # Preencher label_len com final do contexto
                dec_input[:label_len, :] = enc_input[-label_len:, :]

                # Adicionar batch dimension
                enc_batch = enc_input.unsqueeze(0)
                dec_batch = dec_input.unsqueeze(0)

                # Fazer previsão
                pred = self._deep_model(enc_batch, dec_batch)
                pred_np = pred.cpu().numpy().flatten()

                # Adicionar às previsões
                predictions.extend(pred_np)

                # Atualizar contexto para próxima iteração
                if iter_idx < n_iterations - 1:  # Não precisa atualizar na última iteração
                    current_context = np.concatenate([current_context, pred_np[:pred_len].reshape(-1, 1)])

        # Pegar apenas os primeiros n_steps previstos
        predictions = np.array(predictions[:n_steps])

        # Desnormalizar
        predictions_denorm = self._deep_scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions_denorm

    def get_forecast(self,
                    start_time: datetime,
                    n_steps: int,
                    dt_minutes: int = 15,
                    method: str = 'mean',
                    context_hours: int = 24) -> np.ndarray:
        """
        Obter previsão de carga para horizonte de planejamento.

        Args:
            start_time: Timestamp inicial da previsão
            n_steps: Número de steps a prever
            dt_minutes: Duração de cada step em minutos (default=15)
            method: Método de previsão ('mean', 'naive', 'moving_average', 'historical', 'profile_persistence', 'deep_autoformer')
            context_hours: Horas de histórico a usar como contexto (default=24)

        Returns:
            Array com previsão de carga em kW
        """
        # Método 'profile_persistence': média por dia da semana
        if method == 'profile_persistence':
            return self._profile_persistence_forecast(start_time, n_steps, dt_minutes)

        # Método 'deep_autoformer': usa modelo Deep Learning treinado
        if method == 'deep_autoformer':
            return self._deep_autoformer_forecast(start_time, n_steps, dt_minutes)

        if self.house is None or self.house.consumption_data is None:
            # Fallback: usar padrão simples se não houver dados
            return self._simple_pattern_forecast(start_time, n_steps, dt_minutes)

        # Método 'historical': usa dados históricos do mesmo período (RECOMENDADO)
        if method == 'historical':
            return self._historical_forecast(start_time, n_steps, dt_minutes)

        # Métodos baseline simples (para comparação) - SEM scaling para evitar instabilidade
        # Obter contexto histórico
        context_steps = int(context_hours * 60 / dt_minutes)
        context = self._get_historical_context(start_time, context_steps, dt_minutes)

        # Aplicar método de previsão baseline
        if method == 'mean':
            # Mean baseline: retorna média constante do contexto
            forecast = self.mean_forecast(context, n_steps)
        elif method == 'naive':
            # Naive: repete último valor conhecido
            forecast = self.naive_persistence(context, n_steps)
        elif method == 'moving_average':
            # Moving average: média móvel constante
            forecast = self.moving_average(context, n_steps, window=12)
        else:
            raise ValueError(f"Unknown forecasting method: {method}")

        return forecast

    def _historical_forecast(self,
                            start_time: datetime,
                            n_steps: int,
                            dt_minutes: int,
                            lookback_days: int = 7) -> np.ndarray:
        """
        Previsão baseada em dados históricos de períodos anteriores similares.
        Usa dados do mesmo horário mas de DIAS ANTERIORES (não do futuro!).

        Por padrão, usa dados de 7 dias atrás (mesmo dia da semana).
        Se não houver dados suficientes, ajusta automaticamente o lookback.

        Args:
            start_time: Timestamp inicial da previsão
            n_steps: Número de steps a prever
            dt_minutes: Duração de cada step
            lookback_days: Quantos dias olhar para trás (default=7, uma semana)

        Returns:
            Array com previsão baseada em histórico passado
        """
        forecast = np.zeros(n_steps)

        # Tentar diferentes lookbacks se o primeiro falhar
        # IMPORTANTE: Tentar 1 dia primeiro (dados mais recentes são mais representativos)
        lookback_options = [1, 2, 3, lookback_days, 14]  # Tentar 1, 2, 3, 7, 14 dias

        for lb in lookback_options:
            historical_start = start_time - timedelta(days=lb)
            current_time = historical_start

            # Testar se há dados válidos (não é só fallback)
            # Testar período maior (24h) para garantir robustez
            test_steps = min(96, n_steps)  # 96 steps = 24 horas
            test_values = []
            test_time = historical_start
            for _ in range(test_steps):
                test_values.append(self.house.get_consumption(test_time))
                test_time += timedelta(minutes=dt_minutes)

            # Se tem variação, os dados existem (não é fallback constante)
            # Aceitar se houver variação razoável ao longo do período testado
            has_variation = len(set(test_values)) > test_steps * 0.1  # Pelo menos 10% valores únicos
            std_dev = np.std(test_values)

            if has_variation or std_dev > 0.01:
                # Dados válidos encontrados, usar este lookback
                current_time = historical_start
                for i in range(n_steps):
                    forecast[i] = self.house.get_consumption(current_time)
                    current_time += timedelta(minutes=dt_minutes)
                return forecast

        # Se chegou aqui, não encontrou dados válidos em nenhum lookback
        # NUNCA usar dados do futuro! Usar padrão simples baseado em histórico disponível
        print(f"  ⚠️ Aviso: Sem dados históricos válidos para {start_time}")
        print(f"     Usando padrão médio do dataset (fallback)")

        # Usar padrão temporal baseado na hora do dia
        # Buscar valores médios por hora dos dados disponíveis
        if self.house.consumption_data is not None:
            forecast_times = []
            current = start_time
            for _ in range(n_steps):
                forecast_times.append(current)
                current += timedelta(minutes=dt_minutes)

            # Para cada step, usar média da hora correspondente
            for i in range(n_steps):
                hour = forecast_times[i].hour
                minute = forecast_times[i].minute

                # Buscar média dessa hora em todo o dataset
                hour_data = self.house.consumption_data[
                    (self.house.consumption_data['hour'] == hour) &
                    (self.house.consumption_data['minute'] == minute)
                ]

                if len(hour_data) > 0:
                    forecast[i] = hour_data['consumption'].mean()
                else:
                    # Se não houver dados para essa hora, usar média geral
                    forecast[i] = self.house.consumption_data['consumption'].mean()
        else:
            # Sem dados, usar padrão simples
            forecast = self._simple_pattern_forecast(start_time, n_steps, dt_minutes)

        return forecast

    def _get_historical_context(self,
                               start_time: datetime,
                               n_steps: int,
                               dt_minutes: int) -> np.ndarray:
        """
        Obter dados históricos antes do start_time.

        Args:
            start_time: Timestamp de referência
            n_steps: Número de steps históricos
            dt_minutes: Duração de cada step

        Returns:
            Array com dados históricos
        """
        context = np.zeros(n_steps)
        current_time = start_time - timedelta(minutes=dt_minutes * n_steps)

        for i in range(n_steps):
            context[i] = self.house.get_consumption(current_time)
            current_time += timedelta(minutes=dt_minutes)

        return context

    def _simple_pattern_forecast(self,
                                 start_time: datetime,
                                 n_steps: int,
                                 dt_minutes: int) -> np.ndarray:
        """
        Padrão simples de consumo quando não há dados históricos.

        Args:
            start_time: Timestamp inicial
            n_steps: Número de steps
            dt_minutes: Duração de cada step

        Returns:
            Array com padrão simples de consumo
        """
        forecast = np.zeros(n_steps)
        current_time = start_time

        for i in range(n_steps):
            hour = current_time.hour
            # Consumo maior durante o dia, menor à noite
            if 7 <= hour < 23:
                base_load = 0.5
                variable_load = 1.0
            else:
                base_load = 0.3
                variable_load = 0.2

            forecast[i] = base_load + np.random.uniform(0, variable_load)
            current_time += timedelta(minutes=dt_minutes)

        return forecast
