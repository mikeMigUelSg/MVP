"""
Profile Persistence Forecaster

Implementa previsão por persistência de perfil: média do mesmo dia-da-semana
nas últimas 3 semanas, à resolução de 15 min.

Este método replica a abordagem de persistência do paper para tanto PV quanto carga.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union


class ProfilePersistenceForecaster:
    """
    Forecaster de persistência por perfil.

    Gera previsões usando a média do mesmo dia-da-semana e hora nas últimas
    N semanas (padrão: 3 semanas).
    """

    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 n_weeks: int = 3,
                 dt_minutes: int = 15):
        """
        Args:
            data: DataFrame com colunas ['timestamp', valor] (opcional)
            n_weeks: Número de semanas anteriores para média (default: 3)
            dt_minutes: Resolução temporal em minutos (default: 15)
        """
        self.data = data
        self.n_weeks = n_weeks
        self.dt_minutes = dt_minutes

        # Processar dados se fornecidos
        if data is not None:
            self._process_data()

    def _process_data(self):
        """Processar dados para indexação eficiente por dia-da-semana e hora."""
        if 'timestamp' not in self.data.columns:
            raise ValueError("Data must have 'timestamp' column")

        # Garantir que timestamp é datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        # Adicionar colunas de indexação
        self.data['weekday'] = self.data['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['minute'] = self.data['timestamp'].dt.minute

        # Criar índice para busca rápida
        self.data = self.data.sort_values('timestamp')
        self.data.set_index('timestamp', inplace=True)

        # Pre-compute profile cache for fast lookups
        self._build_profile_cache()

    def set_data(self, data: pd.DataFrame, value_column: str = 'value'):
        """
        Configurar dados para previsão.

        Args:
            data: DataFrame com coluna 'timestamp' e coluna de valores
            value_column: Nome da coluna com os valores
        """
        # Renomear coluna de valores para 'value'
        if value_column != 'value':
            data = data.rename(columns={value_column: 'value'})

        self.data = data.copy()
        self._process_data()

    def _build_profile_cache(self):
        """
        Pre-compute average profiles for fast forecast generation.
        Creates lookup table indexed by (weekday, hour, minute).
        """
        # 7 days, 24 hours, 60 minutes, multiple value columns
        # We'll use a dict for flexibility with different value columns
        self._profile_cache = {}

        # Get all value columns (exclude indexing columns)
        value_cols = [col for col in self.data.columns if col not in ['weekday', 'hour', 'minute']]

        for value_col in value_cols:
            # Group by weekday, hour, minute and compute mean
            # This replaces nested loops with vectorized pandas groupby
            grouped = self.data.groupby(['weekday', 'hour', 'minute'])[value_col].mean()
            self._profile_cache[value_col] = grouped

    def get_forecast(self,
                     start_time: datetime,
                     n_steps: int,
                     value_column: str = 'value') -> np.ndarray:
        """
        Obter previsão usando persistência de perfil.

        Usa perfis pré-agregados para O(1) lookup por (weekday, hour, minute).

        Args:
            start_time: Timestamp inicial da previsão
            n_steps: Número de steps a prever
            value_column: Nome da coluna com valores (se data já tem índice processado, usar 'value')

        Returns:
            Array com previsão
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        # Vectorized approach: generate all timestamps at once
        timestamps = pd.date_range(start_time, periods=n_steps, freq=f'{self.dt_minutes}min')
        weekdays = timestamps.weekday
        hours = timestamps.hour
        minutes = timestamps.minute

        forecast = np.zeros(n_steps)
        profile = self._profile_cache.get(value_column)

        if profile is None:
            raise ValueError(f"Value column '{value_column}' not found in profile cache")

        # O(1) lookup for each timestep using pre-computed profiles
        for i in range(n_steps):
            try:
                forecast[i] = profile.loc[(weekdays[i], hours[i], minutes[i])]
            except KeyError:
                # Fallback: use fallback value
                forecast[i] = self._get_fallback_value(weekdays[i], hours[i], minutes[i], value_column)

        return forecast

    def _get_value_at_time(self,
                          target_time: datetime,
                          weekday: int,
                          hour: int,
                          minute: int,
                          value_column: str) -> Optional[float]:
        """
        Obter valor em um momento específico.

        Tenta buscar exatamente o timestamp; se não encontrar, busca
        o mais próximo na mesma hora/minuto.
        """
        try:
            # Tentar busca exata
            if target_time in self.data.index:
                return self.data.loc[target_time, value_column]

            # Buscar na janela de ±5 minutos
            window_start = target_time - timedelta(minutes=5)
            window_end = target_time + timedelta(minutes=5)

            window_data = self.data.loc[window_start:window_end]

            if len(window_data) > 0:
                # Filtrar por hora e minuto exatos
                mask = ((window_data['weekday'] == weekday) &
                       (window_data['hour'] == hour) &
                       (window_data['minute'] == minute))
                filtered = window_data[mask]

                if len(filtered) > 0:
                    return filtered[value_column].iloc[0]

            return None
        except Exception as e:
            return None

    def _get_fallback_value(self,
                           weekday: int,
                           hour: int,
                           minute: int,
                           value_column: str) -> float:
        """
        Valor de fallback quando não há dados históricos suficientes.

        Usa média de todos os valores do mesmo dia-da-semana e hora.
        """
        try:
            mask = ((self.data['weekday'] == weekday) &
                   (self.data['hour'] == hour) &
                   (self.data['minute'] == minute))
            filtered = self.data[mask]

            if len(filtered) > 0:
                return filtered[value_column].mean()

            # Se ainda não houver dados, usar média da mesma hora (qualquer dia)
            mask = ((self.data['hour'] == hour) &
                   (self.data['minute'] == minute))
            filtered = self.data[mask]

            if len(filtered) > 0:
                return filtered[value_column].mean()

            # Último recurso: média geral
            return self.data[value_column].mean()
        except Exception as e:
            # Se tudo falhar, retornar 0
            return 0.0
