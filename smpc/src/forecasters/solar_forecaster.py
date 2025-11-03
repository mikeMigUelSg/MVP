"""
Solar Forecaster - Previsão de produção fotovoltaica

Implementa métodos de previsão para produção de energia solar.
Baseado nos métodos baseline do deep_autoformer_minimal.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class SolarForecaster:
    """
    Forecaster para previsão de produção fotovoltaica.

    Este módulo implementa métodos simples de previsão baseados em
    dados históricos de produção solar.
    """

    def __init__(self, solar_panel=None):
        """
        Args:
            solar_panel: Instância da classe SolarPanel com dados históricos
        """
        self.solar_panel = solar_panel

    def mean_forecast(self,
                     context: np.ndarray,
                     n_steps: int) -> np.ndarray:
        """
        Baseline: Média do contexto

        Usa a média dos valores de contexto histórico como previsão constante.
        Para produção solar, este método funciona bem em condições estáveis.

        Args:
            context: Array com valores históricos (ex: últimas 96 observações)
            n_steps: Número de steps a prever

        Returns:
            Array com previsão constante baseada na média do contexto

        Exemplo:
            >>> context = np.array([2.0, 2.5, 3.0, 2.8])  # últimas 4 horas
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
        Para produção solar, funciona bem em horizontes curtos com céu limpo.

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

    def zero_at_night(self,
                     forecast: np.ndarray,
                     start_time: datetime,
                     dt_minutes: int = 15) -> np.ndarray:
        """
        Ajusta previsão para zero durante a noite.

        Aplica conhecimento físico: produção solar é zero à noite.

        Args:
            forecast: Previsão inicial
            start_time: Timestamp inicial
            dt_minutes: Duração de cada step

        Returns:
            Previsão ajustada com zeros à noite
        """
        adjusted_forecast = forecast.copy()
        current_time = start_time

        for i in range(len(forecast)):
            hour = current_time.hour
            # Produção zero entre 20h e 7h
            if hour < 7 or hour >= 20:
                adjusted_forecast[i] = 0.0
            current_time += timedelta(minutes=dt_minutes)

        return adjusted_forecast

    def get_forecast(self,
                    start_time: datetime,
                    n_steps: int,
                    dt_minutes: int = 15,
                    method: str = 'mean',
                    context_hours: int = 24,
                    apply_night_constraint: bool = True) -> np.ndarray:
        """
        Obter previsão de produção solar para horizonte de planejamento.

        Args:
            start_time: Timestamp inicial da previsão
            n_steps: Número de steps a prever
            dt_minutes: Duração de cada step em minutos (default=15)
            method: Método de previsão ('mean', 'naive', 'moving_average', 'historical')
            context_hours: Horas de histórico a usar como contexto (default=24)
            apply_night_constraint: Se True, força produção zero à noite (default=True)

        Returns:
            Array com previsão de produção em kW
        """
        if self.solar_panel is None or self.solar_panel.production_data is None:
            # Fallback: usar padrão simples se não houver dados
            return self._simple_pattern_forecast(start_time, n_steps, dt_minutes)

        # Método 'historical': usa dados históricos do mesmo período (RECOMENDADO)
        if method == 'historical':
            forecast = self._historical_forecast(start_time, n_steps, dt_minutes)
            if apply_night_constraint:
                forecast = self.zero_at_night(forecast, start_time, dt_minutes)
            return forecast

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

        # Aplicar restrição física: zero à noite
        if apply_night_constraint:
            forecast = self.zero_at_night(forecast, start_time, dt_minutes)

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
            # Para solar: testar período maior (24h) para cobrir dia e noite
            test_steps = min(96, n_steps)  # 96 steps = 24 horas
            test_values = []
            test_time = historical_start
            for _ in range(test_steps):
                test_values.append(self.solar_panel.get_production(test_time))
                test_time += timedelta(minutes=dt_minutes)

            # Se tem variação, os dados existem (não é fallback constante)
            # Para solar, verificar se há produção em algum momento do período testado
            non_zero_count = sum(1 for v in test_values if v > 0.01)
            has_variation = len(set(test_values)) > 2

            # Aceitar se houver variação OU se houver produção não-zero em pelo menos 10% dos pontos
            # (isso garante que zeros noturnos sejam aceitos se houver dados de dia)
            if has_variation or non_zero_count > test_steps * 0.1:
                # Dados válidos encontrados, usar este lookback
                current_time = historical_start
                for i in range(n_steps):
                    forecast[i] = self.solar_panel.get_production(current_time)
                    current_time += timedelta(minutes=dt_minutes)
                return forecast

        # Se chegou aqui, não encontrou dados válidos em nenhum lookback
        # NUNCA usar dados do futuro! Usar padrão simples baseado em histórico disponível
        print(f"  ⚠️ Aviso: Sem dados históricos válidos para {start_time}")
        print(f"     Usando padrão médio do dataset (fallback)")

        # Usar padrão temporal baseado na hora do dia
        if self.solar_panel.production_data is not None:
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
                hour_data = self.solar_panel.production_data[
                    (self.solar_panel.production_data['hour'] == hour) &
                    (self.solar_panel.production_data['minute'] == minute)
                ]

                if len(hour_data) > 0:
                    forecast[i] = hour_data['pv_1'].mean()
                else:
                    # Se não houver dados para essa hora, zero (noite)
                    forecast[i] = 0.0
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
            context[i] = self.solar_panel.get_production(current_time)
            current_time += timedelta(minutes=dt_minutes)

        return context

    def _simple_pattern_forecast(self,
                                 start_time: datetime,
                                 n_steps: int,
                                 dt_minutes: int) -> np.ndarray:
        """
        Padrão simples de produção quando não há dados históricos.

        Args:
            start_time: Timestamp inicial
            n_steps: Número de steps
            dt_minutes: Duração de cada step

        Returns:
            Array com padrão simples de produção
        """
        forecast = np.zeros(n_steps)
        current_time = start_time
        capacity_kw = self.solar_panel.capacity_kw if self.solar_panel else 5.0

        for i in range(n_steps):
            hour = current_time.hour
            # Produção zero à noite
            if hour < 7 or hour > 19:
                forecast[i] = 0.0
            else:
                # Modelo simples: onda senoidal
                hour_angle = (hour - 7) / 12 * np.pi
                forecast[i] = capacity_kw * np.sin(hour_angle)

            current_time += timedelta(minutes=dt_minutes)

        return forecast
