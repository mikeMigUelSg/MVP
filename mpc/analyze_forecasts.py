"""
Script Interativo para Análise de Previsões de Carga e Produção Fotovoltaica

Este script permite visualizar e inspecionar as previsões usadas pelo otimizador MPC.
Clique num ponto para ver as previsões que foram feitas naquele instante.

Uso:
    python3 analyze_forecasts.py [--start "2025-01-01 00:00:00"] [--days 7] [--method mean]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import yaml

from src.components.house import House
from src.components.solar import SolarPanel
from src.forecasters import LoadForecaster, SolarForecaster


class ForecastAnalyzer:
    """
    Analisador interativo de previsões para carga e produção solar
    """

    def __init__(self, config_path='config.yaml'):
        """
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        # Carregar configuração
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parâmetros de simulação
        self.dt_minutes = self.config['simulation']['timestep_minutes']
        self.horizon_steps = self.config['controller']['mpc']['horizon_steps']

        # Criar componentes
        print("Carregando dados...")
        self.house = House(data_file=self.config['house']['data_file'])
        self.solar = SolarPanel(
            capacity_kw=self.config['solar']['capacity_kw'],
            data_file=self.config['solar']['data_file']
        )

        # Criar forecasters
        self.load_forecaster = LoadForecaster(house=self.house)
        self.solar_forecaster = SolarForecaster(solar_panel=self.solar)

        # Estado da análise
        self.current_timestamp = None
        self.timestamps = None
        self.load_real = None
        self.solar_real = None
        self.forecast_method = 'mean'

        # Plot state
        self.fig = None
        self.ax_load = None
        self.ax_solar = None
        self.ax_comparison = None
        self.click_marker_load = None
        self.click_marker_solar = None
        self.forecast_lines_load = []
        self.forecast_lines_solar = []
        self.comparison_lines = []

        print("✓ Componentes carregados")

    def generate_data(self, start_time, n_steps, method='mean'):
        """
        Gera dados reais e previsões para análise

        Args:
            start_time: Timestamp inicial
            n_steps: Número de steps a simular
            method: Método de previsão ('mean', 'naive', 'moving_average')
        """
        self.forecast_method = method
        self.current_timestamp = start_time
        self.timestamps = []
        self.load_real = []
        self.solar_real = []

        print(f"\nGerando dados de {start_time} por {n_steps} steps...")
        print(f"Método de previsão: {method}")

        current_time = start_time
        for i in range(n_steps):
            self.timestamps.append(current_time)
            self.load_real.append(self.house.get_consumption(current_time))
            self.solar_real.append(self.solar.get_production(current_time))
            current_time += timedelta(minutes=self.dt_minutes)

        self.timestamps = np.array(self.timestamps)
        self.load_real = np.array(self.load_real)
        self.solar_real = np.array(self.solar_real)

        print(f"✓ Gerados {n_steps} steps de dados")

    def get_forecast_at_time(self, click_idx):
        """
        Obtém as previsões feitas num determinado instante

        Args:
            click_idx: Índice do tempo clicado

        Returns:
            tuple: (load_forecast, solar_forecast, forecast_timestamps)
        """
        if click_idx < 0 or click_idx >= len(self.timestamps):
            return None, None, None

        forecast_start = self.timestamps[click_idx]

        # Obter previsões
        try:
            load_forecast = self.load_forecaster.get_forecast(
                start_time=forecast_start,
                n_steps=self.horizon_steps,
                dt_minutes=self.dt_minutes,
                method=self.forecast_method,
                context_hours=24
            )
        except Exception as e:
            print(f"Erro ao obter previsão de carga: {e}")
            load_forecast = None

        try:
            solar_forecast = self.solar_forecaster.get_forecast(
                start_time=forecast_start,
                n_steps=self.horizon_steps,
                dt_minutes=self.dt_minutes,
                method=self.forecast_method,
                context_hours=24,
                apply_night_constraint=True
            )
        except Exception as e:
            print(f"Erro ao obter previsão solar: {e}")
            solar_forecast = None

        # Gerar timestamps de previsão
        forecast_timestamps = []
        current_time = forecast_start
        for _ in range(self.horizon_steps):
            forecast_timestamps.append(current_time)
            current_time += timedelta(minutes=self.dt_minutes)

        return load_forecast, solar_forecast, np.array(forecast_timestamps)

    def plot_interactive(self):
        """
        Cria plot interativo com capacidade de clicar para inspecionar previsões
        """
        # Criar figura com subplots
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        self.ax_load = self.fig.add_subplot(gs[0, :])
        self.ax_solar = self.fig.add_subplot(gs[1, :])
        self.ax_comparison = self.fig.add_subplot(gs[2, :])

        # Plot 1: Carga
        self.ax_load.plot(self.timestamps, self.load_real,
                         'b-', linewidth=2, label='Carga Real', alpha=0.8)
        self.ax_load.set_xlabel('Tempo')
        self.ax_load.set_ylabel('Carga (kW)')
        self.ax_load.set_title('Consumo de Carga - Clique num ponto para ver previsões')
        self.ax_load.grid(True, alpha=0.3)
        self.ax_load.legend(loc='upper right')

        # Plot 2: Solar
        self.ax_solar.plot(self.timestamps, self.solar_real,
                          'orange', linewidth=2, label='Produção Real', alpha=0.8)
        self.ax_solar.set_xlabel('Tempo')
        self.ax_solar.set_ylabel('Produção Solar (kW)')
        self.ax_solar.set_title('Produção Fotovoltaica - Clique num ponto para ver previsões')
        self.ax_solar.grid(True, alpha=0.3)
        self.ax_solar.legend(loc='upper right')

        # Plot 3: Comparação (inicialmente vazio)
        self.ax_comparison.set_xlabel('Steps à frente')
        self.ax_comparison.set_ylabel('Potência (kW)')
        self.ax_comparison.set_title('Previsões vs Realidade (clique num ponto acima)')
        self.ax_comparison.grid(True, alpha=0.3)

        # Conectar evento de clique
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.suptitle(f'Análise de Previsões - Método: {self.forecast_method.upper()} | '
                    f'Horizonte: {self.horizon_steps} steps ({self.horizon_steps * self.dt_minutes / 60:.1f}h)',
                    fontsize=14, fontweight='bold')

        print("\n" + "="*80)
        print("MODO INTERATIVO ATIVADO")
        print("="*80)
        print("→ Clique nos gráficos de carga ou solar para inspecionar previsões")
        print("→ A previsão feita naquele momento aparecerá em verde tracejado")
        print("→ O gráfico inferior mostra previsão vs realidade")
        print("="*80 + "\n")

        plt.show()

    def on_click(self, event):
        """
        Handler para eventos de clique no plot

        Args:
            event: Evento de clique do matplotlib
        """
        # Verificar se clicou num dos gráficos principais
        if event.inaxes not in [self.ax_load, self.ax_solar]:
            return

        # Converter clique para índice de timestamp
        click_time = event.xdata
        if click_time is None:
            return

        # Encontrar o timestamp mais próximo
        # Convert matplotlib date to datetime
        from matplotlib.dates import num2date
        click_datetime = num2date(click_time)

        # Remove timezone info to make it compatible with our timestamps
        if click_datetime.tzinfo is not None:
            click_datetime = click_datetime.replace(tzinfo=None)

        # Encontrar índice mais próximo
        time_diffs = [abs((t - click_datetime).total_seconds()) for t in self.timestamps]
        click_idx = np.argmin(time_diffs)

        clicked_timestamp = self.timestamps[click_idx]

        print(f"\n→ Clique detectado em: {clicked_timestamp}")
        print(f"  Gerando previsões com método '{self.forecast_method}'...")

        # Obter previsões feitas naquele momento
        load_forecast, solar_forecast, forecast_timestamps = self.get_forecast_at_time(click_idx)

        # DEBUG: Verificar se previsões são constantes
        if load_forecast is not None:
            load_std = np.std(load_forecast)
            load_unique = len(np.unique(load_forecast))
            print(f"  DEBUG Load: std={load_std:.3f} kW, valores únicos={load_unique}/{len(load_forecast)}")
            if load_std < 0.01:
                print(f"  ⚠️ AVISO: Load forecast é praticamente CONSTANTE!")

        if solar_forecast is not None:
            solar_std = np.std(solar_forecast)
            solar_unique = len(np.unique(solar_forecast))
            print(f"  DEBUG Solar: std={solar_std:.3f} kW, valores únicos={solar_unique}/{len(solar_forecast)}")
            if solar_std < 0.01:
                print(f"  ⚠️ AVISO: Solar forecast é praticamente CONSTANTE!")

        if load_forecast is None or solar_forecast is None:
            print("  ✗ Erro ao obter previsões")
            return

        # Limpar previsões anteriores
        for line in self.forecast_lines_load:
            line.remove()
        for line in self.forecast_lines_solar:
            line.remove()
        for line in self.comparison_lines:
            line.remove()
        self.forecast_lines_load = []
        self.forecast_lines_solar = []
        self.comparison_lines = []

        # Plotar previsões nos gráficos principais
        line_load, = self.ax_load.plot(forecast_timestamps, load_forecast,
                                       'g--', linewidth=2, label=f'Previsão em {clicked_timestamp.strftime("%H:%M")}',
                                       alpha=0.7)
        self.forecast_lines_load.append(line_load)

        line_solar, = self.ax_solar.plot(forecast_timestamps, solar_forecast,
                                         'g--', linewidth=2, label=f'Previsão em {clicked_timestamp.strftime("%H:%M")}',
                                         alpha=0.7)
        self.forecast_lines_solar.append(line_solar)

        # Adicionar marcador no ponto clicado
        if self.click_marker_load:
            self.click_marker_load.remove()
        if self.click_marker_solar:
            self.click_marker_solar.remove()

        self.click_marker_load = self.ax_load.scatter(
            [clicked_timestamp], [self.load_real[click_idx]],
            color='red', s=100, zorder=5, label='Ponto selecionado'
        )
        self.click_marker_solar = self.ax_solar.scatter(
            [clicked_timestamp], [self.solar_real[click_idx]],
            color='red', s=100, zorder=5, label='Ponto selecionado'
        )

        # Atualizar legendas
        self.ax_load.legend(loc='upper right')
        self.ax_solar.legend(loc='upper right')

        # Plotar comparação detalhada
        self.ax_comparison.clear()

        # Extrair realidade correspondente
        end_idx = min(click_idx + self.horizon_steps, len(self.timestamps))
        actual_steps = end_idx - click_idx

        load_real_slice = self.load_real[click_idx:end_idx]
        solar_real_slice = self.solar_real[click_idx:end_idx]

        steps = np.arange(actual_steps)

        # Plot comparação
        line1, = self.ax_comparison.plot(steps, load_real_slice, 'b-', linewidth=2,
                                        label='Carga Real', marker='o', markersize=3)
        line2, = self.ax_comparison.plot(steps, load_forecast[:actual_steps], 'b--', linewidth=2,
                                        label='Carga Prevista', alpha=0.7)
        line3, = self.ax_comparison.plot(steps, solar_real_slice, 'orange', linewidth=2,
                                        label='Solar Real', marker='o', markersize=3)
        line4, = self.ax_comparison.plot(steps, solar_forecast[:actual_steps], 'g--', linewidth=2,
                                        label='Solar Previsto', alpha=0.7)

        self.comparison_lines = [line1, line2, line3, line4]

        # Calcular erros
        load_mae = np.mean(np.abs(load_real_slice - load_forecast[:actual_steps]))
        solar_mae = np.mean(np.abs(solar_real_slice - solar_forecast[:actual_steps]))

        self.ax_comparison.set_xlabel('Steps à frente (cada step = 15 min)')
        self.ax_comparison.set_ylabel('Potência (kW)')
        self.ax_comparison.set_title(
            f'Previsões vs Realidade em {clicked_timestamp.strftime("%Y-%m-%d %H:%M")}\n'
            f'MAE Carga: {load_mae:.3f} kW | MAE Solar: {solar_mae:.3f} kW'
        )
        self.ax_comparison.legend(loc='upper right')
        self.ax_comparison.grid(True, alpha=0.3)

        print(f"  ✓ Previsão visualizada")
        print(f"  MAE Carga: {load_mae:.3f} kW")
        print(f"  MAE Solar: {solar_mae:.3f} kW")
        print(f"  Horizonte efetivo: {actual_steps} steps")

        # Redesenhar
        self.fig.canvas.draw()


def main():
    """
    Função principal
    """
    parser = argparse.ArgumentParser(
        description='Análise interativa de previsões de carga e PV'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--start', type=str, default='2025-01-01 00:00:00',
                       help='Timestamp inicial (formato: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--days', type=float, default=7.0,
                       help='Número de dias a analisar')
    parser.add_argument('--method', type=str, default='historical',
                       choices=['historical', 'mean', 'naive', 'moving_average'],
                       help='Método de previsão')

    args = parser.parse_args()

    # Parse start time
    start_time = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S')
    n_steps = int(args.days * 24 * 60 / 15)  # 15 min timestep

    print("="*80)
    print("ANÁLISE INTERATIVA DE PREVISÕES")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Start:  {start_time}")
    print(f"Days:   {args.days} ({n_steps} steps)")
    print(f"Method: {args.method}")
    print("="*80)

    # Criar analisador
    analyzer = ForecastAnalyzer(config_path=args.config)

    # Gerar dados
    analyzer.generate_data(start_time, n_steps, method=args.method)

    # Criar plot interativo
    analyzer.plot_interactive()


if __name__ == '__main__':
    main()
