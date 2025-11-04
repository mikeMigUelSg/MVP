"""
Script para comparar visualmente todos os métodos de previsão lado a lado
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.components.house import House
from src.components.solar import SolarPanel
from src.forecasters import LoadForecaster, SolarForecaster


def main():
    # Configuração
    start_time = datetime(2025, 1, 15, 12, 0)
    n_steps = 96
    dt_minutes = 15

    # Carregar componentes
    print("Carregando dados...")
    house = House(data_file='data/load/merged_consumos.xlsx')
    solar = SolarPanel(capacity_kw=5.0, data_file='data/solar/pvdata.csv')

    # Criar forecasters
    load_forecaster = LoadForecaster(house=house)
    solar_forecaster = SolarForecaster(solar_panel=solar)

    # Métodos a comparar
    methods = ['historical', 'mean', 'naive', 'moving_average']
    colors = ['blue', 'red', 'green', 'orange']

    print(f"\nComparando métodos para {start_time}")
    print("="*80)

    # Criar figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Carga
    for method, color in zip(methods, colors):
        forecast = load_forecaster.get_forecast(
            start_time=start_time,
            n_steps=n_steps,
            dt_minutes=dt_minutes,
            method=method
        )

        steps = np.arange(n_steps)
        linestyle = '-' if method == 'historical' else '--'
        linewidth = 2.5 if method == 'historical' else 1.5

        ax1.plot(steps, forecast, color=color, linestyle=linestyle,
                linewidth=linewidth, label=f'{method.upper()} (std={np.std(forecast):.3f})',
                alpha=0.8)

        print(f"{method.upper():15s}: mean={np.mean(forecast):.3f} kW, std={np.std(forecast):.3f} kW")

    ax1.set_xlabel('Steps (15 min cada)', fontsize=11)
    ax1.set_ylabel('Carga (kW)', fontsize=11)
    ax1.set_title(f'Comparação de Métodos de Previsão de Carga - {start_time}',
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Solar
    print()
    for method, color in zip(methods, colors):
        forecast = solar_forecaster.get_forecast(
            start_time=start_time,
            n_steps=n_steps,
            dt_minutes=dt_minutes,
            method=method,
            apply_night_constraint=True
        )

        steps = np.arange(n_steps)
        linestyle = '-' if method == 'historical' else '--'
        linewidth = 2.5 if method == 'historical' else 1.5

        ax2.plot(steps, forecast, color=color, linestyle=linestyle,
                linewidth=linewidth, label=f'{method.upper()} (std={np.std(forecast):.3f})',
                alpha=0.8)

        print(f"{method.upper():15s}: mean={np.mean(forecast):.3f} kW, std={np.std(forecast):.3f} kW")

    ax2.set_xlabel('Steps (15 min cada)', fontsize=11)
    ax2.set_ylabel('Produção Solar (kW)', fontsize=11)
    ax2.set_title(f'Comparação de Métodos de Previsão Solar - {start_time}',
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    print("\n" + "="*80)
    print("NOTA:")
    print("  • HISTORICAL (linha sólida): Varia ao longo do tempo - RECOMENDADO para MPC")
    print("  • Outros (linhas tracejadas): Constantes - apenas baselines para comparação")
    print("="*80)

    plt.savefig('forecast_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("\nGráfico salvo: forecast_methods_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
