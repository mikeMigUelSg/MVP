"""
Script de teste para os forecasters de carga e produção solar

Este script demonstra como usar os forecasters integrados no MPC.
"""

import numpy as np
from datetime import datetime
from src.components.house import House
from src.components.solar import SolarPanel
from src.forecasters import LoadForecaster, SolarForecaster


def test_forecasters():
    """
    Testa os forecasters de carga e produção solar
    """
    print("=" * 80)
    print("TESTE DOS FORECASTERS")
    print("=" * 80)
    print()

    # Configuração
    start_time = datetime(2025, 1, 15, 12, 0)  # 15 Jan 2025, 12:00
    n_steps = 96  # 24 horas (96 steps de 15 minutos)
    dt_minutes = 15

    print(f"Timestamp inicial: {start_time}")
    print(f"Horizonte: {n_steps} steps ({n_steps * dt_minutes / 60:.1f} horas)")
    print()

    # Teste 1: Load Forecaster
    print("-" * 80)
    print("TESTE 1: Load Forecaster")
    print("-" * 80)

    try:
        house = House(data_file="data/load/merged_consumos.xlsx")
        load_forecaster = LoadForecaster(house=house)

        methods = ['mean', 'naive', 'moving_average']
        for method in methods:
            forecast = load_forecaster.get_forecast(
                start_time=start_time,
                n_steps=n_steps,
                dt_minutes=dt_minutes,
                method=method,
                context_hours=24
            )

            print(f"\nMétodo: {method}")
            print(f"  Shape: {forecast.shape}")
            print(f"  Mean:  {np.mean(forecast):.3f} kW")
            print(f"  Min:   {np.min(forecast):.3f} kW")
            print(f"  Max:   {np.max(forecast):.3f} kW")
            print(f"  Std:   {np.std(forecast):.3f} kW")

    except Exception as e:
        print(f"Erro no Load Forecaster: {e}")

    print()

    # Teste 2: Solar Forecaster
    print("-" * 80)
    print("TESTE 2: Solar Forecaster")
    print("-" * 80)

    try:
        solar = SolarPanel(capacity_kw=5.0, data_file="data/solar/pvdata.csv")
        solar_forecaster = SolarForecaster(solar_panel=solar)

        methods = ['mean', 'naive', 'moving_average']
        for method in methods:
            forecast = solar_forecaster.get_forecast(
                start_time=start_time,
                n_steps=n_steps,
                dt_minutes=dt_minutes,
                method=method,
                context_hours=24,
                apply_night_constraint=True
            )

            print(f"\nMétodo: {method}")
            print(f"  Shape: {forecast.shape}")
            print(f"  Mean:  {np.mean(forecast):.3f} kW")
            print(f"  Min:   {np.min(forecast):.3f} kW")
            print(f"  Max:   {np.max(forecast):.3f} kW")
            print(f"  Std:   {np.std(forecast):.3f} kW")

            # Verificar se zero à noite está funcionando
            zeros = np.sum(forecast == 0.0)
            print(f"  Zeros: {zeros}/{n_steps} ({zeros/n_steps*100:.1f}%)")

    except Exception as e:
        print(f"Erro no Solar Forecaster: {e}")

    print()

    # Teste 3: Integração com MPC
    print("-" * 80)
    print("TESTE 3: Integração com MPC Controller")
    print("-" * 80)

    try:
        from src.controllers.mpc_controller import MPCController

        # Criar MPC com forecasters
        mpc = MPCController(
            horizon_steps=96,
            dt_minutes=15,
            export_price=0.05,
            contracted_power_kw=6.9,
            forecast_method='mean'
        )

        print("\nMPC Controller criado com sucesso!")
        print(f"  Horizonte: {mpc.horizon_steps} steps")
        print(f"  Método de previsão: {mpc.forecast_method}")
        print(f"  Forecasters: {'Inicializados' if mpc.load_forecaster else 'Serão inicializados automaticamente'}")

    except Exception as e:
        print(f"Erro na integração com MPC: {e}")

    print()
    print("=" * 80)
    print("TESTES CONCLUÍDOS")
    print("=" * 80)


if __name__ == "__main__":
    test_forecasters()
