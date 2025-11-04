#!/usr/bin/env python3
"""
Teste de qualidade dos forecasters
Compara previsões vs. realidade para diagnosticar problemas
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

from src.components.solar import SolarPanel
from src.components.house import House
from src.forecasters.load_forecaster import LoadForecaster
from src.forecasters.solar_forecaster import SolarForecaster


def test_forecaster_quality():
    """Testa qualidade dos forecasters comparando com dados reais"""

    print("=" * 80)
    print("TESTE DE QUALIDADE DOS FORECASTERS")
    print("=" * 80)

    # Criar componentes
    script_dir = Path(__file__).parent
    solar = SolarPanel(
        capacity_kw=5.0,
        data_file=str(script_dir / "data/solar/pvdata.csv")
    )
    house = House(
        data_file=str(script_dir / "data/load/merged_consumos.xlsx")
    )

    # Criar forecasters
    load_forecaster = LoadForecaster(house=house)
    solar_forecaster = SolarForecaster(solar_panel=solar)

    # Período de teste (mesmo da simulação)
    test_start = datetime(2025, 2, 1, 0, 0, 0)
    test_days = 4
    dt_minutes = 15
    horizon_steps = 96  # 24 horas de previsão

    # Arrays para armazenar resultados
    results = []

    print(f"\nTestando previsões de {test_start} por {test_days} dias")
    print(f"Horizonte de previsão: {horizon_steps} steps ({horizon_steps * dt_minutes / 60:.1f} horas)")
    print("\nMétodo usado: 'historical' (lookback de 7 dias)")
    print("-" * 80)

    # Testar previsões a cada 6 horas
    current_time = test_start
    test_intervals = test_days * 4  # 4 intervalos por dia (cada 6h)

    for interval_idx in range(test_intervals):
        # Fazer previsão
        load_forecast = load_forecaster.get_forecast(
            start_time=current_time,
            n_steps=horizon_steps,
            dt_minutes=dt_minutes,
            method='historical',
            context_hours=24
        )

        solar_forecast = solar_forecaster.get_forecast(
            start_time=current_time,
            n_steps=horizon_steps,
            dt_minutes=dt_minutes,
            method='historical',
            context_hours=24,
            apply_night_constraint=True
        )

        # Obter valores reais
        load_real = np.zeros(horizon_steps)
        solar_real = np.zeros(horizon_steps)
        test_time = current_time

        for i in range(horizon_steps):
            load_real[i] = house.get_consumption(test_time)
            solar_real[i] = solar.get_production(test_time)
            test_time += timedelta(minutes=dt_minutes)

        # Calcular erros
        load_mae = np.mean(np.abs(load_forecast - load_real))
        load_rmse = np.sqrt(np.mean((load_forecast - load_real) ** 2))
        load_mape = np.mean(np.abs((load_forecast - load_real) / (load_real + 0.01))) * 100

        solar_mae = np.mean(np.abs(solar_forecast - solar_real))
        solar_rmse = np.sqrt(np.mean((solar_forecast - solar_real) ** 2))
        # MAPE só para valores solares > 0.1 kW (evitar divisão por zero à noite)
        solar_mask = solar_real > 0.1
        if np.sum(solar_mask) > 0:
            solar_mape = np.mean(np.abs((solar_forecast[solar_mask] - solar_real[solar_mask]) / solar_real[solar_mask])) * 100
        else:
            solar_mape = 0.0

        results.append({
            'timestamp': current_time,
            'load_mae': load_mae,
            'load_rmse': load_rmse,
            'load_mape': load_mape,
            'solar_mae': solar_mae,
            'solar_rmse': solar_rmse,
            'solar_mape': solar_mape,
            'load_forecast_mean': np.mean(load_forecast),
            'load_real_mean': np.mean(load_real),
            'solar_forecast_mean': np.mean(solar_forecast),
            'solar_real_mean': np.mean(solar_real)
        })

        # Mostrar progresso
        if interval_idx < 3 or interval_idx == test_intervals - 1:
            print(f"\n📍 Previsão em {current_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   LOAD:  MAE={load_mae:.3f} kW, RMSE={load_rmse:.3f} kW, MAPE={load_mape:.1f}%")
            print(f"          Média prevista={np.mean(load_forecast):.3f} kW, real={np.mean(load_real):.3f} kW")
            print(f"   SOLAR: MAE={solar_mae:.3f} kW, RMSE={solar_rmse:.3f} kW, MAPE={solar_mape:.1f}%")
            print(f"          Média prevista={np.mean(solar_forecast):.3f} kW, real={np.mean(solar_real):.3f} kW")

        # Avançar 6 horas
        current_time += timedelta(hours=6)

    # Resumo
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RESUMO DA QUALIDADE DOS FORECASTERS")
    print("=" * 80)

    print("\n📊 LOAD FORECASTER:")
    print(f"   MAE médio:  {df_results['load_mae'].mean():.3f} kW (± {df_results['load_mae'].std():.3f})")
    print(f"   RMSE médio: {df_results['load_rmse'].mean():.3f} kW")
    print(f"   MAPE médio: {df_results['load_mape'].mean():.1f}% (± {df_results['load_mape'].std():.1f}%)")

    # Diagnóstico
    if df_results['load_mae'].mean() > 0.5:
        print(f"   ⚠️  PROBLEMA: MAE muito alto (>{0.5} kW)")
        print(f"   → Previsões de consumo têm erro de {df_results['load_mae'].mean():.3f} kW em média")
        print(f"   → Com consumo médio de ~0.9 kW, isso é {df_results['load_mae'].mean()/0.9*100:.0f}% de erro!")

    print("\n📊 SOLAR FORECASTER:")
    print(f"   MAE médio:  {df_results['solar_mae'].mean():.3f} kW (± {df_results['solar_mae'].std():.3f})")
    print(f"   RMSE médio: {df_results['solar_rmse'].mean():.3f} kW")
    print(f"   MAPE médio: {df_results['solar_mape'].mean():.1f}% (± {df_results['solar_mape'].std():.1f}%)")

    # Diagnóstico
    if df_results['solar_mae'].mean() > 0.5:
        print(f"   ⚠️  PROBLEMA: MAE muito alto (>{0.5} kW)")
        print(f"   → Previsões de solar têm erro de {df_results['solar_mae'].mean():.3f} kW em média")
        print(f"   → Com produção média de ~1.8 kW (dia), isso é {df_results['solar_mae'].mean()/1.8*100:.0f}% de erro!")

    print("\n" + "=" * 80)
    print("IMPACTO NO MPC")
    print("=" * 80)

    total_error = df_results['load_mae'].mean() + df_results['solar_mae'].mean()
    print(f"\n• Erro total combinado: {total_error:.3f} kW")
    print(f"• Net demand error: ~{total_error:.3f} kW (load - solar)")

    if total_error > 1.0:
        print(f"\n❌ ERRO CRÍTICO!")
        print(f"   Com erro de {total_error:.3f} kW, o MPC não consegue otimizar corretamente!")
        print(f"   Exemplo: MPC prevê excesso solar de 2 kW mas na realidade há déficit de 1 kW")
        print(f"   Resultado: Bateria carrega quando deveria descarregar → importa da rede!")
    elif total_error > 0.5:
        print(f"\n⚠️  ERRO MODERADO")
        print(f"   Com erro de {total_error:.3f} kW, o MPC pode tomar decisões sub-ótimas")
    else:
        print(f"\n✅ ERRO ACEITÁVEL")
        print(f"   Com erro de {total_error:.3f} kW, o MPC deve funcionar razoavelmente bem")

    # Salvar resultados
    df_results.to_csv('results/data/forecast_quality.csv', index=False)
    print(f"\n✅ Resultados salvos em: results/data/forecast_quality.csv")

    return df_results


def test_specific_case():
    """Testa um caso específico problemático"""

    print("\n" + "=" * 80)
    print("TESTE DE CASO ESPECÍFICO PROBLEMÁTICO")
    print("=" * 80)

    # Criar componentes
    script_dir = Path(__file__).parent
    solar = SolarPanel(
        capacity_kw=5.0,
        data_file=str(script_dir / "data/solar/pvdata.csv")
    )
    house = House(
        data_file=str(script_dir / "data/load/merged_consumos.xlsx")
    )

    # Criar forecasters
    load_forecaster = LoadForecaster(house=house)
    solar_forecaster = SolarForecaster(solar_panel=solar)

    # Caso problemático: 2025-02-01 17:00 (início da noite, peak price)
    problem_time = datetime(2025, 2, 1, 17, 0, 0)
    horizon = 24  # próximas 6 horas

    print(f"\n🔍 Analisando: {problem_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Contexto: Início da noite, preço peak (0.18 €/kWh)")
    print(f"   Horizonte: {horizon} steps (6 horas)")

    # Fazer previsão
    load_forecast = load_forecaster.get_forecast(
        start_time=problem_time,
        n_steps=horizon,
        dt_minutes=15,
        method='historical'
    )

    solar_forecast = solar_forecaster.get_forecast(
        start_time=problem_time,
        n_steps=horizon,
        dt_minutes=15,
        method='historical'
    )

    # Obter valores reais
    load_real = np.zeros(horizon)
    solar_real = np.zeros(horizon)
    test_time = problem_time

    for i in range(horizon):
        load_real[i] = house.get_consumption(test_time)
        solar_real[i] = solar.get_production(test_time)
        test_time += timedelta(minutes=15)

    # Mostrar comparação
    print("\n📊 COMPARAÇÃO (primeiros 12 steps = 3 horas):")
    print("\nTimestamp           | Load Prev | Load Real | Erro  | Solar Prev | Solar Real | Erro")
    print("-" * 95)

    test_time = problem_time
    for i in range(min(12, horizon)):
        load_err = load_forecast[i] - load_real[i]
        solar_err = solar_forecast[i] - solar_real[i]
        print(f"{test_time.strftime('%H:%M')} | {load_forecast[i]:9.3f} | {load_real[i]:9.3f} | "
              f"{load_err:+6.3f} | {solar_forecast[i]:10.3f} | {solar_real[i]:10.3f} | {solar_err:+6.3f}")
        test_time += timedelta(minutes=15)

    # Net demand (load - solar)
    net_demand_forecast = load_forecast - solar_forecast
    net_demand_real = load_real - solar_real
    net_demand_error = net_demand_forecast - net_demand_real

    print("\n📈 NET DEMAND (load - solar):")
    print(f"   Média prevista: {np.mean(net_demand_forecast):.3f} kW")
    print(f"   Média real:     {np.mean(net_demand_real):.3f} kW")
    print(f"   Erro médio:     {np.mean(net_demand_error):.3f} kW")
    print(f"   MAE:            {np.mean(np.abs(net_demand_error)):.3f} kW")

    if np.mean(net_demand_error) > 0.3:
        print(f"\n❌ PROBLEMA: MPC prevê net demand {np.mean(net_demand_error):.3f} kW MAIOR que o real!")
        print(f"   → MPC pensa que vai precisar importar mais energia da rede")
        print(f"   → Não usa bateria porque 'acha' que vai precisar dela mais tarde")
        print(f"   → Resultado: Importa da rede desnecessariamente!")
    elif np.mean(net_demand_error) < -0.3:
        print(f"\n⚠️  MPC prevê net demand {-np.mean(net_demand_error):.3f} kW MENOR que o real!")
        print(f"   → MPC pensa que vai ter mais solar/menos consumo")
        print(f"   → Pode não carregar bateria o suficiente")


if __name__ == "__main__":
    # Teste geral de qualidade
    df_results = test_forecaster_quality()

    # Teste de caso específico
    test_specific_case()

    print("\n" + "=" * 80)
    print("TESTES COMPLETOS")
    print("=" * 80)
