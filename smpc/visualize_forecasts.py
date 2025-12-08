#!/usr/bin/env python3
"""
Visualização de Previsões
Mostra as previsões de load, solar e R exatamente como são usadas na simulação.
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.forecasters.profile_persistence_forecaster import ProfilePersistenceForecaster
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import SimpleTariff, BiHorariaTariff


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_historical_data(solar, house, end_time, days=30):
    """Load historical data for training forecasters."""
    start_time = end_time - timedelta(days=days)
    print(f"Carregando dados históricos: {start_time} até {end_time}")

    # Generate timestamps
    dt = timedelta(minutes=15)
    timestamps = []
    current = start_time
    while current <= end_time:
        timestamps.append(current)
        current += dt

    # Get solar and load data
    solar_data = []
    load_data = []
    for ts in timestamps:
        solar_data.append(solar.get_production(ts))
        load_data.append(house.get_consumption(ts))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'P_pv': solar_data,
        'P_load': load_data
    })

    return df


def get_actual_data(solar, house, start_time, n_steps, dt_minutes):
    """Get actual solar and load data for comparison."""
    timestamps = []
    solar_actual = []
    load_actual = []

    dt = timedelta(minutes=dt_minutes)
    current = start_time

    for _ in range(n_steps):
        timestamps.append(current)
        solar_actual.append(solar.get_production(current))
        load_actual.append(house.get_consumption(current))
        current += dt

    return timestamps, np.array(solar_actual), np.array(load_actual)


def main():
    """Main function."""

    print("=" * 80)
    print("VISUALIZAÇÃO DE PREVISÕES - Exatamente como usado na simulação")
    print("=" * 80)

    # Load configuration
    config = load_config()
    sim_config = config['simulation']
    dt_minutes = sim_config['timestep_minutes']

    # Parse dates
    sim_start = datetime.strptime(sim_config['start_date'], '%Y-%m-%d %H:%M:%S')
    sim_end = datetime.strptime(sim_config['end_date'], '%Y-%m-%d %H:%M:%S')

    print(f"\nPeríodo de simulação: {sim_start} até {sim_end}")

    # Initialize components
    print("\n1. Inicializando componentes...")
    solar_config = config['solar']
    scale_factor_solar = solar_config.get('solar_capacity_kwp', 1.0) * solar_config.get('scale_factor_for_1kwp', 1.0)
    solar = SolarPanel(
        capacity_kw=solar_config.get('solar_capacity_kwp', 1.0),
        data_file=solar_config['data_file'],
        scale_factor=scale_factor_solar
    )

    house_config = config['house']
    scale_factor_house = house_config.get('scale_factor', 1.0)
    house = House(
        data_file=house_config['data_file'],
        scale_factor=scale_factor_house
    )

    # Initialize tariff
    tariff_config = config['tariff']
    if tariff_config['type'] == 'simple':
        tariff = SimpleTariff(tariff_config['simple']['price'])
    else:
        tariff = BiHorariaTariff(
            tariff_config['bihoraria']['peak_price'],
            tariff_config['bihoraria']['off_peak_price'],
            tariff_config['bihoraria']['peak_hours_weekday'],
            tariff_config['bihoraria']['peak_hours_weekend']
        )

    # Load historical data
    print("\n2. Carregando dados históricos...")
    historical_data = load_historical_data(solar, house, sim_start, days=30)
    print(f"   {len(historical_data)} pontos de dados carregados")

    # Initialize forecasters (exactly as in simulation)
    print("\n3. Inicializando forecasters...")
    if config['controller']['type'] == 'sdp':
        n_weeks = config['controller']['sdp']['forecaster'].get('n_weeks', 3)
        horizon_steps = config['controller']['sdp']['horizon_steps']
        policy_update_hours = config['controller']['sdp'].get('policy_update_hours', 12)
    elif config['controller']['type'] == 'mpc':
        n_weeks = 3
        horizon_steps = config['controller']['mpc']['horizon_steps']
        policy_update_hours = 4  # MPC atualiza mais frequentemente
    else:
        n_weeks = 3
        horizon_steps = 96
        policy_update_hours = 4

    pv_forecaster = ProfilePersistenceForecaster(n_weeks=n_weeks, dt_minutes=dt_minutes)
    pv_forecaster.set_data(historical_data[['timestamp', 'P_pv']], value_column='P_pv')

    load_forecaster = ProfilePersistenceForecaster(n_weeks=n_weeks, dt_minutes=dt_minutes)
    load_forecaster.set_data(historical_data[['timestamp', 'P_load']], value_column='P_load')

    print(f"   Método: Profile Persistence ({n_weeks} semanas)")
    print(f"   Horizonte: {horizon_steps} passos = {horizon_steps * dt_minutes / 60:.1f} horas")
    print(f"   Atualização de policy: a cada {policy_update_hours} horas")

    # Generate policy update times (exactly as in SDP controller)
    print("\n4. Determinando pontos de atualização de policy...")
    update_times = []
    current = sim_start
    while current <= sim_end:
        update_times.append(current)
        current += timedelta(hours=policy_update_hours)

    print(f"   {len(update_times)} atualizações de policy durante a simulação")
    for t in update_times:
        print(f"   - {t.strftime('%Y-%m-%d %H:%M')}")

    # Choose a representative update time for detailed analysis
    if len(update_times) > 0:
        analysis_time = update_times[0]
    else:
        analysis_time = sim_start

    print(f"\n5. Análise detalhada no ponto: {analysis_time.strftime('%Y-%m-%d %H:%M')}")

    # Get forecasts (N+1 steps, as in SDP)
    N = horizon_steps
    P_pv_bar = pv_forecaster.get_forecast(analysis_time, N + 1)
    P_load_bar = load_forecaster.get_forecast(analysis_time, N + 1)
    R_bar = P_pv_bar - P_load_bar

    # Get actual data
    timestamps, solar_actual, load_actual = get_actual_data(
        solar, house, analysis_time, N + 1, dt_minutes
    )
    residual_actual = solar_actual - load_actual

    # Get prices
    price_forecast = tariff.get_prices_for_horizon(analysis_time, N + 1, dt_minutes)

    # Calculate errors
    mae_solar = np.mean(np.abs(solar_actual - P_pv_bar))
    mae_load = np.mean(np.abs(load_actual - P_load_bar))
    mae_residual = np.mean(np.abs(residual_actual - R_bar))

    rmse_solar = np.sqrt(np.mean((solar_actual - P_pv_bar)**2))
    rmse_load = np.sqrt(np.mean((load_actual - P_load_bar)**2))
    rmse_residual = np.sqrt(np.mean((residual_actual - R_bar)**2))

    print(f"\n   Erros de Previsão:")
    print(f"   Solar    - MAE: {mae_solar:.3f} kW  |  RMSE: {rmse_solar:.3f} kW")
    print(f"   Load     - MAE: {mae_load:.3f} kW  |  RMSE: {rmse_load:.3f} kW")
    print(f"   Residual - MAE: {mae_residual:.3f} kW  |  RMSE: {rmse_residual:.3f} kW")

    # Create time array for plotting (hours from start)
    time_hours = np.arange(N + 1) * (dt_minutes / 60)

    # ========== PLOT 1: Detailed analysis at single update point ==========
    print("\n6. Gerando gráficos...")
    fig1, axes = plt.subplots(4, 1, figsize=(15, 13))
    fig1.suptitle(f'Previsões no ponto de atualização: {analysis_time.strftime("%Y-%m-%d %H:%M")}\n' +
                  f'Horizonte: {N+1} passos ({(N+1)*dt_minutes/60:.1f}h), Método: Profile Persistence ({n_weeks} semanas)',
                  fontsize=13, fontweight='bold')

    # Plot 1: Solar Production
    ax = axes[0]
    ax.plot(time_hours, solar_actual, 'o-', label='Real', color='#FF8C00', markersize=3, linewidth=1.5)
    ax.plot(time_hours, P_pv_bar, 's--', label='Previsão', color='#DC143C', markersize=2, linewidth=1.2, alpha=0.7)
    ax.fill_between(time_hours, solar_actual, P_pv_bar, alpha=0.2, color='gray')
    ax.set_ylabel('Solar P_pv (kW)', fontsize=10, fontweight='bold')
    ax.set_title(f'Produção Solar  |  MAE: {mae_solar:.3f} kW, RMSE: {rmse_solar:.3f} kW', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Load Consumption
    ax = axes[1]
    ax.plot(time_hours, load_actual, 'o-', label='Real', color='#1E90FF', markersize=3, linewidth=1.5)
    ax.plot(time_hours, P_load_bar, 's--', label='Previsão', color='#DC143C', markersize=2, linewidth=1.2, alpha=0.7)
    ax.fill_between(time_hours, load_actual, P_load_bar, alpha=0.2, color='gray')
    ax.set_ylabel('Load P_load (kW)', fontsize=10, fontweight='bold')
    ax.set_title(f'Consumo da Casa  |  MAE: {mae_load:.3f} kW, RMSE: {rmse_load:.3f} kW', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Residual (Net Production)
    ax = axes[2]
    ax.plot(time_hours, residual_actual, 'o-', label='R real', color='#228B22', markersize=3, linewidth=1.5)
    ax.plot(time_hours, R_bar, 's--', label='R̄ previsão', color='#DC143C', markersize=2, linewidth=1.2, alpha=0.7)
    ax.fill_between(time_hours, residual_actual, R_bar, alpha=0.2, color='gray')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Residual R (kW)', fontsize=10, fontweight='bold')
    ax.set_title(f'Residual R = P_pv - P_load  |  MAE: {mae_residual:.3f} kW, RMSE: {rmse_residual:.3f} kW', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Electricity Price
    ax = axes[3]
    ax.plot(time_hours, price_forecast, 'o-', label='Preço', color='#8B008B', markersize=3, linewidth=1.5)
    ax.set_ylabel('Preço (€/kWh)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Tempo (horas desde início da previsão)', fontsize=10, fontweight='bold')
    ax.set_title(f'Preço da Eletricidade', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path(config['output']['plots_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath1 = output_dir / f"forecast_detailed_{analysis_time.strftime('%Y%m%d_%H%M')}.png"
    fig1.savefig(filepath1, dpi=150, bbox_inches='tight')
    print(f"   Salvo: {filepath1}")

    # ========== PLOT 2: Rolling forecast windows ==========
    print("\n7. Gerando janelas de previsão rolantes...")

    fig2, axes = plt.subplots(3, 1, figsize=(16, 11))
    fig2.suptitle(f'Janelas de Previsão Rolantes\n' +
                  f'Atualizações a cada {policy_update_hours}h durante {sim_start.strftime("%Y-%m-%d")}',
                  fontsize=13, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(update_times)))

    for idx, update_time in enumerate(update_times):
        # Get forecasts at this update point
        pv_fc = pv_forecaster.get_forecast(update_time, N + 1)
        load_fc = load_forecaster.get_forecast(update_time, N + 1)
        r_fc = pv_fc - load_fc

        # Create timestamps for this window
        window_times = [update_time + timedelta(minutes=dt_minutes * i) for i in range(N + 1)]

        # Plot solar
        axes[0].plot(window_times, pv_fc, '-', color=colors[idx],
                     linewidth=1.8, alpha=0.7, label=f'{update_time.strftime("%H:%M")}')

        # Plot load
        axes[1].plot(window_times, load_fc, '-', color=colors[idx],
                     linewidth=1.8, alpha=0.7, label=f'{update_time.strftime("%H:%M")}')

        # Plot residual
        axes[2].plot(window_times, r_fc, '-', color=colors[idx],
                     linewidth=1.8, alpha=0.7, label=f'{update_time.strftime("%H:%M")}')

    # Format axes
    axes[0].set_ylabel('Solar P_pv (kW)', fontsize=10, fontweight='bold')
    axes[0].set_title(f'Produção Solar Prevista (Horizonte: {(N+1)*dt_minutes/60:.1f}h)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=8, title='Início')

    axes[1].set_ylabel('Load P_load (kW)', fontsize=10, fontweight='bold')
    axes[1].set_title(f'Consumo Previsto (Horizonte: {(N+1)*dt_minutes/60:.1f}h)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=8, title='Início')

    axes[2].set_ylabel('Residual R (kW)', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Tempo', fontsize=10, fontweight='bold')
    axes[2].set_title(f'Residual R̄ = P_pv - P_load (Horizonte: {(N+1)*dt_minutes/60:.1f}h)', fontsize=11)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=8, title='Início')

    # Format x-axis
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        fig2.autofmt_xdate()

    plt.tight_layout()

    # Save
    filepath2 = output_dir / f"forecast_rolling_{sim_start.strftime('%Y%m%d')}.png"
    fig2.savefig(filepath2, dpi=150, bbox_inches='tight')
    print(f"   Salvo: {filepath2}")

    print("\n" + "=" * 80)
    print("VISUALIZAÇÃO COMPLETA")
    print("=" * 80)
    print(f"\nGráficos salvos em: {output_dir}")
    print(f"  - {filepath1.name}")
    print(f"  - {filepath2.name}")
    print("\nPara visualizar, abra os ficheiros PNG gerados.")


if __name__ == "__main__":
    main()
