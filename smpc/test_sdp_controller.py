"""
Script de teste para o controlador SDP.

Este script demonstra:
1. Carregamento de dados históricos
2. Criação de forecasters de persistência por perfil
3. Calibração de parâmetros (t_1/2 e σ)
4. Execução do controlador SDP
5. Comparação de resultados
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

from src.components.battery import Battery
from src.components.house import House
from src.components.solar import SolarPanel
from src.components.tariff import BiHorariaTariff
from src.forecasters.profile_persistence_forecaster import ProfilePersistenceForecaster
from src.controllers.sdp_controller import (
    SDPController, SDPParams, PlantModel, ResidualModel
)
from src.controllers.sdp_calibration import quick_calibrate


def load_data():
    """Carregar dados históricos de PV e carga."""
    print("\n=== Carregando Dados ===")

    # Dados de carga
    load_file = "data/load/merged_consumos.xlsx"
    house = House(load_file)

    # Dados de PV
    pv_file = "data/solar/pvdata.csv"
    solar = SolarPanel(capacity_kw=5.0, data_file=pv_file)

    # Tarifa bi-horária (valores típicos Portugal)
    tariff = BiHorariaTariff(
        peak_price=0.20,      # €/kWh em horas de ponta
        off_peak_price=0.12   # €/kWh em horas de vazio
    )

    print("Dados carregados com sucesso!\n")

    return house, solar, tariff


def prepare_forecasters(house, solar):
    """Preparar forecasters de persistência por perfil."""
    print("=== Preparando Forecasters ===")

    # Preparar dados de carga
    if house.consumption_data is not None:
        # Converter para formato adequado
        load_data = house.consumption_data.copy()

        # Criar timestamps
        load_data['timestamp'] = pd.to_datetime(
            load_data.apply(
                lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                           f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
                axis=1
            )
        )
        load_data = load_data[['timestamp', 'consumption']].rename(
            columns={'consumption': 'P_load'}
        )
    else:
        load_data = None

    # Preparar dados de PV
    if solar.production_data is not None:
        pv_data = solar.production_data.copy()

        # Criar timestamps se não existir
        if 'timestamp' not in pv_data.columns:
            pv_data['timestamp'] = pd.to_datetime(
                pv_data.apply(
                    lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                               f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
                    axis=1
                )
            )

        # Renomear coluna de produção e converter para kW
        prod_col = 'pv_1' if 'pv_1' in pv_data.columns else 'production'
        pv_data = pv_data[['timestamp', prod_col]].copy()
        pv_data['P_pv'] = pv_data[prod_col] / 1000.0  # W para kW
        pv_data = pv_data[['timestamp', 'P_pv']]
    else:
        pv_data = None

    # Criar forecasters
    load_forecaster = ProfilePersistenceForecaster(n_weeks=3, dt_minutes=15)
    pv_forecaster = ProfilePersistenceForecaster(n_weeks=3, dt_minutes=15)

    if load_data is not None:
        load_forecaster.set_data(load_data, value_column='P_load')
        print(f"  Load forecaster configurado com {len(load_data)} pontos")

    if pv_data is not None:
        pv_forecaster.set_data(pv_data, value_column='P_pv')
        print(f"  PV forecaster configurado com {len(pv_data)} pontos")

    print("Forecasters prontos!\n")

    return load_forecaster, pv_forecaster


def calibrate_sdp(house, solar, load_forecaster, pv_forecaster):
    """Calibrar parâmetros do SDP."""
    print("=== Calibrando Parâmetros SDP ===")

    # Preparar dados históricos combinados
    load_data = house.consumption_data.copy()
    load_data['timestamp'] = pd.to_datetime(
        load_data.apply(
            lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                       f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
            axis=1
        )
    )
    load_data = load_data[['timestamp', 'consumption']].rename(
        columns={'consumption': 'P_load'}
    )

    # Preparar dados de PV
    pv_data = solar.production_data.copy()

    # Criar timestamps se não existir
    if 'timestamp' not in pv_data.columns:
        pv_data['timestamp'] = pd.to_datetime(
            pv_data.apply(
                lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                           f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
                axis=1
            )
        )

    # Renomear coluna de produção e converter para kW
    prod_col = 'pv_1' if 'pv_1' in pv_data.columns else 'production'
    pv_data = pv_data[['timestamp', prod_col]].copy()
    pv_data['P_pv'] = pv_data[prod_col] / 1000.0  # W para kW
    pv_data = pv_data[['timestamp', 'P_pv']]

    # Merge dos dados
    historical_data = pd.merge(
        pv_data, load_data, on='timestamp', how='inner'
    )

    print(f"Dados históricos: {len(historical_data)} pontos")

    # Escolher janela de calibração (primeiro dia completo)
    start_time = historical_data['timestamp'].min()
    start_time = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0)

    print(f"Calibração iniciando em: {start_time}")

    # Calibração rápida
    results = quick_calibrate(
        historical_data=historical_data,
        start_time=start_time,
        pv_forecaster=pv_forecaster,
        load_forecaster=load_forecaster,
        dt_minutes=15
    )

    return results


def create_sdp_controller(calib_results, battery, tariff):
    """Criar controlador SDP com parâmetros calibrados."""
    print("=== Criando Controlador SDP ===")

    # Parâmetros do SDP
    params = SDPParams(
        N=36,  # 9h horizonte
        dt_minutes=15,
        n_x=100,  # Reduzido para 100 para velocidade
        n_R=30,   # Reduzido para 30 para velocidade
        t_half_minutes=calib_results['t_half_minutes'],
        sigma_R=calib_results['sigma_kw'],
        policy_update_hours=6,
        soc_target=0.5,
        terminal_cost_weight=1.0
    )

    print(f"  Horizonte: {params.N} steps ({params.N * params.dt_minutes / 60:.1f}h)")
    print(f"  Discretização: {params.n_x} pontos SOC, {params.n_R} pontos Resíduo")
    print(f"  t_1/2: {params.t_half_minutes:.1f} min")
    print(f"  σ: {params.sigma_R:.3f} kW")

    # Modelo do resíduo
    residual_model = ResidualModel(
        t_half_minutes=params.t_half_minutes,
        dt_minutes=params.dt_minutes,
        sigma=params.sigma_R
    )

    # Modelo da planta
    plant = PlantModel(
        C_bat=battery.capacity_kwh,
        P_nom=battery.max_power_kw,
        P_lim=10.0,  # Limite de injeção (kW) - ajustar conforme necessário
        eta_charge=battery.efficiency_charge,
        eta_discharge=battery.efficiency_discharge,
        dt_minutes=params.dt_minutes,
        soc_min=battery.min_soc,
        soc_max=battery.max_soc
    )

    # Tarifas médias da tarifa bi-horária
    # Calcular média ponderada (assumindo ~12h ponta + 12h vazio)
    c_s = (tariff.peak_price + tariff.off_peak_price) / 2  # Média para SDP
    c_f = 0.05  # €/kWh venda (tarifa de injeção típica)

    print(f"  Tarifa média compra: {c_s:.3f} €/kWh")
    print(f"  Tarifa venda: {c_f:.3f} €/kWh")

    # Criar controlador
    controller = SDPController(
        params=params,
        plant=plant,
        residual_model=residual_model,
        c_s=c_s,
        c_f=c_f
    )

    print("Controlador SDP criado!\n")

    return controller


def simulate_one_day(controller, battery, house, solar, tariff,
                     load_forecaster, pv_forecaster, start_date):
    """Simular um dia com o controlador SDP."""
    print(f"\n=== Simulando dia {start_date.date()} ===")

    # Reset battery
    battery.reset(soc=0.5)

    # Simulação
    dt_minutes = 15
    dt_hours = dt_minutes / 60.0
    n_steps = int(24 * 60 / dt_minutes)  # 1 dia

    results = {
        'timestamp': [],
        'soc': [],
        'battery_power': [],
        'solar_power': [],
        'load_power': [],
        'grid_power': [],
        'residual': [],
        'price': []
    }

    current_time = start_date

    for step in range(n_steps):
        # Obter valores atuais
        solar_power = solar.get_production(current_time)
        load_power = house.get_consumption(current_time)
        soc = battery.get_soc()
        price = tariff.get_price(current_time)

        # Calcular ação SDP
        battery_action = controller.compute_action(
            timestamp=current_time,
            solar_power=solar_power,
            load_power=load_power,
            battery=battery,
            tariff=tariff,
            solar_panel=solar,
            house=house,
            pv_forecaster=pv_forecaster,
            load_forecaster=load_forecaster
        )

        # Aplicar ação à bateria
        battery.step(battery_action, dt_hours)

        # Calcular potência da rede
        grid_power = load_power - solar_power - battery_action

        # Salvar resultados
        results['timestamp'].append(current_time)
        results['soc'].append(soc)
        results['battery_power'].append(battery_action)
        results['solar_power'].append(solar_power)
        results['load_power'].append(load_power)
        results['grid_power'].append(grid_power)
        results['residual'].append(solar_power - load_power)
        results['price'].append(price)

        # Avançar tempo
        current_time += timedelta(minutes=dt_minutes)

        if step % 24 == 0:  # A cada 6h
            print(f"  Step {step}/{n_steps} - SOC: {soc:.2%} - "
                  f"Battery: {battery_action:.2f} kW - Grid: {grid_power:.2f} kW - "
                  f"Price: {price:.3f} €/kWh")

    return pd.DataFrame(results)


def plot_results(results):
    """Plotar resultados da simulação."""
    # Criar diretório results se não existir
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    # SOC
    axes[0].plot(results['timestamp'], results['soc'], 'b-', linewidth=2)
    axes[0].set_ylabel('SOC')
    axes[0].set_title('Estado da Bateria')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Potências
    axes[1].plot(results['timestamp'], results['solar_power'], 'y-',
                 label='Solar', linewidth=2)
    axes[1].plot(results['timestamp'], results['load_power'], 'r-',
                 label='Carga', linewidth=2)
    axes[1].set_ylabel('Potência (kW)')
    axes[1].set_title('Solar e Carga')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Bateria
    axes[2].plot(results['timestamp'], results['battery_power'], 'g-', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Bateria (kW)')
    axes[2].set_title('Ação da Bateria (+ = carga, - = descarga)')
    axes[2].grid(True, alpha=0.3)

    # Rede
    axes[3].plot(results['timestamp'], results['grid_power'], 'm-', linewidth=2)
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_ylabel('Rede (kW)')
    axes[3].set_title('Potência da Rede (+ = compra, - = venda)')
    axes[3].grid(True, alpha=0.3)

    # Preços
    axes[4].plot(results['timestamp'], results['price'], 'c-', linewidth=2)
    axes[4].set_ylabel('Preço (€/kWh)')
    axes[4].set_xlabel('Tempo')
    axes[4].set_title('Tarifa de Eletricidade')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/sdp_simulation.png', dpi=150)
    print("\nGráfico salvo em: results/sdp_simulation.png")
    plt.show()


def main():
    """Função principal."""
    print("\n" + "="*60)
    print("TESTE DO CONTROLADOR SDP")
    print("="*60)

    # 1. Carregar dados
    house, solar, tariff = load_data()

    # 2. Preparar forecasters
    load_forecaster, pv_forecaster = prepare_forecasters(house, solar)

    # 3. Calibrar SDP
    calib_results = calibrate_sdp(house, solar, load_forecaster, pv_forecaster)

    # 4. Criar bateria
    battery = Battery(
        capacity_kwh=10.0,
        max_power_kw=5.0,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        initial_soc=0.5,
        min_soc=0.1,
        max_soc=0.9
    )

    # 5. Criar controlador SDP
    controller = create_sdp_controller(calib_results, battery, tariff)

    # 6. Simular um dia
    # Escolher data de simulação (usar dados disponíveis)
    # Usar dados de carga como referência (mais confiável)
    start_date = datetime(2025, 4, 8, 0, 0, 0)  # 8 de janeiro (após período de calibração)

    results = simulate_one_day(
        controller=controller,
        battery=battery,
        house=house,
        solar=solar,
        tariff=tariff,
        load_forecaster=load_forecaster,
        pv_forecaster=pv_forecaster,
        start_date=start_date
    )

    # 7. Plotar resultados
    plot_results(results)

    # 8. Estatísticas
    print("\n=== Estatísticas ===")
    print(f"SOC médio: {results['soc'].mean():.2%}")
    print(f"SOC final: {results['soc'].iloc[-1]:.2%}")

    energy_charge = results[results['battery_power'] > 0]['battery_power'].sum() * 0.25
    energy_discharge = abs(results[results['battery_power'] < 0]['battery_power'].sum()) * 0.25
    energy_import = results[results['grid_power'] > 0]['grid_power'].sum() * 0.25
    energy_export = abs(results[results['grid_power'] < 0]['grid_power'].sum()) * 0.25

    print(f"\nBateria:")
    print(f"  Energia carregada:   {energy_charge:.2f} kWh")
    print(f"  Energia descarregada: {energy_discharge:.2f} kWh")
    print(f"  Eficiência ciclo:     {(energy_discharge/energy_charge*100) if energy_charge > 0 else 0:.1f}%")

    print(f"\nRede:")
    print(f"  Energia importada:   {energy_import:.2f} kWh")
    print(f"  Energia exportada:   {energy_export:.2f} kWh")

    # Calcular custos com tarifas reais
    cost_import = 0.0
    revenue_export = 0.0

    for idx, row in results.iterrows():
        if row['grid_power'] > 0:  # Importação
            cost_import += row['grid_power'] * row['price'] * 0.25
        elif row['grid_power'] < 0:  # Exportação
            revenue_export += abs(row['grid_power']) * 0.05 * 0.25  # 0.05 €/kWh tarifa injeção

    net_cost = cost_import - revenue_export

    print(f"\nCustos (com tarifa bi-horária):")
    print(f"  Custo importação:    {cost_import:.2f} €")
    print(f"  Receita exportação:  {revenue_export:.2f} €")
    print(f"  Custo líquido:       {net_cost:.2f} €")

    print("\n" + "="*60)
    print("TESTE CONCLUÍDO")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
