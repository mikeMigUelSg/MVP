"""
Exemplo simples de uso do controlador SDP.

Este script demonstra uso básico sem precisar de dados históricos completos.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.controllers import (
    SDPController, SDPParams, PlantModel, ResidualModel
)


def create_synthetic_forecasters():
    """
    Criar forecasters sintéticos para demonstração.

    Retorna objetos que simulam ProfilePersistenceForecaster
    mas com dados sintéticos simples.
    """

    class SimpleForecast:
        def get_forecast(self, start_time, n_steps):
            """Gerar previsão sintética."""
            forecast = np.zeros(n_steps)
            current = start_time

            for i in range(n_steps):
                hour = current.hour + current.minute / 60.0

                # Perfil dependente do tipo
                if self.type == 'pv':
                    # Produção solar: pico ao meio-dia
                    if 6 <= hour <= 18:
                        forecast[i] = 5.0 * np.sin((hour - 6) * np.pi / 12)
                    else:
                        forecast[i] = 0.0

                elif self.type == 'load':
                    # Carga: maior durante o dia e noite
                    if 7 <= hour <= 22:
                        forecast[i] = 2.0 + 1.0 * np.sin((hour - 7) * np.pi / 15)
                    else:
                        forecast[i] = 1.0

                current += timedelta(minutes=15)

            return forecast

        def __init__(self, forecast_type):
            self.type = forecast_type

    return SimpleForecast('pv'), SimpleForecast('load')


def run_simple_example():
    """Executar exemplo simples do SDP."""

    print("\n" + "="*60)
    print("EXEMPLO SIMPLES - CONTROLADOR SDP")
    print("="*60 + "\n")

    # 1. Parâmetros do SDP (valores típicos)
    print("1. Configurando parâmetros SDP...")
    params = SDPParams(
        N=36,                    # 9h horizonte
        dt_minutes=15,           # 15 min resolução
        n_x=100,                 # 100 pontos SOC (reduzido para velocidade)
        n_R=30,                  # 30 pontos Resíduo (reduzido)
        t_half_minutes=45.0,     # 45 min meia-vida
        sigma_R=0.5,             # 0.5 kW desvio padrão
        policy_update_hours=6,   # Recalcular a cada 6h
        soc_target=0.5,
        terminal_cost_weight=1.0
    )
    print(f"   Horizonte: {params.N} steps = {params.N * params.dt_minutes / 60:.1f}h")
    print(f"   Discretização: {params.n_x} SOC, {params.n_R} Resíduo")
    print(f"   Meia-vida: {params.t_half_minutes} min, σ: {params.sigma_R} kW\n")

    # 2. Modelo da planta
    print("2. Criando modelo da planta...")
    plant = PlantModel(
        C_bat=10.0,              # 10 kWh bateria
        P_nom=5.0,               # 5 kW potência nominal
        P_lim=10.0,              # 10 kW limite injeção
        eta_charge=0.95,
        eta_discharge=0.95,
        dt_minutes=15,
        soc_min=0.1,
        soc_max=0.9
    )
    print(f"   Bateria: {plant.C_bat} kWh, {plant.P_nom} kW")
    print(f"   Limite injeção: {plant.P_lim} kW\n")

    # 3. Modelo do resíduo
    print("3. Criando modelo de resíduo...")
    residual_model = ResidualModel(
        t_half_minutes=params.t_half_minutes,
        dt_minutes=params.dt_minutes,
        sigma=params.sigma_R
    )
    print(f"   κ (coef. persistência): {residual_model.kappa:.4f}")
    print(f"   Variância estacionária: {residual_model.stationary_variance():.4f} kW²\n")

    # 4. Criar controlador SDP
    print("4. Criando controlador SDP...")
    controller = SDPController(
        params=params,
        plant=plant,
        residual_model=residual_model,
        c_s=0.20,  # 0.20 €/kWh compra
        c_f=0.05   # 0.05 €/kWh venda
    )
    print("   Controlador SDP criado!\n")

    # 5. Criar forecasters sintéticos
    print("5. Criando forecasters sintéticos...")
    pv_forecaster, load_forecaster = create_synthetic_forecasters()
    print("   Forecasters prontos!\n")

    # 6. Simular alguns passos
    print("6. Simulando 24h (96 passos)...\n")

    # Estado inicial
    from src.components.battery import Battery
    battery = Battery(
        capacity_kwh=10.0,
        max_power_kw=5.0,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        initial_soc=0.5
    )

    # Objetos dummy para house e solar
    class DummyHouse:
        def get_consumption(self, t):
            hour = t.hour + t.minute / 60.0
            if 7 <= hour <= 22:
                return 2.0 + 1.0 * np.sin((hour - 7) * np.pi / 15)
            return 1.0

    class DummySolar:
        def get_production(self, t):
            hour = t.hour + t.minute / 60.0
            if 6 <= hour <= 18:
                return 5.0 * np.sin((hour - 6) * np.pi / 12)
            return 0.0

    house = DummyHouse()
    solar = DummySolar()

    # Simular
    start_time = datetime(2024, 6, 15, 0, 0, 0)  # Dia de verão
    n_steps = 96  # 24h
    dt_hours = 0.25

    results = {
        'time': [],
        'soc': [],
        'battery_action': [],
        'pv': [],
        'load': [],
        'grid': []
    }

    current_time = start_time

    for step in range(n_steps):
        # Obter estado atual
        pv_power = solar.get_production(current_time)
        load_power = house.get_consumption(current_time)
        soc = battery.get_soc()

        # Computar ação SDP
        action = controller.compute_action(
            timestamp=current_time,
            solar_power=pv_power,
            load_power=load_power,
            battery=battery,
            tariff=None,
            solar_panel=solar,
            house=house,
            pv_forecaster=pv_forecaster,
            load_forecaster=load_forecaster
        )

        # Aplicar ação
        battery.step(action, dt_hours)

        # Calcular grid
        grid_power = load_power - pv_power - action

        # Salvar
        results['time'].append(current_time.hour + current_time.minute / 60.0)
        results['soc'].append(soc)
        results['battery_action'].append(action)
        results['pv'].append(pv_power)
        results['load'].append(load_power)
        results['grid'].append(grid_power)

        # Print progress
        if step % 24 == 0:
            print(f"   {step:3d}/96 - {current_time.strftime('%H:%M')} - "
                  f"SOC: {soc:.1%} - Bat: {action:+6.2f} kW - "
                  f"Grid: {grid_power:+6.2f} kW")

        current_time += timedelta(minutes=15)

    print("\n7. Resultados:\n")
    print(f"   SOC inicial: {results['soc'][0]:.1%}")
    print(f"   SOC final:   {results['soc'][-1]:.1%}")
    print(f"   SOC médio:   {np.mean(results['soc']):.1%}")

    energy_charge = sum([a * 0.25 for a in results['battery_action'] if a > 0])
    energy_discharge = sum([-a * 0.25 for a in results['battery_action'] if a < 0])
    energy_import = sum([g * 0.25 for g in results['grid'] if g > 0])
    energy_export = sum([-g * 0.25 for g in results['grid'] if g < 0])

    print(f"\n   Energia carregada:   {energy_charge:.2f} kWh")
    print(f"   Energia descarregada: {energy_discharge:.2f} kWh")
    print(f"   Eficiência ciclo:     {energy_discharge/energy_charge*100:.1f}%")
    print(f"\n   Energia importada:   {energy_import:.2f} kWh")
    print(f"   Energia exportada:   {energy_export:.2f} kWh")

    # Custo estimado
    cost_import = energy_import * 0.20
    revenue_export = energy_export * 0.05
    net_cost = cost_import - revenue_export

    print(f"\n   Custo importação:    {cost_import:.2f} €")
    print(f"   Receita exportação:  {revenue_export:.2f} €")
    print(f"   Custo líquido:       {net_cost:.2f} €")

    print("\n" + "="*60)
    print("EXEMPLO CONCLUÍDO")
    print("="*60 + "\n")

    # Plot simples (opcional)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # SOC
        axes[0].plot(results['time'], results['soc'], 'b-', linewidth=2)
        axes[0].set_ylabel('SOC')
        axes[0].set_title('Estado da Bateria')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

        # PV e Load
        axes[1].plot(results['time'], results['pv'], 'y-', label='Solar', linewidth=2)
        axes[1].plot(results['time'], results['load'], 'r-', label='Carga', linewidth=2)
        axes[1].set_ylabel('Potência (kW)')
        axes[1].set_title('Solar e Carga')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Battery e Grid
        axes[2].plot(results['time'], results['battery_action'], 'g-',
                     label='Bateria', linewidth=2)
        axes[2].plot(results['time'], results['grid'], 'm-',
                     label='Rede', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Potência (kW)')
        axes[2].set_xlabel('Hora do Dia')
        axes[2].set_title('Ações (+ = carga/compra, - = descarga/venda)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/sdp_simple_example.png', dpi=150)
        print("Gráfico salvo em: results/sdp_simple_example.png")
        plt.show()

    except ImportError:
        print("matplotlib não disponível - pulando gráfico")


if __name__ == "__main__":
    run_simple_example()
