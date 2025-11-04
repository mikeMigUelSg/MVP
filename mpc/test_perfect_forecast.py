#!/usr/bin/env python3
"""
Teste comparativo: MPC com forecasts reais vs. MPC com previsões perfeitas
"""
import yaml
from datetime import datetime
from pathlib import Path

from src.components.battery import Battery
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import BiHorariaTariff
from src.controllers.mpc_controller import MPCController
from src.system import EnergyManagementSystem


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_test():
    """Run comparison test"""
    print("=" * 80)
    print("TESTE: MPC COM FORECASTS vs. MPC COM PREVISÕES PERFEITAS")
    print("=" * 80)

    config = load_config("config.yaml")

    # Configurações
    start_time = datetime.strptime(config['simulation']['start_date'], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(config['simulation']['end_date'], "%Y-%m-%d %H:%M:%S")
    dt_minutes = config['simulation']['timestep_minutes']

    # Criar componentes (mesmo para ambos os testes)
    script_dir = Path(__file__).parent

    battery_config = config['battery']
    solar_config = config['solar']
    house_config = config['house']
    tariff_config = config['tariff']['bihoraria']
    mpc_config = config['controller']['mpc']
    grid_config = config.get('grid', {})

    results = {}

    # ========================================================================
    # TESTE 1: MPC com forecasts normais (histórico)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TESTE 1: MPC COM FORECASTS (lookback 1 dia)")
    print("=" * 80)

    battery1 = Battery(
        capacity_kwh=battery_config['capacity_kwh'],
        max_power_kw=battery_config['max_power_kw'],
        efficiency_charge=battery_config['efficiency_charge'],
        efficiency_discharge=battery_config['efficiency_discharge'],
        initial_soc=battery_config['initial_soc'],
        min_soc=battery_config['min_soc'],
        max_soc=battery_config['max_soc'],
        degradation_cost_per_kwh=battery_config['degradation_cost_per_kwh'],
        max_charge_current_kw=battery_config.get('max_charge_current_kw'),
        max_discharge_current_kw=battery_config.get('max_discharge_current_kw')
    )

    solar1 = SolarPanel(
        capacity_kw=solar_config['capacity_kw'],
        data_file=str(script_dir / solar_config['data_file'])
    )

    house1 = House(
        data_file=str(script_dir / house_config['data_file'])
    )

    tariff1 = BiHorariaTariff(
        peak_price=tariff_config['peak_price'],
        off_peak_price=tariff_config['off_peak_price']
    )

    controller1 = MPCController(
        horizon_steps=mpc_config['horizon_steps'],
        dt_minutes=dt_minutes,
        export_price=mpc_config.get('export_price', 0.05),
        contracted_power_kw=grid_config.get('contracted_power_kw', None),
        use_perfect_forecast=False  # Usar forecasts normais
    )

    system1 = EnergyManagementSystem(battery1, solar1, house1, tariff1, controller1)
    df1 = system1.simulate(start_time, end_time, dt_minutes)
    summary1 = system1.get_summary()
    savings1 = system1.get_savings(df1)

    results['forecast'] = {
        'summary': summary1,
        'savings': savings1,
        'df': df1
    }

    print(f"\n✅ Simulação completa")
    print(f"   Total cost: {summary1['total_cost']:.2f} €")
    print(f"   Self-consumption: {summary1['self_consumption_rate']*100:.1f}%")
    print(f"   Self-sufficiency: {summary1['self_sufficiency_rate']*100:.1f}%")
    print(f"   Grid import: {summary1['total_grid_import_kwh']:.2f} kWh")
    print(f"   Grid export: {summary1['total_grid_export_kwh']:.2f} kWh")

    # ========================================================================
    # TESTE 2: MPC com previsões perfeitas
    # ========================================================================
    print("\n" + "=" * 80)
    print("TESTE 2: MPC COM PREVISÕES PERFEITAS")
    print("=" * 80)

    battery2 = Battery(
        capacity_kwh=battery_config['capacity_kwh'],
        max_power_kw=battery_config['max_power_kw'],
        efficiency_charge=battery_config['efficiency_charge'],
        efficiency_discharge=battery_config['efficiency_discharge'],
        initial_soc=battery_config['initial_soc'],
        min_soc=battery_config['min_soc'],
        max_soc=battery_config['max_soc'],
        degradation_cost_per_kwh=battery_config['degradation_cost_per_kwh'],
        max_charge_current_kw=battery_config.get('max_charge_current_kw'),
        max_discharge_current_kw=battery_config.get('max_discharge_current_kw')
    )

    solar2 = SolarPanel(
        capacity_kw=solar_config['capacity_kw'],
        data_file=str(script_dir / solar_config['data_file'])
    )

    house2 = House(
        data_file=str(script_dir / house_config['data_file'])
    )

    tariff2 = BiHorariaTariff(
        peak_price=tariff_config['peak_price'],
        off_peak_price=tariff_config['off_peak_price']
    )

    controller2 = MPCController(
        horizon_steps=mpc_config['horizon_steps'],
        dt_minutes=dt_minutes,
        export_price=mpc_config.get('export_price', 0.05),
        contracted_power_kw=grid_config.get('contracted_power_kw', None),
        use_perfect_forecast=True  # ⭐ USAR PREVISÕES PERFEITAS
    )

    system2 = EnergyManagementSystem(battery2, solar2, house2, tariff2, controller2)
    df2 = system2.simulate(start_time, end_time, dt_minutes)
    summary2 = system2.get_summary()
    savings2 = system2.get_savings(df2)

    results['perfect'] = {
        'summary': summary2,
        'savings': savings2,
        'df': df2
    }

    print(f"\n✅ Simulação completa")
    print(f"   Total cost: {summary2['total_cost']:.2f} €")
    print(f"   Self-consumption: {summary2['self_consumption_rate']*100:.1f}%")
    print(f"   Self-sufficiency: {summary2['self_sufficiency_rate']*100:.1f}%")
    print(f"   Grid import: {summary2['total_grid_import_kwh']:.2f} kWh")
    print(f"   Grid export: {summary2['total_grid_export_kwh']:.2f} kWh")

    # ========================================================================
    # COMPARAÇÃO
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARAÇÃO: FORECASTS vs. PREVISÕES PERFEITAS")
    print("=" * 80)

    print(f"\n{'Métrica':<30} {'Forecasts':<15} {'Perfeitas':<15} {'Diferença'}")
    print("-" * 80)

    # Total cost
    cost_diff = summary2['total_cost'] - summary1['total_cost']
    cost_pct = (cost_diff / abs(summary1['total_cost']) * 100) if summary1['total_cost'] != 0 else 0
    print(f"{'Total Cost (€)':<30} {summary1['total_cost']:>14.2f} {summary2['total_cost']:>14.2f} {cost_diff:>+8.2f} ({cost_pct:+.1f}%)")

    # Grid import
    import_diff = summary2['total_grid_import_kwh'] - summary1['total_grid_import_kwh']
    import_pct = (import_diff / summary1['total_grid_import_kwh'] * 100) if summary1['total_grid_import_kwh'] > 0 else 0
    print(f"{'Grid Import (kWh)':<30} {summary1['total_grid_import_kwh']:>14.2f} {summary2['total_grid_import_kwh']:>14.2f} {import_diff:>+8.2f} ({import_pct:+.1f}%)")

    # Grid export
    export_diff = summary2['total_grid_export_kwh'] - summary1['total_grid_export_kwh']
    export_pct = (export_diff / summary1['total_grid_export_kwh'] * 100) if summary1['total_grid_export_kwh'] > 0 else 0
    print(f"{'Grid Export (kWh)':<30} {summary1['total_grid_export_kwh']:>14.2f} {summary2['total_grid_export_kwh']:>14.2f} {export_diff:>+8.2f} ({export_pct:+.1f}%)")

    # Self-consumption
    sc1 = summary1['self_consumption_rate'] * 100
    sc2 = summary2['self_consumption_rate'] * 100
    sc_diff = sc2 - sc1
    print(f"{'Self-Consumption (%)':<30} {sc1:>14.1f} {sc2:>14.1f} {sc_diff:>+8.1f}pp")

    # Self-sufficiency
    ss1 = summary1['self_sufficiency_rate'] * 100
    ss2 = summary2['self_sufficiency_rate'] * 100
    ss_diff = ss2 - ss1
    print(f"{'Self-Sufficiency (%)':<30} {ss1:>14.1f} {ss2:>14.1f} {ss_diff:>+8.1f}pp")

    # Battery cycles
    cycles_diff = summary2['battery_cycles'] - summary1['battery_cycles']
    cycles_pct = (cycles_diff / summary1['battery_cycles'] * 100) if summary1['battery_cycles'] > 0 else 0
    print(f"{'Battery Cycles':<30} {summary1['battery_cycles']:>14.2f} {summary2['battery_cycles']:>14.2f} {cycles_diff:>+8.2f} ({cycles_pct:+.1f}%)")

    # Análise
    print("\n" + "=" * 80)
    print("ANÁLISE")
    print("=" * 80)

    if abs(cost_diff) < 0.10:
        print("\n✅ Forecasts estão MUITO BONS!")
        print("   Diferença de custo < 0.10 € (quase igual ao ótimo)")
    elif abs(cost_diff) < 0.50:
        print("\n✅ Forecasts estão BOM!")
        print(f"   Diferença de custo = {abs(cost_diff):.2f} € ({abs(cost_pct):.1f}%)")
        print("   Há margem para melhoria mas já está razoável")
    elif abs(cost_diff) < 1.00:
        print("\n⚠️  Forecasts podem melhorar")
        print(f"   Diferença de custo = {abs(cost_diff):.2f} € ({abs(cost_pct):.1f}%)")
        print("   Considere usar modelo mais avançado (Autoformer)")
    else:
        print("\n❌ Forecasts têm MUITO ERRO!")
        print(f"   Diferença de custo = {abs(cost_diff):.2f} € ({abs(cost_pct):.1f}%)")
        print("   MPC não consegue otimizar com forecasts tão ruins")

    # Potencial de melhoria
    potential_savings = abs(cost_diff)
    print(f"\n💰 POTENCIAL DE ECONOMIA com melhores forecasts: {potential_savings:.2f} €")
    print(f"   ({potential_savings / (end_time - start_time).days:.2f} €/dia)")

    # Salvar resultados
    df1.to_csv('results/data/results_with_forecasts.csv')
    df2.to_csv('results/data/results_perfect_forecasts.csv')
    print(f"\n✅ Resultados salvos:")
    print(f"   - results/data/results_with_forecasts.csv")
    print(f"   - results/data/results_perfect_forecasts.csv")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_test()
