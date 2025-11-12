"""
Test that curtailment appears in the energy balance comparison table
"""
import yaml
import pandas as pd
from datetime import datetime, timedelta

from src.components.battery import Battery
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import SimpleTariff
from src.components.inverter import Inverter
from src.controllers.rule_based import RuleBasedController
from src.system import EnergyManagementSystem

# Load config to get max_export_kw
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create components
battery = Battery(
    capacity_kwh=5.0,
    max_power_kw=5.0,
    efficiency_charge=0.94,
    efficiency_discharge=0.94,
    initial_soc=0.5,
    min_soc=0.1,
    max_soc=0.9
)

solar = SolarPanel(
    capacity_kw=config['solar']['solar_capacity_kwp'],
    data_file=config['solar']['data_file'],
    scale_factor=config['solar']['scale_factor_for_1kwp']
)

house = House(data_file=config['house']['data_file'])
tariff = SimpleTariff(price=0.15)
inverter = Inverter(max_power_kw=3.0)  # Small inverter for testing

controller = RuleBasedController(
    high_price_threshold=0.20,
    strategy='simple'
)

system = EnergyManagementSystem(
    battery=battery,
    solar=solar,
    house=house,
    tariff=tariff,
    inverter=inverter,
    controller=controller,
    export_price=0.05,
    max_export_kw=2.0  # Low export limit for testing
)

# Run a short simulation (1 day)
start_time = datetime(2025, 7, 1, 0, 0, 0)
end_time = start_time + timedelta(days=1)

print("Running short simulation to test curtailment table...")
results_df = system.simulate(start_time, end_time, dt_minutes=15)

# Get energy balance comparison
energy_balance = system.get_energy_balance_comparison(results_df)

print("\n" + "="*70)
print("ENERGY BALANCE COMPARISON (showing curtailment)")
print("="*70)

# Extract data
with_bat = energy_balance['system_with_battery']
pv_only = energy_balance['pv_without_battery']
no_pv = energy_balance['no_pv_no_battery']

# Display table
metric_width = 30
value_width = 18

print(f"\n{'Metric':<{metric_width}} | {'With Battery':^{value_width}} | {'PV Only':^{value_width}} | {'No PV':^{value_width}}")
print("-" * metric_width + "-+-" + "-" * value_width + "-+-" + "-" * value_width + "-+-" + "-" * value_width)

def fmt_kwh(val):
    return f"{val:>10.2f} kWh"

print(f"{'Total Solar Produced':<{metric_width}} | {fmt_kwh(with_bat['solar_produced_kwh']):^{value_width}} | {fmt_kwh(pv_only['solar_produced_kwh']):^{value_width}} | {fmt_kwh(no_pv['solar_produced_kwh']):^{value_width}}")
print(f"{'Total Solar Curtailed':<{metric_width}} | {fmt_kwh(with_bat['solar_curtailed_kwh']):^{value_width}} | {fmt_kwh(pv_only['solar_curtailed_kwh']):^{value_width}} | {fmt_kwh(no_pv['solar_curtailed_kwh']):^{value_width}}")
print(f"{'Total Load Consumed':<{metric_width}} | {fmt_kwh(with_bat['load_consumed_kwh']):^{value_width}} | {fmt_kwh(pv_only['load_consumed_kwh']):^{value_width}} | {fmt_kwh(no_pv['load_consumed_kwh']):^{value_width}}")
print(f"{'Total Grid Import':<{metric_width}} | {fmt_kwh(with_bat['grid_import_kwh']):^{value_width}} | {fmt_kwh(pv_only['grid_import_kwh']):^{value_width}} | {fmt_kwh(no_pv['grid_import_kwh']):^{value_width}}")
print(f"{'Total Grid Export':<{metric_width}} | {fmt_kwh(with_bat['grid_export_kwh']):^{value_width}} | {fmt_kwh(pv_only['grid_export_kwh']):^{value_width}} | {fmt_kwh(no_pv['grid_export_kwh']):^{value_width}}")

print("\n" + "="*70)
print("✓ Curtailment successfully added to the table!")
print("="*70)

# Show summary statistics
summary = system.get_summary()
if 'total_solar_curtailed_kwh' in summary:
    curtail_pct = summary.get('curtailment_rate', 0) * 100
    print(f"\nSystem Summary:")
    print(f"  Total solar curtailed: {summary['total_solar_curtailed_kwh']:.2f} kWh")
    print(f"  Curtailment rate: {curtail_pct:.2f}%")
