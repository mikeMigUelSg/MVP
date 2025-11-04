"""
System Sizing Optimization based on Payback Time

This script optimizes the sizing of solar panels and battery storage
to minimize the payback period of the investment.

Uses the same simulation logic as simulate.py to ensure consistency.
"""

import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Tuple, Dict
import multiprocessing as mp
from functools import partial
import time

# Import components
from src.components.battery import Battery
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import SimpleTariff, BiHorariaTariff

# Import controllers
from src.controllers.rule_based import RuleBasedController

# Import system
from src.system import EnergyManagementSystem


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def simulate_single_configuration(
    battery_kwh: float,
    solar_kwp: float,
    config: dict,
    house_shared: House,
    days: int = 30
) -> Dict:
    """
    Simulate a single battery/solar configuration.

    Args:
        battery_kwh: Battery capacity in kWh
        solar_kwp: Solar capacity in kWp
        config: Configuration dictionary
        house_shared: Shared House object (read-only)
        days: Number of days to simulate

    Returns:
        Dictionary with results
    """
    # Create components for this configuration
    battery = Battery(
        capacity_kwh=battery_kwh,
        max_power_kw=config['battery']['max_power_kw'],
        max_charge_current_kw=config['battery']['max_charge_current_kw'],
        max_discharge_current_kw=config['battery']['max_discharge_current_kw'],
        efficiency_charge=config['battery']['efficiency_charge'],
        efficiency_discharge=config['battery']['efficiency_discharge'],
        initial_soc=config['battery']['initial_soc'],
        min_soc=config['battery']['min_soc'],
        max_soc=config['battery']['max_soc'],
        degradation_cost_per_kwh=config['battery']['degradation_cost_per_kwh']
    )

    # Calculate scale factor for solar
    scale_factor = solar_kwp * config['optimize']['solar']['scale_factor_for_1kwp']

    solar = SolarPanel(
        capacity_kw=solar_kwp,
        data_file=config['solar']['data_file'],
        scale_factor=scale_factor
    )

    # Create new house instance (can't share due to state)
    house = House(data_file=config['house']['data_file'])

    # Create tariff
    tariff_config = config['tariff']
    if tariff_config['type'] == 'simple':
        tariff = SimpleTariff(price=tariff_config['simple']['price'])
    elif tariff_config['type'] == 'bihoraria':
        tariff = BiHorariaTariff(
            peak_price=tariff_config['bihoraria']['peak_price'],
            off_peak_price=tariff_config['bihoraria']['off_peak_price']
        )
        tariff.peak_hours_weekday = tariff_config['bihoraria']['peak_hours_weekday']
        tariff.peak_hours_weekend = tariff_config['bihoraria']['peak_hours_weekend']

    # Create controller (using rule-based for optimization)
    controller = RuleBasedController(
        high_price_threshold=config['controller']['rule_based']['high_price_threshold'],
        low_soc_threshold=config['controller']['rule_based']['low_soc_threshold'],
        high_soc_threshold=config['controller']['rule_based']['high_soc_threshold']
    )

    # Create system
    system = EnergyManagementSystem(
        battery=battery,
        solar=solar,
        house=house,
        tariff=tariff,
        controller=controller,
        export_price=config['grid']['export_price']
    )

    # Simulate for specified days
    start_date = datetime.strptime(config['simulation']['start_date'], '%Y-%m-%d %H:%M:%S')
    end_date = start_date + timedelta(days=days)

    results_df = system.simulate(
        start_time=start_date,
        end_time=end_date,
        dt_minutes=config['simulation']['timestep_minutes']
    )

    # Get costs and savings
    savings = system.get_savings(results_df)

    # Calculate total solar production from results_df
    dt_hours = config['simulation']['timestep_minutes'] / 60.0
    total_solar_kwh = (results_df['solar_power'].sum() * dt_hours)

    return {
        'battery_kwh': battery_kwh,
        'solar_kwp': solar_kwp,
        'system_cost': savings['system_cost'],
        'baseline_cost': savings['baseline_no_pv_cost'],
        'monthly_savings': savings['baseline_no_pv_cost'] - savings['system_cost'],
        'grid_import_kwh': savings['system_import_kwh'],
        'grid_export_kwh': savings['system_export_kwh'],
        'solar_production_kwh': total_solar_kwh
    }


def calculate_investment_cost(battery_kwh: float, solar_kwp: float, config: dict) -> float:
    """
    Calculate total investment cost.

    Args:
        battery_kwh: Battery capacity in kWh
        solar_kwp: Solar capacity in kWp
        config: Configuration dictionary

    Returns:
        Total investment cost in €
    """
    optimize_config = config['optimize']

    battery_cost = battery_kwh * optimize_config['batery']['price_per_kwh']
    solar_cost = solar_kwp * optimize_config['solar']['price_per_kwp']
    inverter_cost = optimize_config['inverter']['price']

    total_cost = battery_cost + solar_cost + inverter_cost

    return total_cost


def worker_function(params: Tuple, config: dict, days: int) -> Dict:
    """
    Worker function for multiprocessing.

    Args:
        params: Tuple of (battery_kwh, solar_kwp)
        config: Configuration dictionary
        days: Days to simulate

    Returns:
        Results dictionary
    """
    battery_kwh, solar_kwp = params

    # Simulate
    result = simulate_single_configuration(battery_kwh, solar_kwp, config, None, days)

    # Calculate investment and payback
    investment_cost = calculate_investment_cost(battery_kwh, solar_kwp, config)

    # Normalize all values to monthly period (30 days)
    total_savings = result['monthly_savings']  # This is actually total savings for the simulated period
    monthly_savings = (total_savings / days) * 30  # Normalize to 30-day month

    # Also normalize grid import/export to monthly averages
    monthly_grid_import = (result['grid_import_kwh'] / days) * 30
    monthly_grid_export = (result['grid_export_kwh'] / days) * 30
    monthly_solar_production = (result['solar_production_kwh'] / days) * 30

    if monthly_savings > 0:
        payback_months = investment_cost / monthly_savings
    else:
        payback_months = float('inf')

    result['investment_cost'] = investment_cost
    result['monthly_savings'] = monthly_savings  # Now correctly normalized to monthly
    result['total_savings'] = total_savings  # Keep original total for reference
    result['grid_import_kwh'] = monthly_grid_import  # Monthly average
    result['grid_export_kwh'] = monthly_grid_export  # Monthly average
    result['solar_production_kwh'] = monthly_solar_production  # Monthly average
    result['simulation_days'] = days
    result['payback_months'] = payback_months
    result['payback_years'] = payback_months / 12 if payback_months < float('inf') else float('inf')

    return result


def optimize_system(config: dict, battery_range: Tuple[float, float, float],
                   solar_range: Tuple[float, float, float],
                   simulation_days: int = 150,
                   n_processes: int = None) -> pd.DataFrame:
    """
    Optimize system sizing using multiprocessing.

    Args:
        config: Configuration dictionary
        battery_range: (min_kwh, max_kwh, step_kwh)
        solar_range: (min_kwp, max_kwp, step_kwp)
        simulation_days: Number of days to simulate (default: 150)
        n_processes: Number of parallel processes (None = use all CPUs)

    Returns:
        DataFrame with all evaluated configurations sorted by payback
    """
    print("\n" + "="*80)
    print("SYSTEM SIZING OPTIMIZATION - PAYBACK ANALYSIS")
    print("="*80)

    print(f"\nOptimization parameters:")
    print(f"  Battery range: {battery_range[0]:.1f} to {battery_range[1]:.1f} kWh (step: {battery_range[2]:.1f} kWh)")
    print(f"  Solar range: {solar_range[0]:.1f} to {solar_range[1]:.1f} kWp (step: {solar_range[2]:.1f} kWp)")
    print(f"  Simulation period: {simulation_days} days (~{simulation_days/30:.1f} months)")

    # Generate ranges
    battery_sizes = np.arange(battery_range[0], battery_range[1] + battery_range[2], battery_range[2])
    solar_sizes = np.arange(solar_range[0], solar_range[1] + solar_range[2], solar_range[2])

    # Create all combinations
    combinations = [(bat, sol) for bat in battery_sizes for sol in solar_sizes]
    total_combinations = len(combinations)

    print(f"\nTotal combinations to evaluate: {total_combinations}")

    # Determine number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    print(f"Using {n_processes} parallel processes")

    print("\nStarting parallel optimization...")
    start_time = time.time()

    # Create partial function with fixed config and days
    worker = partial(worker_function, config=config, days=simulation_days)

    # Run in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = []

        # Use imap for progress tracking
        for i, result in enumerate(pool.imap(worker, combinations), 1):
            results.append(result)

            # Print progress every 50 combinations
            if i % 50 == 0 or i == total_combinations:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (total_combinations - i) / rate if rate > 0 else 0
                print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) "
                      f"- {rate:.1f} comb/s - ETA: {remaining:.0f}s")

    elapsed_time = time.time() - start_time
    print(f"\n✓ Optimization completed in {elapsed_time:.2f}s ({elapsed_time/total_combinations:.2f}s per combination)")

    # Create DataFrame and sort by payback
    df = pd.DataFrame(results)
    df = df.sort_values('payback_months')

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    return df


def print_top_results(df: pd.DataFrame, n: int = 10):
    """Print top N results."""
    print(f"\nTop {n} configurations by payback period:")
    print("-" * 145)
    print(f"{'Rank':<6} {'Battery':<10} {'Solar':<10} {'Solar Prod':<12} {'Investment':<12} {'Avg Monthly':<12} {'Payback':<12} {'Payback':<10}")
    print(f"{'':6} {'(kWh)':<10} {'(kWp)':<10} {'(kWh/month)':<12} {'(€)':<12} {'Savings(€)':<12} {'(months)':<12} {'(years)':<10}")
    print("-" * 145)

    for idx, row in df.head(n).iterrows():
        rank = list(df.index).index(idx) + 1
        battery = row['battery_kwh']
        solar = row['solar_kwp']
        solar_prod = row['solar_production_kwh']
        investment = row['investment_cost']
        savings = row['monthly_savings']
        payback_months = row['payback_months']
        payback_years = row['payback_years']

        if payback_months < float('inf'):
            print(f"{rank:<6} {battery:<10.1f} {solar:<10.1f} {solar_prod:<12.1f} {investment:<12.2f} {savings:<12.2f} {payback_months:<12.1f} {payback_years:<10.2f}")
        else:
            print(f"{rank:<6} {battery:<10.1f} {solar:<10.1f} {solar_prod:<12.1f} {investment:<12.2f} {savings:<12.2f} {'Infinite':<12} {'Infinite':<10}")


def main():
    """Main entry point."""
    # Load configuration
    config_file = "config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)

    # Check if optimize section exists
    if 'optimize' not in config:
        print("ERROR: 'optimize' section not found in config.yaml")
        print("Please add the optimization parameters to the config file.")
        sys.exit(1)

    # Define optimization ranges
    # Battery: 0 to 30 kWh, step 1 kWh
    battery_range = (0, 15, 5)

    # Solar: 0 to 15 kWp, step 0.5 kWp
    solar_range = (0, 5, 1)

    # Get simulation days from config or use default
    simulation_days = config.get('optimize', {}).get('simulation_days', 150)

    # Run optimization
    results_df = optimize_system(config, battery_range, solar_range, simulation_days=simulation_days)

    # Print top results
    print_top_results(results_df, n=15)

    # Save results
    output_dir = Path("results/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "optimization_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to: {output_file}")

    # Save best configuration
    if len(results_df) > 0:
        best = results_df.iloc[0]

        print("\n" + "="*80)
        print("OPTIMAL CONFIGURATION")
        print("="*80)
        print(f"\nBattery capacity: {best['battery_kwh']:.1f} kWh")
        print(f"Solar capacity: {best['solar_kwp']:.1f} kWp")
        print(f"\nInvestment cost: {best['investment_cost']:.2f} €")
        print(f"Average monthly savings (estimated): {best['monthly_savings']:.2f} €")
        print(f"  Based on {best['simulation_days']} days simulation (~{best['simulation_days']/30:.1f} months)")

        if best['payback_months'] < float('inf'):
            print(f"Payback period: {best['payback_years']:.2f} years ({best['payback_months']:.1f} months)")
        else:
            print(f"Payback period: Infinite (no savings)")

        print(f"\nAverage monthly solar production: {best['solar_production_kwh']:.2f} kWh")
        print(f"Average monthly grid import: {best['grid_import_kwh']:.2f} kWh")
        print(f"Average monthly grid export: {best['grid_export_kwh']:.2f} kWh")

        # Save best config
        best_config = {
            'optimal_sizing': {
                'battery_kwh': float(best['battery_kwh']),
                'solar_kwp': float(best['solar_kwp']),
                'investment_cost_eur': float(best['investment_cost']),
                'monthly_savings_eur': float(best['monthly_savings']),
                'payback_months': float(best['payback_months']) if best['payback_months'] < float('inf') else None,
                'payback_years': float(best['payback_years']) if best['payback_years'] < float('inf') else None,
                'monthly_solar_production_kwh': float(best['solar_production_kwh']),
                'monthly_grid_import_kwh': float(best['grid_import_kwh']),
                'monthly_grid_export_kwh': float(best['grid_export_kwh']),
                'simulation_days': int(best['simulation_days']),
                'note': 'Monthly values are estimated averages based on simulation period'
            }
        }

        best_config_file = output_dir / "optimal_configuration.yaml"
        with open(best_config_file, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        print(f"\n✓ Optimal configuration saved to: {best_config_file}")

        print("="*80)

    return results_df


if __name__ == "__main__":
    results = main()
