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
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Import components
from src.components.battery import Battery
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import SimpleTariff, BiHorariaTariff
from src.components.inverter import Inverter

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
    solar_data: pd.DataFrame,
    house_data: pd.DataFrame,
    house_mean: float
) -> Dict:
    """
    Simulate a single battery/solar configuration.

    Args:
        battery_kwh: Battery capacity in kWh
        solar_kwp: Solar capacity in kWp
        config: Configuration dictionary
        solar_data: Pre-loaded solar production DataFrame
        house_data: Pre-loaded house consumption DataFrame
        house_mean: Pre-computed mean house consumption

    Returns:
        Dictionary with results
    """
    # Create components for this configuration
    # Calculate max charge/discharge power from capacity and rate factors
    max_charge_power = battery_kwh * config['battery']['charge_rate_factor']
    max_discharge_power = battery_kwh * config['battery']['discharge_rate_factor']

    battery = Battery(
        capacity_kwh=battery_kwh,
        max_power_kw=config['battery']['max_power_kw'],
        max_charge_current_kw=max_charge_power,
        max_discharge_current_kw=max_discharge_power,
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
        scale_factor=scale_factor,
        preloaded_data=solar_data
    )

    # Create new house instance with pre-loaded data
    house_config = config['house']
    house = House(
        scale_factor=house_config.get('scale_factor', 1.0),
        preloaded_data=house_data,
        preloaded_mean=house_mean
    )

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
    rb_config = config['controller']['rule_based']
    controller = RuleBasedController(
        strategy=rb_config.get('strategy', 'simple'),
        high_price_threshold=rb_config['high_price_threshold'],
        low_soc_threshold=rb_config['low_soc_threshold'],
        high_soc_threshold=rb_config['high_soc_threshold'],
        charge_soc_min=rb_config.get('charge_soc_min', 0.5),
        bihoraria_target_soc=rb_config.get('bihoraria_target_soc', 0.85)
    )

    # Create inverter
    inverter_power = config['inverter']['max_power_kw']
    if inverter_power <= 4.0:
        inverter_price = config['optimize']['inverter']['price_3kw']
    else:
        inverter_price = config['optimize']['inverter']['price_5kw']

    inverter = Inverter(
        max_power_kw=inverter_power,
        price=inverter_price
    )

    # Create system
    system = EnergyManagementSystem(
        battery=battery,
        solar=solar,
        house=house,
        tariff=tariff,
        inverter=inverter,
        controller=controller,
        export_price=config['grid']['export_price'],
        max_export_kw=config['grid'].get('max_export_kw', float('inf'))
    )

    # Use start_date and end_date from config
    start_date = datetime.strptime(config['simulation']['start_date'], '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(config['simulation']['end_date'], '%Y-%m-%d %H:%M:%S')

    # Calculate actual simulation days
    simulation_days = (end_date - start_date).days

    results_df = system.simulate(
        start_time=start_date,
        end_time=end_date,
        dt_minutes=config['simulation']['timestep_minutes'],
        show_progress=False  # Disable individual progress bars when running in parallel
    )

    # Get costs and savings
    savings = system.get_savings(results_df)

    # Calculate total solar production from results_df
    dt_hours = config['simulation']['timestep_minutes'] / 60.0
    total_solar_kwh = (results_df['solar_power'].sum() * dt_hours)

    # Calculate house consumption
    total_consumption_kwh = (results_df['load_power'].sum() * dt_hours)

    # Calculate self-consumption rate (how much solar is used locally, not exported)
    # Self-consumption = (Solar - Export) / Solar
    self_consumption_rate = ((total_solar_kwh - savings['system_export_kwh']) / total_solar_kwh * 100) if total_solar_kwh > 0 else 0

    # Calculate self-sufficiency rate (how much consumption is met without grid import)
    # Self-sufficiency = (Consumption - Import) / Consumption
    self_sufficiency_rate = ((total_consumption_kwh - savings['system_import_kwh']) / total_consumption_kwh * 100) if total_consumption_kwh > 0 else 0

    return {
        'battery_kwh': battery_kwh,
        'solar_kwp': solar_kwp,
        'system_cost': savings['system_cost'],
        'baseline_cost': savings['baseline_no_pv_cost'],
        'monthly_savings': savings['baseline_no_pv_cost'] - savings['system_cost'],
        'grid_import_kwh': savings['system_import_kwh'],
        'grid_export_kwh': savings['system_export_kwh'],
        'solar_production_kwh': total_solar_kwh,
        'consumption_kwh': total_consumption_kwh,
        'self_consumption_pct': self_consumption_rate,
        'self_sufficiency_pct': self_sufficiency_rate,
        'simulation_days': simulation_days
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

    # Calculate number of panels (assuming 0.5 kWp per panel)
    num_panels = solar_kwp / 0.5 if solar_kwp > 0 else 0

    battery_cost = optimize_config['batery']['material_cost'] + optimize_config['batery']['repurpose_kwh'] * battery_kwh
    solar_cost = solar_kwp * optimize_config['solar']['price_per_kwp']

    # Get actual inverter price based on power rating
    inverter_power = config['inverter']['max_power_kw']
    if inverter_power <= 4.0:
        inverter_cost = optimize_config['inverter']['price_3kw']
    else:
        inverter_cost = optimize_config['inverter']['price_5kw']

    # Add structure and labor costs
    structure_cost = num_panels * optimize_config['solar'].get('structure_cost_per_panel', 0)

    labor_config = optimize_config.get('labor', {})
    labor_cost = labor_config.get('base_cost', 0) + num_panels * labor_config.get('cost_per_panel', 0)

    # Add Balance of System costs (fixed additional costs)
    bos_pv_cost = optimize_config.get('balance_of_system_pv', 0) if solar_kwp > 0 else 0
    bos_bss_cost = optimize_config.get('balance_of_system_bss', 0) if battery_kwh > 0 else 0

    total_cost = battery_cost + solar_cost + inverter_cost + structure_cost + labor_cost + bos_pv_cost + bos_bss_cost

    return total_cost


def load_shared_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Load solar and house data once for sharing across all simulations.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (solar_data, house_data, house_mean)
    """
    print("\nLoading shared data files...")

    # Load solar data
    print(f"  Loading solar data from: {config['solar']['data_file']}")
    temp_solar = SolarPanel(capacity_kw=1.0, data_file=config['solar']['data_file'])
    solar_data = temp_solar.production_data

    # Load house data
    house_config = config['house']
    print(f"  Loading house data from: {house_config['data_file']}")
    temp_house = House(data_file=house_config['data_file'])
    house_data = temp_house.consumption_data
    house_mean = temp_house._mean_consumption

    print("✓ Shared data loaded successfully\n")

    return solar_data, house_data, house_mean


def worker_function(params: Tuple, config: dict, solar_data: pd.DataFrame,
                   house_data: pd.DataFrame, house_mean: float) -> Dict:
    """
    Worker function for multiprocessing.

    Args:
        params: Tuple of (battery_kwh, solar_kwp)
        config: Configuration dictionary
        solar_data: Pre-loaded solar production DataFrame
        house_data: Pre-loaded house consumption DataFrame
        house_mean: Pre-computed mean house consumption

    Returns:
        Results dictionary
    """
    battery_kwh, solar_kwp = params

    # Simulate
    result = simulate_single_configuration(battery_kwh, solar_kwp, config,
                                          solar_data, house_data, house_mean)

    # Get actual simulation days from result
    days = result['simulation_days']

    # Calculate investment and payback
    investment_cost = calculate_investment_cost(battery_kwh, solar_kwp, config)

    # Normalize all values to monthly period (30 days)
    total_savings = result['monthly_savings']  # This is actually total savings for the simulated period
    monthly_savings = (total_savings / days) * 30  # Normalize to 30-day month

    # Also normalize grid import/export to monthly averages
    monthly_grid_import = (result['grid_import_kwh'] / days) * 30
    monthly_grid_export = (result['grid_export_kwh'] / days) * 30
    monthly_solar_production = (result['solar_production_kwh'] / days) * 30
    monthly_consumption = (result['consumption_kwh'] / days) * 30

    # Calculate payback periods (matching simulate.py logic)
    if monthly_savings > 0:
        payback_years = (investment_cost / monthly_savings) / 12
        payback_months = investment_cost / monthly_savings
    else:
        payback_years = 999  # Infinite
        payback_months = 999 * 12  # Infinite

    result['investment_cost'] = investment_cost
    result['monthly_savings'] = monthly_savings  # Now correctly normalized to monthly
    result['total_savings'] = total_savings  # Keep original total for reference
    result['grid_import_kwh'] = monthly_grid_import  # Monthly average
    result['grid_export_kwh'] = monthly_grid_export  # Monthly average
    result['solar_production_kwh'] = monthly_solar_production  # Monthly average
    result['consumption_kwh'] = monthly_consumption  # Monthly average
    result['self_consumption_pct'] = result['self_consumption_pct']  # Already a percentage
    result['self_sufficiency_pct'] = result['self_sufficiency_pct']  # Already a percentage
    result['simulation_days'] = days
    result['payback_months'] = payback_months
    result['payback_years'] = payback_years

    return result


def optimize_system(config: dict, battery_range: Tuple[float, float, float],
                   solar_range: Tuple[float, float, float],
                   n_processes: int = None) -> pd.DataFrame:
    """
    Optimize system sizing using multiprocessing.

    Args:
        config: Configuration dictionary
        battery_range: (min_kwh, max_kwh, step_kwh)
        solar_range: (min_kwp, max_kwp, step_kwp)
        n_processes: Number of parallel processes (None = use all CPUs)

    Returns:
        DataFrame with all evaluated configurations sorted by payback
    """
    # Calculate simulation days from config
    start_date = datetime.strptime(config['simulation']['start_date'], '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(config['simulation']['end_date'], '%Y-%m-%d %H:%M:%S')
    simulation_days = (end_date - start_date).days

    print("\n" + "="*80)
    print("SYSTEM SIZING OPTIMIZATION - PAYBACK ANALYSIS")
    print("="*80)

    print(f"\nOptimization parameters:")
    print(f"  Battery range: {battery_range[0]:.1f} to {battery_range[1]:.1f} kWh (step: {battery_range[2]:.1f} kWh)")
    print(f"  Solar range: {solar_range[0]:.1f} to {solar_range[1]:.1f} kWp (step: {solar_range[2]:.1f} kWp)")
    print(f"  Simulation period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({simulation_days} days, ~{simulation_days/30:.1f} months)")

    # Load shared data once before multiprocessing
    solar_data, house_data, house_mean = load_shared_data(config)

    # Generate ranges
    battery_sizes = np.arange(battery_range[0], battery_range[1] + battery_range[2], battery_range[2])
    solar_sizes = np.arange(solar_range[0], solar_range[1] + solar_range[2], solar_range[2])

    # Create all combinations
    combinations = [(bat, sol) for bat in battery_sizes for sol in solar_sizes]
    total_combinations = len(combinations)

    print(f"Total combinations to evaluate: {total_combinations}")

    # Determine number of processes
    if n_processes is None:
        n_processes = mp.cpu_count()
    print(f"Using {n_processes} parallel processes")

    print("\nStarting parallel optimization...")
    start_time = time.time()

    # Create partial function with fixed config and pre-loaded data
    worker = partial(worker_function, config=config, solar_data=solar_data,
                    house_data=house_data, house_mean=house_mean)

    # Run in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = []

        # Use imap with tqdm for progress tracking
        with tqdm(total=total_combinations, desc="Optimizing", unit="combo", ncols=100) as pbar:
            for result in pool.imap(worker, combinations):
                results.append(result)
                pbar.update(1)

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
    # Calculate consumption statistics from the first row (all should have same consumption)
    if len(df) > 0:
        first_row = df.iloc[0]
        monthly_consumption = first_row['consumption_kwh']
        # Estimate total consumption for the simulation period
        days = first_row['simulation_days']
        total_consumption = monthly_consumption * (days / 30)

        print(f"\nConsumption Statistics:")
        print(f"  Total consumption (simulation period): {total_consumption:.2f} kWh ({days} days)")
        print(f"  Average monthly consumption: {monthly_consumption:.2f} kWh")

    print(f"\nTop {n} configurations by payback period:")
    print("-" * 185)
    print(f"{'Rank':<6} {'Battery':<10} {'Solar':<10} {'Solar Prod':<12} {'Self Cons.':<12} {'Self Suff.':<12} {'Investment':<12} {'Avg Monthly':<12} {'Payback':<12} {'Payback':<10}")
    print(f"{'':6} {'(kWh)':<10} {'(kWp)':<10} {'(kWh/month)':<12} {'(%)':<12} {'(%)':<12} {'(€)':<12} {'Savings(€)':<12} {'(months)':<12} {'(years)':<10}")
    print("-" * 185)

    for idx, row in df.head(n).iterrows():
        rank = list(df.index).index(idx) + 1
        battery = row['battery_kwh']
        solar = row['solar_kwp']
        solar_prod = row['solar_production_kwh']
        self_cons = row['self_consumption_pct']
        self_suff = row['self_sufficiency_pct']
        investment = row['investment_cost']
        savings = row['monthly_savings']
        payback_months = row['payback_months']
        payback_years = row['payback_years']

        if payback_years < 999:
            print(f"{rank:<6} {battery:<10.1f} {solar:<10.1f} {solar_prod:<12.1f} {self_cons:<12.1f} {self_suff:<12.1f} {investment:<12.2f} {savings:<12.2f} {payback_months:<12.1f} {payback_years:<10.2f}")
        else:
            print(f"{rank:<6} {battery:<10.1f} {solar:<10.1f} {solar_prod:<12.1f} {self_cons:<12.1f} {self_suff:<12.1f} {investment:<12.2f} {savings:<12.2f} {'Infinite':<12} {'Infinite':<10}")


def plot_3d_optimization_results(df: pd.DataFrame, output_dir: Path):
    """
    Create and display a 3D plot of optimization results.

    Args:
        df: DataFrame with optimization results
        output_dir: Directory to save the plot
    """
    print("\n" + "="*80)
    print("CREATING 3D VISUALIZATION")
    print("="*80)

    # Filter out infinite payback values for better visualization
    df_finite = df[df['payback_years'] < 999].copy()

    if len(df_finite) == 0:
        print("WARNING: No configurations with finite payback period. Skipping 3D plot.")
        return

    # Extract data
    battery_kwh = df_finite['battery_kwh'].values
    solar_kwp = df_finite['solar_kwp'].values
    payback_years = df_finite['payback_years'].values

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot with color mapping
    scatter = ax.scatter(battery_kwh, solar_kwp, payback_years,
                        c=payback_years, cmap='RdYlGn_r',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Payback Period (years)', rotation=270, labelpad=20, fontsize=12)

    # Mark the best configuration
    best = df_finite.iloc[0]
    ax.scatter([best['battery_kwh']], [best['solar_kwp']], [best['payback_years']],
              color='red', s=300, marker='*', edgecolors='black', linewidth=2,
              label=f'Optimal: {best["battery_kwh"]:.1f} kWh, {best["solar_kwp"]:.1f} kWp, {best["payback_years"]:.2f} years')

    # Labels and title
    ax.set_xlabel('Battery Capacity (kWh)', fontsize=12, labelpad=10)
    ax.set_ylabel('Solar Capacity (kWp)', fontsize=12, labelpad=10)
    ax.set_zlabel('Payback Period (years)', fontsize=12, labelpad=10)
    ax.set_title('System Sizing Optimization: Payback Period Analysis',
                fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "optimization_3d_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 3D plot saved to: {plot_file}")

    # Display plot
    print("\nDisplaying 3D plot...")
    plt.show()

    print("="*80)


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
    battery_range = (0, 15, 3.33)

    # Solar: 0 to 15 kWp, step 0.5 kWp
    solar_range = (0, 20, 2)

    # Run optimization (uses start_date and end_date from config)
    results_df = optimize_system(config, battery_range, solar_range)

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

        if best['payback_years'] < 999:
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
                'payback_months': float(best['payback_months']) if best['payback_years'] < 999 else None,
                'payback_years': float(best['payback_years']) if best['payback_years'] < 999 else None,
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

    # Create and display 3D visualization
    plot_3d_optimization_results(results_df, output_dir)

    return results_df


if __name__ == "__main__":
    results = main()
