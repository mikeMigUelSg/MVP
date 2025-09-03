#!/usr/bin/env python3
"""
run_sim.py - Main entry point for ESS energy arbitrage simulation
"""

import os
import sys
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ess.battery import Battery
from ess.io import prepare_simulation_data, save_results
from ess.strategies import ArbitrageStrategy, OptimalArbitrageStrategy
from ess.simulator import EnergyArbitrageSimulator


def load_config(config_path: str = "configs/scenario.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_output_dirs(config: dict):
    """Create output directories if they don't exist."""
    Path("outputs").mkdir(exist_ok=True)
    if config['output'].get('generate_plots'):
        Path(config['output']['plots_dir']).mkdir(parents=True, exist_ok=True)


def plot_results(results_df: pd.DataFrame, config: dict):
    """Generate visualization plots."""
    plots_dir = config['output']['plots_dir']
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Prices and battery action
    ax1 = axes[0]
    ax1.plot(results_df.index, results_df['price_omie_eur_kwh'] * 1000, 
             label='OMIE Price', color='blue', alpha=0.7)
    
    # Highlight battery actions
    charge_mask = results_df['battery_action'] == 'charge'
    discharge_mask = results_df['battery_action'] == 'discharge'
    
    if charge_mask.any():
        ax1.scatter(results_df.index[charge_mask], 
                   results_df.loc[charge_mask, 'price_omie_eur_kwh'] * 1000,
                   color='green', alpha=0.5, s=10, label='Charging')
    if discharge_mask.any():
        ax1.scatter(results_df.index[discharge_mask], 
                   results_df.loc[discharge_mask, 'price_omie_eur_kwh'] * 1000,
                   color='red', alpha=0.5, s=10, label='Discharging')
    
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Electricity Prices and Battery Actions')
    
    # Plot 2: Power flows
    ax2 = axes[1]
    ax2.plot(results_df.index, results_df['consumption_kw'], 
             label='Consumption', color='orange', alpha=0.7)
    ax2.fill_between(results_df.index, 0, results_df['battery_charge_kwh'] * 4, 
                     color='green', alpha=0.3, label='Battery Charge')
    ax2.fill_between(results_df.index, 0, -results_df['battery_discharge_kwh'] * 4, 
                     color='red', alpha=0.3, label='Battery Discharge')
    ax2.set_ylabel('Power (kW)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Power Flows')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 3: Battery State of Charge
    ax3 = axes[2]
    ax3.plot(results_df.index, results_df['battery_soc'] * 100, 
             label='SOC', color='purple', linewidth=2)
    ax3.fill_between(results_df.index, 0, results_df['battery_soc'] * 100, 
                     color='purple', alpha=0.2)
    ax3.set_ylabel('SOC (%)')
    ax3.set_ylim([0, 100])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Battery State of Charge')
    
    # Plot 4: Cumulative savings
    ax4 = axes[3]
    cumulative_savings = results_df['savings_eur'].cumsum()
    ax4.plot(results_df.index, cumulative_savings, 
             label='Cumulative Savings', color='darkgreen', linewidth=2)
    ax4.fill_between(results_df.index, 0, cumulative_savings, 
                     color='green', alpha=0.2)
    ax4.set_ylabel('Savings (EUR)')
    ax4.set_xlabel('Time')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Cumulative Savings')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/simulation_overview.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Daily summary plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    daily_savings = results_df['savings_eur'].resample('D').sum()
    daily_savings.plot(kind='bar', ax=ax, color='green', alpha=0.7)
    ax.set_ylabel('Daily Savings (EUR)')
    ax.set_xlabel('Date')
    ax.set_title('Daily Savings from Energy Arbitrage')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/daily_savings.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main simulation function."""
    print("="*60)
    print("ESS ENERGY ARBITRAGE SIMULATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    create_output_dirs(config)
    
    # Parse dates
    start_date = datetime.strptime(config['period']['start_date'], "%Y-%m-%d")
    if 'end_date' in config['period']:
        end_date = datetime.strptime(config['period']['end_date'], "%Y-%m-%d")
    else:
        end_date = start_date + timedelta(days=config['period']['num_days'] - 1)
    
    print(f"\nSimulation period: {start_date.date()} to {end_date.date()}")
    
    # Load and prepare data
    print("\nPreparing simulation data...")
    consumption_df, prices_df = prepare_simulation_data(
        config['consumption']['profile_file'],
        config['consumption']['annual_consumption_kwh'],
        start_date,
        end_date,
        config['consumption']['profile_column'],
        consumption_model=config['consumption'].get('consumption_model', False)
    )

    # Rename price column for clarity
    if 'price_eur_per_kwh' in prices_df.columns:
        prices_df = prices_df.rename(columns={'price_eur_per_kwh': 'price_omie_eur_kwh'})

    # Apply tariff to compute final prices
    from ess.tariff import apply_indexed_tariff
    if config['tariff']['type'] == 'indexed':
        prices_df = apply_indexed_tariff(prices_df, config['tariff']['indexed'], config['tariff']['vat_rate'])
        idx_cfg = config['tariff']['indexed']
        daily_fixed_cost = idx_cfg['k3_eur_day'] + idx_cfg['tariff_power_eur_kva_day'] * config['power_contract']['contracted_power_kva']
    else:
        raise NotImplementedError("Only indexed tariff is implemented for now")

    # Create battery
    battery = Battery(
        capacity_kwh=config['battery']['capacity_kwh'],
        soc_init=config['battery']['soc_init'],
        max_charge_kw=config['battery']['max_charge_kw'],
        max_discharge_kw=config['battery']['max_discharge_kw'],
        efficiency=config['battery']['efficiency'],
        soc_min=config['battery']['soc_min'],
        soc_max=config['battery']['soc_max']
    )
    
    # Create strategy
    strategy_type = config['strategy']['type']
    allow_export = config['strategy'].get('allow_grid_export', False)
    if strategy_type == 'arbitrage':
        strategy = ArbitrageStrategy(
            charge_threshold_percentile=config['strategy']['arbitrage']['charge_threshold_percentile'],
            discharge_threshold_percentile=config['strategy']['arbitrage']['discharge_threshold_percentile'],
            min_price_spread=config['strategy']['arbitrage']['min_price_spread_eur_mwh'],
            allow_grid_export=allow_export
        )
    elif strategy_type == 'optimal':
        strategy = OptimalArbitrageStrategy(
            optimization_window_hours=config['strategy']['optimal']['optimization_window_hours'],
            use_simple_optimization=config['strategy']['optimal']['use_simple_optimization'],
            allow_grid_export=allow_export
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    print(f"Strategy: {strategy_type.capitalize()}")
    
    # Create and run simulator
    simulator = EnergyArbitrageSimulator(battery, strategy)

    print("\nRunning simulation...")
    results_df = simulator.run(
        consumption_df,
        prices_df,
        start_date,
        end_date,
        vat_rate=config['tariff']['vat_rate'],
        daily_fixed_cost_eur=daily_fixed_cost
    )
    
    # Calculate and display metrics
    metrics = simulator.calculate_summary_metrics(results_df)
    simulator.print_summary(metrics)
    
    # Save results
    if config['output']['save_timeline']:
        save_results(results_df, config['output']['timeline_file'])
    
    if config['output']['save_summary']:
        with open(config['output']['summary_file'], 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nSummary saved to {config['output']['summary_file']}")
    
    # Generate plots
    if config['output']['generate_plots']:
        print("\nGenerating plots...")
        plot_results(results_df, config)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return results_df, metrics


if __name__ == "__main__":
    results, metrics = main()