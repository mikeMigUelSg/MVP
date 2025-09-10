#!/usr/bin/env python3
"""
run_sim.py - Main entry point for ESS energy arbitrage simulation
Updated with invoice-style cost breakdown visualization
"""

import os
import sys
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
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


def calculate_daily_fixed_costs(config: dict) -> dict:
    """Calculate daily fixed costs with proper VAT application."""
    tcfg = config['tariff']
    idx_cfg = tcfg['indexed']
    contracted_power_kva = config['power_contract']['contracted_power_kva']
    
    # Fixed costs breakdown
    costs = {}
    
    # K3 term (always standard VAT)
    costs['k3_daily'] = idx_cfg['k3_eur_day']
    
    # Power term with conditional VAT
    power_cost_base = idx_cfg['tariff_power_eur_kva_day'] * contracted_power_kva
    
    # Check if contracted power qualifies for reduced VAT on fixed power term
    power_threshold = tcfg.get('fixed_power_reduced_vat_threshold_kva', 6.9)
    if contracted_power_kva <= power_threshold:
        power_vat_rate = tcfg.get('fixed_power_reduced_vat_rate', 0.06)
        print(f"✓ Using reduced VAT ({power_vat_rate:.0%}) for power term (contracted power: {contracted_power_kva} kVA <= {power_threshold} kVA)")
    else:
        power_vat_rate = tcfg.get('fixed_power_vat_rate', 0.23)
        print(f"✓ Using standard VAT ({power_vat_rate:.0%}) for power term (contracted power: {contracted_power_kva} kVA > {power_threshold} kVA)")
    
    costs['power_daily'] = power_cost_base * (1 + power_vat_rate)
    
    # CAV fee (monthly converted to daily)
    cav_daily = tcfg['cav_fee_eur_month'] / 30 * (1 + tcfg['cav_vat_rate'])
    costs['cav_daily'] = cav_daily
    
    # DGEG fee (monthly converted to daily)  
    dgeg_daily = tcfg['dgeg_fee_eur_month'] / 30 * (1 + tcfg['dgeg_vat_rate'])
    costs['dgeg_daily'] = dgeg_daily
    
    # Total daily fixed cost
    total_daily_fixed = (
        costs['k3_daily'] + 
        costs['power_daily'] + 
        costs['cav_daily'] + 
        costs['dgeg_daily']
    )
    costs['total_daily'] = total_daily_fixed
    
    return costs


def print_tariff_summary(config: dict, fixed_costs: dict):
    """Print a summary of the tariff configuration."""
    print("\n" + "="*50)
    print("TARIFF CONFIGURATION SUMMARY")
    print("="*50)
    
    tcfg = config['tariff']
    idx_cfg = tcfg['indexed']
    contracted_power = config['power_contract']['contracted_power_kva']
    
    print(f"Tariff type: {tcfg['type']}")
    print(f"Option: {idx_cfg['option']}")
    print(f"Cycle: {idx_cfg['cycle']}")
    print(f"Contracted power: {contracted_power} kVA")
    
    print(f"\nEnergy VAT rates:")
    print(f"  Standard VAT: {tcfg['vat_rate']:.0%}")
    print(f"  Reduced VAT: {tcfg['reduced_vat_rate']:.0%}")
    print(f"  Reduced VAT threshold: {tcfg['reduced_vat_power_threshold_kva']} kVA")
    print(f"  Reduced VAT allowance: {tcfg['reduced_vat_kwh_per_30_days']} kWh per {tcfg['vat_cycle_days']} days")
    
    print(f"\nFixed power term VAT rates:")
    power_threshold = tcfg.get('fixed_power_reduced_vat_threshold_kva', 6.9)
    if contracted_power <= power_threshold:
        rate_used = tcfg.get('fixed_power_reduced_vat_rate', 0.06)
        print(f"  Applied VAT: {rate_used:.0%} (reduced - power <= {power_threshold} kVA)")
    else:
        rate_used = tcfg.get('fixed_power_vat_rate', 0.23)
        print(f"  Applied VAT: {rate_used:.0%} (standard - power > {power_threshold} kVA)")
    
    print(f"\nDaily fixed costs:")
    print(f"  K3 term: €{fixed_costs['k3_daily']:.4f}")
    print(f"  Power term (with VAT): €{fixed_costs['power_daily']:.4f}")
    print(f"  CAV fee: €{fixed_costs['cav_daily']:.4f}")
    print(f"  DGEG fee: €{fixed_costs['dgeg_daily']:.4f}")
    print(f"  Total daily fixed: €{fixed_costs['total_daily']:.4f}")
    print("="*50)


def plot_cost_breakdown_invoice(results_df: pd.DataFrame, config: dict, metrics: dict, fixed_costs: dict):
    """
    Generate an invoice-style cost breakdown comparison plot.
    Shows all cost components for scenarios with and without battery.
    """
    plots_dir = config['output']['plots_dir']
    
    # Calculate period information
    n_days = (results_df.index[-1] - results_df.index[0]).days + 1
    start_date = results_df.index[0].strftime('%Y-%m-%d')
    end_date = results_df.index[-1].strftime('%Y-%m-%d')
    
    # Get tariff configuration
    tcfg = config['tariff']
    contracted_power = config['power_contract']['contracted_power_kva']
    
    # Calculate energy consumption totals
    total_house_consumption_kwh = results_df['house_consumption_kwh'].sum()
    total_grid_import_kwh = results_df['total_grid_import_kwh'].sum()
    
    # Calculate average prices
    avg_omie_price = results_df['price_omie_eur_kwh'].mean()
    avg_final_price = results_df['price_final_eur_kwh'].mean()
    
    # WITHOUT BATTERY - Cost Components
    without_battery = {}
    
    # Energy costs (simplified - all at same VAT rate based on contracted power)
    energy_vat_rate = tcfg['reduced_vat_rate'] if contracted_power <= tcfg['reduced_vat_power_threshold_kva'] else tcfg['vat_rate']
    
    # Base energy cost components
    omie_cost = (results_df['price_omie_eur_kwh'] * results_df['house_consumption_kwh']).sum()
    energy_tariff_cost = (results_df['tariff_energy_eur_kwh'] * results_df['house_consumption_kwh']).sum()
    k2_cost = tcfg['indexed']['k2_eur_kwh'] * total_house_consumption_kwh
    
    without_battery['OMIE Market'] = omie_cost
    without_battery['Network Access (K2)'] = k2_cost
    without_battery['Time-of-Use Tariff'] = energy_tariff_cost
    without_battery['Energy VAT'] = (omie_cost + k2_cost + energy_tariff_cost) * energy_vat_rate
    
    # IEC tax
    iec_cost = tcfg['iec_tax_eur_kwh'] * total_house_consumption_kwh
    without_battery['IEC Tax'] = iec_cost
    without_battery['IEC VAT'] = iec_cost * tcfg['iec_vat_rate']
    
    # Fixed costs
    without_battery['K3 Daily Fee'] = fixed_costs['k3_daily'] * n_days
    without_battery['Contracted Power'] = tcfg['indexed']['tariff_power_eur_kva_day'] * contracted_power * n_days
    
    # Power VAT (conditional)
    power_vat_rate = tcfg['fixed_power_reduced_vat_rate'] if contracted_power <= tcfg['fixed_power_reduced_vat_threshold_kva'] else tcfg['fixed_power_vat_rate']
    without_battery['Power VAT'] = without_battery['Contracted Power'] * power_vat_rate
    
    # Other fees
    without_battery['CAV Fee'] = tcfg['cav_fee_eur_month'] / 30 * n_days
    without_battery['CAV VAT'] = without_battery['CAV Fee'] * tcfg['cav_vat_rate']
    without_battery['DGEG Fee'] = tcfg['dgeg_fee_eur_month'] / 30 * n_days
    without_battery['DGEG VAT'] = without_battery['DGEG Fee'] * tcfg['dgeg_vat_rate']
    
    # WITH BATTERY - Cost Components
    with_battery = {}
    
    # Energy costs with battery (reduced grid import)
    omie_cost_battery = (results_df['price_omie_eur_kwh'] * results_df['total_grid_import_kwh']).sum()
    energy_tariff_cost_battery = (results_df['tariff_energy_eur_kwh'] * results_df['total_grid_import_kwh']).sum()
    k2_cost_battery = tcfg['indexed']['k2_eur_kwh'] * total_grid_import_kwh
    
    with_battery['OMIE Market'] = omie_cost_battery
    with_battery['Network Access (K2)'] = k2_cost_battery
    with_battery['Time-of-Use Tariff'] = energy_tariff_cost_battery
    with_battery['Energy VAT'] = (omie_cost_battery + k2_cost_battery + energy_tariff_cost_battery) * energy_vat_rate
    
    # IEC tax
    iec_cost_battery = tcfg['iec_tax_eur_kwh'] * total_grid_import_kwh
    with_battery['IEC Tax'] = iec_cost_battery
    with_battery['IEC VAT'] = iec_cost_battery * tcfg['iec_vat_rate']
    
    # Fixed costs (same as without battery)
    with_battery['K3 Daily Fee'] = without_battery['K3 Daily Fee']
    with_battery['Contracted Power'] = without_battery['Contracted Power']
    with_battery['Power VAT'] = without_battery['Power VAT']
    with_battery['CAV Fee'] = without_battery['CAV Fee']
    with_battery['CAV VAT'] = without_battery['CAV VAT']
    with_battery['DGEG Fee'] = without_battery['DGEG Fee']
    with_battery['DGEG VAT'] = without_battery['DGEG VAT']
    
    # Create the invoice-style plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f'Energy Cost Breakdown - Invoice Style\nPeriod: {start_date} to {end_date} ({n_days} days)', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for different cost categories
    colors = {
        'OMIE Market': '#1f77b4',
        'Network Access (K2)': '#ff7f0e', 
        'Time-of-Use Tariff': '#2ca02c',
        'Energy VAT': '#d62728',
        'IEC Tax': '#9467bd',
        'IEC VAT': '#8c564b',
        'K3 Daily Fee': '#e377c2',
        'Contracted Power': '#7f7f7f',
        'Power VAT': '#bcbd22',
        'CAV Fee': '#17becf',
        'CAV VAT': '#aec7e8',
        'DGEG Fee': '#ffbb78',
        'DGEG VAT': '#98df8a'
    }
    
    def plot_invoice(ax, costs_dict, title, total_consumption_kwh):
        """Helper function to plot one invoice."""
        # Sort costs by value for better visualization
        sorted_costs = sorted(costs_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        labels = []
        values = []
        bar_colors = []
        
        for label, value in sorted_costs:
            if abs(value) > 0.01:  # Only show costs > 1 cent
                labels.append(label)
                values.append(value)
                bar_colors.append(colors.get(label, '#666666'))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            label_x = width + 0.5 if width > 0 else width - 0.5
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'€{value:.2f}', 
                   ha=ha, va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Cost (EUR)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        # Add total box
        total = sum(values)
        box_text = f'TOTAL: €{total:.2f}\n'
        box_text += f'Consumption: {total_consumption_kwh:.1f} kWh\n'
        box_text += f'Avg cost: €{total/total_consumption_kwh:.4f}/kWh'
        
        # Create fancy box for total
        fancy_box = FancyBboxPatch((0.02, 0.02), 0.35, 0.15,
                                   boxstyle="round,pad=0.02",
                                   transform=ax.transAxes,
                                   facecolor='lightgray',
                                   edgecolor='black',
                                   linewidth=2,
                                   alpha=0.9)
        ax.add_patch(fancy_box)
        ax.text(0.195, 0.095, box_text, transform=ax.transAxes,
               fontsize=10, fontweight='bold', ha='center', va='center')
        
        return total
    
    # Plot both invoices
    total_without = plot_invoice(ax1, without_battery, 'WITHOUT BATTERY', total_house_consumption_kwh)
    total_with = plot_invoice(ax2, with_battery, 'WITH BATTERY', total_grid_import_kwh)
    
    # Add savings information at the bottom
    savings = total_without - total_with
    savings_pct = (savings / total_without * 100) if total_without > 0 else 0
    
    fig.text(0.5, 0.02, 
            f'SAVINGS: €{savings:.2f} ({savings_pct:.1f}%) | Grid Reduction: {total_house_consumption_kwh - total_grid_import_kwh:.1f} kWh',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f"{plots_dir}/cost_breakdown_invoice.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create a second detailed comparison plot
    fig2, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    cost_categories = list(without_battery.keys())
    without_values = [without_battery[cat] for cat in cost_categories]
    with_values = [with_battery[cat] for cat in cost_categories]
    
    x = np.arange(len(cost_categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, without_values, width, label='Without Battery', 
                   color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, with_values, width, label='With Battery',
                   color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.5:  # Only label bars > €0.50
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'€{height:.1f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)
    
    ax.set_xlabel('Cost Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost (EUR)', fontsize=12, fontweight='bold')
    ax.set_title(f'Detailed Cost Comparison - {start_date} to {end_date}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cost_categories, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add difference annotations for major components
    for i, cat in enumerate(cost_categories):
        diff = with_values[i] - without_values[i]
        if abs(diff) > 1:  # Only show differences > €1
            y_pos = max(without_values[i], with_values[i]) + 2
            ax.annotate(f'Δ €{diff:.1f}',
                       xy=(i, y_pos),
                       ha='center',
                       fontsize=9,
                       color='darkred' if diff > 0 else 'darkgreen',
                       fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cost_comparison_detailed.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nCost breakdown plots saved to {plots_dir}")
    
    return {
        'without_battery': without_battery,
        'with_battery': with_battery,
        'total_without': total_without,
        'total_with': total_with,
        'savings': savings
    }


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
    
    # Calculate fixed costs with proper VAT
    fixed_costs = calculate_daily_fixed_costs(config)
    
    # Parse dates
    start_date = datetime.strptime(config['period']['start_date'], "%Y-%m-%d")
    if 'end_date' in config['period']:
        end_date = datetime.strptime(config['period']['end_date'], "%Y-%m-%d")
    else:
        end_date = start_date + timedelta(days=config['period']['num_days'] - 1)
    
    print(f"\nSimulation period: {start_date.date()} to {end_date.date()}")
    
    # Print tariff summary
    print_tariff_summary(config, fixed_costs)
    
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
        prices_df = apply_indexed_tariff(prices_df, config['tariff'])
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
    
    print(f"\nStrategy: {strategy_type.capitalize()}")
    
    # Create and run simulator
    simulator = EnergyArbitrageSimulator(battery, strategy)

    print("\nRunning simulation...")
    results_df = simulator.run(
        consumption_df,
        prices_df,
        start_date,
        end_date,
        vat_rate=config['tariff']['vat_rate'],
        reduced_vat_rate=config['tariff']['reduced_vat_rate'],
        iec_vat_rate=config['tariff']['iec_vat_rate'],
        contracted_power_kva=config['power_contract']['contracted_power_kva'],
        vat_reduced_power_threshold_kva=config['tariff']['reduced_vat_power_threshold_kva'],
        daily_fixed_cost_eur=fixed_costs['total_daily']
    )
    
    # Calculate and display metrics
    metrics = simulator.calculate_summary_metrics(results_df)
    
    # Add fixed costs breakdown to metrics
    metrics['fixed_costs_breakdown'] = fixed_costs
    
    simulator.print_summary(metrics)
    
    # Print additional fixed costs summary
    n_days = (end_date - start_date).days + 1
    print(f"\n--- FIXED COSTS BREAKDOWN ({n_days} days) ---")
    print(f"K3 term:         €{fixed_costs['k3_daily'] * n_days:.2f}")
    print(f"Power term:      €{fixed_costs['power_daily'] * n_days:.2f}")
    print(f"CAV fees:        €{fixed_costs['cav_daily'] * n_days:.2f}")
    print(f"DGEG fees:       €{fixed_costs['dgeg_daily'] * n_days:.2f}")
    print(f"Total fixed:     €{fixed_costs['total_daily'] * n_days:.2f}")
    
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
        
        # Generate the new invoice-style cost breakdown
        print("\nGenerating cost breakdown invoice...")
        cost_breakdown = plot_cost_breakdown_invoice(results_df, config, metrics, fixed_costs)
        
        # Print cost breakdown summary
        print("\n--- DETAILED COST BREAKDOWN ---")
        print("\nWITHOUT BATTERY:")
        for component, cost in cost_breakdown['without_battery'].items():
            if abs(cost) > 0.01:
                print(f"  {component:.<30} €{cost:>8.2f}")
        print(f"  {'TOTAL':.<30} €{cost_breakdown['total_without']:>8.2f}")
        
        print("\nWITH BATTERY:")
        for component, cost in cost_breakdown['with_battery'].items():
            if abs(cost) > 0.01:
                print(f"  {component:.<30} €{cost:>8.2f}")
        print(f"  {'TOTAL':.<30} €{cost_breakdown['total_with']:>8.2f}")
        
        print(f"\n  {'SAVINGS':.<30} €{cost_breakdown['savings']:>8.2f}")
        
        # Save cost breakdown to JSON
        breakdown_file = config['output']['summary_file'].replace('.json', '_cost_breakdown.json')
        with open(breakdown_file, 'w') as f:
            json.dump(cost_breakdown, f, indent=2, default=str)
        print(f"\nCost breakdown saved to {breakdown_file}")
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return results_df, metrics


if __name__ == "__main__":
    results, metrics = main()