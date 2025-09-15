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
from ess.billing import BillingEngine


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
    print(f"  Reduced VAT threshold: {tcfg.get('fixed_power_reduced_vat_threshold_kva', 6.9)} kVA")
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


def plot_cost_breakdown_invoice(results_df: pd.DataFrame, prices_df: pd.DataFrame, config: dict, metrics: dict, fixed_costs: dict, ledger: pd.DataFrame):
    """
    Generate an invoice-style cost breakdown comparison plot using the **ledger**
    (ensures Energy VAT is the dynamic, cycle-based value and matches billing).
    """
    plots_dir = config['output']['plots_dir']

    # Period info
    n_days = (ledger.index[-1].date() - ledger.index[0].date()).days + 1
    start_date = ledger.index[0].strftime('%Y-%m-%d')
    end_date = ledger.index[-1].strftime('%Y-%m-%d')

    # Energy totals
    total_house_consumption_kwh = float(ledger['energy_without_kwh'].sum())
    total_grid_import_kwh = float(ledger['energy_with_kwh'].sum())

    # Build cost components **from ledger sums** (no re-computation)
    without_battery = {
        'OMIE Market': float(ledger['omie_without_eur'].sum()),
        'Network Access (K2)': float(ledger['k2_without_eur'].sum()),
        'Time-of-Use Tariff': float(ledger['tariff_without_eur'].sum()),
        'Energy VAT': float(ledger['energy_vat_without_eur'].sum()),
        'IEC Tax': float(ledger['iec_without_eur'].sum()),
        'IEC VAT': float(ledger['iec_vat_without_eur'].sum()),
        # Fixed costs (as computed earlier)
        'K3 Daily Fee': float(fixed_costs['k3_daily'] * n_days),
        'Contracted Power (incl. VAT)': float(fixed_costs['power_daily'] * n_days),
        'CAV (incl. VAT)': float(fixed_costs['cav_daily'] * n_days),
        'DGEG (incl. VAT)': float(fixed_costs['dgeg_daily'] * n_days),
    }

    with_battery = {
        'OMIE Market': float(ledger['omie_with_eur'].sum()),
        'Network Access (K2)': float(ledger['k2_with_eur'].sum()),
        'Time-of-Use Tariff': float(ledger['tariff_with_eur'].sum()),
        'Energy VAT': float(ledger['energy_vat_with_eur'].sum()),
        'IEC Tax': float(ledger['iec_with_eur'].sum()),
        'IEC VAT': float(ledger['iec_vat_with_eur'].sum()),
        # Fixed costs are identical in both scenarios
        'K3 Daily Fee': float(fixed_costs['k3_daily'] * n_days),
        'Contracted Power (incl. VAT)': float(fixed_costs['power_daily'] * n_days),
        'CAV (incl. VAT)': float(fixed_costs['cav_daily'] * n_days),
        'DGEG (incl. VAT)': float(fixed_costs['dgeg_daily'] * n_days),
    }

    # Create the invoice-style plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f'Energy Cost Breakdown - Invoice Style\nPeriod: {start_date} to {end_date} ({n_days} days)', 
                 fontsize=16, fontweight='bold')

    colors = {
        'OMIE Market': '#1f77b4',
        'Network Access (K2)': '#ff7f0e', 
        'Time-of-Use Tariff': '#2ca02c',
        'Energy VAT': '#d62728',
        'IEC Tax': '#9467bd',
        'IEC VAT': '#8c564b',
        'K3 Daily Fee': '#e377c2',
        'Contracted Power (incl. VAT)': '#7f7f7f',
        'CAV (incl. VAT)': '#17becf',
        'DGEG (incl. VAT)': '#ffbb78',
    }

    def plot_invoice(ax, costs_dict, title, total_consumption_kwh):
        sorted_costs = sorted(costs_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        labels, values, bar_colors = [], [], []
        for label, value in sorted_costs:
            if abs(value) > 0.01:
                labels.append(label)
                values.append(value)
                bar_colors.append(colors.get(label, '#666666'))
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            label_x = width + 0.5 if width > 0 else width - 0.5
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'€{value:.2f}', ha=ha, va='center', fontsize=9, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Cost (EUR)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linewidth=0.8)
        total = sum(values)
        box_text = f'TOTAL: €{total:.2f}\nConsumption: {total_consumption_kwh:.1f} kWh\nAvg cost: €{(total/total_consumption_kwh if total_consumption_kwh else 0):.4f}/kWh'
        fancy_box = FancyBboxPatch((0.02, 0.02), 0.35, 0.15, boxstyle="round,pad=0.02", transform=ax.transAxes, facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(fancy_box)
        ax.text(0.195, 0.095, box_text, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', va='center')
        return total

    total_without = plot_invoice(ax1, without_battery, 'WITHOUT BATTERY', total_house_consumption_kwh)
    total_with = plot_invoice(ax2, with_battery, 'WITH BATTERY', total_grid_import_kwh)
    savings = total_without - total_with
    savings_pct = (savings / total_without * 100) if total_without > 0 else 0

    fig.text(0.5, 0.02, f'SAVINGS: €{savings:.2f} ({savings_pct:.1f}%) | Grid Reduction: {total_house_consumption_kwh - total_grid_import_kwh:.1f} kWh',
             ha='center', fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f"{plots_dir}/cost_breakdown_invoice.png", dpi=150, bbox_inches='tight')

    # Detailed grouped comparison (same keys as above)
    fig2, ax = plt.subplots(figsize=(14, 8))
    cost_categories = list(without_battery.keys())
    without_values = [without_battery[cat] for cat in cost_categories]
    with_values = [with_battery[cat] for cat in cost_categories]
    x = np.arange(len(cost_categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, without_values, width, label='Without Battery', color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, with_values, width, label='With Battery', color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., height, f'€{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    ax.set_xlabel('Cost Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost (EUR)', fontsize=12, fontweight='bold')
    ax.set_title(f'Detailed Cost Comparison - {start_date} to {end_date}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cost_categories, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    for i, cat in enumerate(cost_categories):
        diff = with_values[i] - without_values[i]
        if abs(diff) > 1:
            y_pos = max(without_values[i], with_values[i]) + 2
            ax.annotate(f'Δ €{diff:.1f}', xy=(i, y_pos), ha='center', fontsize=9, color='darkred' if diff > 0 else 'darkgreen', fontweight='bold')
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


def plot_results(results_df: pd.DataFrame, price_df: pd.DataFrame, config: dict):
    """Generate visualization plots."""
    plots_dir = config['output']['plots_dir']

    df = results_df.join(price_df[['price_omie_eur_kwh']], how='left')

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax1 = axes[0]
    ax1.plot(df.index, df['price_omie_eur_kwh'] * 1000,
             label='OMIE Price', color='blue', alpha=0.7)
    
    # Highlight battery actions
    charge_mask = results_df['battery_action'] == 'charge'
    discharge_mask = results_df['battery_action'] == 'discharge'
    
    if charge_mask.any():
        ax1.scatter(df.index[charge_mask],
                   df.loc[charge_mask, 'price_omie_eur_kwh'] * 1000,
                   color='green', alpha=0.5, s=10, label='Charging')
    if discharge_mask.any():
        ax1.scatter(df.index[discharge_mask],
                   df.loc[discharge_mask, 'price_omie_eur_kwh'] * 1000,
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
    
    # Adjust layout and save (single window)
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/simulation_overview.png", dpi=150, bbox_inches='tight')
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
    )

    # Billing
    n_days = (end_date - start_date).days + 1
    billing = BillingEngine(
        tariff_cfg=config['tariff'],
        daily_fixed_cost_eur=fixed_costs['total_daily']
    )

    ledger = billing.generate_ledger(results_df, prices_df)
    invoice = billing.invoice_from_ledger(ledger)
    metrics = billing.metrics_from_ledger(ledger)

    # Save timeline and summary
    if config['output']['save_timeline']:
        save_results(results_df, config['output']['timeline_file'])

    if config['output']['save_summary']:
        summary = {**metrics, 'fixed_costs_breakdown': fixed_costs}
        with open(config['output']['summary_file'], 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {config['output']['summary_file']}")

    # Print billing summary
    print("\n--- BILLING SUMMARY ---")
    print(f"Cost without battery:  €{invoice.total_without_battery:.2f}")
    print(f"Cost with battery:     €{invoice.total_with_battery:.2f}")
    print(f"Savings:               €{invoice.savings:.2f}")

    # Generate plots
    if config['output']['generate_plots']:
        print("\nGenerating plots...")
        plot_results(results_df, prices_df, config)
        plot_cost_breakdown_invoice(results_df, prices_df, config, metrics, fixed_costs, ledger)
  
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return results_df, metrics

if __name__ == "__main__":
    results, metrics = main()


