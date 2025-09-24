#!/usr/bin/env python3
"""
run_sim.py - Main entry point for ESS energy arbitrage simulation
Updated with invoice-style cost breakdown visualization
FIXED: All plots now appear simultaneously
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
from ess.tariff import create_tariff_processor, get_active_tariff_type


def load_config(config_path: str = "configs/scenario.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_output_dirs(config: dict):
    """Create output directories if they don't exist."""
    Path("outputs").mkdir(exist_ok=True)
    if config['output'].get('generate_plots'):
        Path(config['output']['plots_dir']).mkdir(parents=True, exist_ok=True)


def calculate_daily_fixed_costs(
    config: dict, tariff_processor=None
) -> dict:
    """Calculate daily fixed costs using the configured tariff processor."""

    tcfg = config['tariff']
    contracted_power_kva = config['power_contract']['contracted_power_kva']
    processor = tariff_processor or create_tariff_processor(tcfg)

    fixed_terms, metadata = processor.compute_daily_fixed_costs(contracted_power_kva)

    costs = {
        'k3_daily': float(fixed_terms.get('k3_daily', 0.0)),
        'power_daily': float(fixed_terms.get('power_daily', 0.0)),
    }

    metadata = metadata or {}

    if 'power_vat_rate' in metadata:
        vat_rate = metadata['power_vat_rate']
        vat_type = metadata.get('vat_type', 'standard')
        threshold = metadata.get('threshold_kva')
        contracted = metadata.get('contracted_power_kva', contracted_power_kva)
        standard_vat = tcfg.get('fixed_power_vat_rate', 0.23)

        if vat_type == 'reduced':
            print(
                f"✓ Using reduced VAT ({vat_rate:.0%}) for power term (contracted power: {contracted} kVA <= {threshold} kVA)"
            )
        elif vat_type == 'tariff_only_reduced':
            print(
                "✓ Applying reduced VAT "
                f"({vat_rate:.0%}) only to TAR potência component; remaining fixed term taxed at {standard_vat:.0%}"
            )
        else:
            print(
                f"✓ Using standard VAT ({vat_rate:.0%}) for power term (contracted power: {contracted} kVA > {threshold} kVA)"
            )
    elif metadata.get('power_term_source'):
        source = metadata['power_term_source'].replace('_', ' ')
        print(
            f"✓ Using provided power term ({source}) -> €{costs['power_daily']:.4f}/day"
        )

    if metadata.get('k3_source') == 'configured' and costs['k3_daily']:
        print(f"✓ Using provided K3 daily cost -> €{costs['k3_daily']:.4f}/day")

    # CAV fee (monthly converted to daily)
    costs['cav_daily'] = tcfg['cav_fee_eur_month'] / 30 * (1 + tcfg['cav_vat_rate'])

    # DGEG fee (monthly converted to daily)
    costs['dgeg_daily'] = tcfg['dgeg_fee_eur_month'] / 30 * (1 + tcfg['dgeg_vat_rate'])

    costs['total_daily'] = (
        costs['k3_daily']
        + costs['power_daily']
        + costs['cav_daily']
        + costs['dgeg_daily']
    )

    return costs


def print_tariff_summary(config: dict, fixed_costs: dict, tariff_processor=None):
    """Print a summary of the tariff configuration."""

    print("\n" + "=" * 50)
    print("TARIFF CONFIGURATION SUMMARY")
    print("=" * 50)

    tcfg = config['tariff']
    contracted_power = config['power_contract']['contracted_power_kva']
    processor = tariff_processor or create_tariff_processor(tcfg)

    active_type = get_active_tariff_type(tcfg)
    print(f"Active tariff: {active_type}")

    # Show what is available in the YAML for clarity
    available = []
    for key in ("indexed", "simples", "bi_horaria"):
        if key in tcfg:
            available.append(key)
    if available:
        print(f"Available tariffs in config: {', '.join(available)}")
        print("Selection priority: TARIFF_ACTIVE env var > tariff.active (YAML).")

    # If env override is being used, make it explicit
    import os as _os
    if _os.getenv("TARIFF_ACTIVE"):
        print(f"(Overridden by env var TARIFF_ACTIVE={_os.getenv('TARIFF_ACTIVE')})")
    print(f"Contracted power: {contracted_power} kVA")

    details = processor.summary_lines()
    if details:
        print("\nTariff details:")
        for line in details:
            print(f"  {line}")

    print("\nEnergy VAT rates:")
    print(f"  Standard VAT: {tcfg['vat_rate']:.0%}")
    print(f"  Reduced VAT: {tcfg['reduced_vat_rate']:.0%}")
    print(
        f"  Reduced VAT threshold: {tcfg.get('fixed_power_reduced_vat_threshold_kva', 6.9)} kVA"
    )
    print(
        f"  Reduced VAT allowance: {tcfg['reduced_vat_kwh_per_30_days']} kWh per {tcfg['vat_cycle_days']} days"
    )

    print("\nDaily fixed costs:")
    print(f"  K3 term: €{fixed_costs['k3_daily']:.4f}")
    print(f"  Power term: €{fixed_costs['power_daily']:.4f}")
    print(f"  CAV fee: €{fixed_costs['cav_daily']:.4f}")
    print(f"  DGEG fee: €{fixed_costs['dgeg_daily']:.4f}")
    print(f"  Total daily fixed: €{fixed_costs['total_daily']:.4f}")
    print("=" * 50)


def run_billing(results_df, prices_df, config, fixed_costs):
    """Run billing with refactored modular structure."""
    # Create billing engine with fixed costs
    billing = BillingEngine(
        tariff_cfg=config['tariff'],
        fixed_costs=fixed_costs  # Pass the fixed_costs dict directly
    )
    
    # Generate ledger and invoice
    ledger = billing.generate_ledger(results_df, prices_df)
    invoice = billing.invoice_from_ledger(ledger)
    metrics = billing.metrics_from_ledger(ledger)
    
    return ledger, invoice, metrics

def print_modular_invoice(invoice):
    """Print invoice with modular structure."""
    print("\n" + "="*60)
    print("MODULAR INVOICE BREAKDOWN")
    print("="*60)
    
    # 1. ELECTRICITY TERM
    print("\n1. ELECTRICITY TERM (Variable)")
    print("-" * 40)
    print("Without Battery:")
    for component, value in invoice.electricity_term_without.items():
        print(f"  {component:30s} €{value:8.2f}")
    electricity_total_without = sum(invoice.electricity_term_without.values())
    print(f"  {'SUBTOTAL':30s} €{electricity_total_without:8.2f}")
    
    print("\nWith Battery:")
    for component, value in invoice.electricity_term_with.items():
        print(f"  {component:30s} €{value:8.2f}")
    electricity_total_with = sum(invoice.electricity_term_with.values())
    print(f"  {'SUBTOTAL':30s} €{electricity_total_with:8.2f}")
    
    electricity_savings = electricity_total_without - electricity_total_with
    print(f"\n  {'ELECTRICITY SAVINGS':30s} €{electricity_savings:8.2f}")
    
    # 2. POWER TERM
    print("\n2. POWER TERM (Fixed)")
    print("-" * 40)
    for component, value in invoice.power_term.items():
        print(f"  {component:30s} €{value:8.2f}")
    power_total = sum(invoice.power_term.values())
    print(f"  {'SUBTOTAL':30s} €{power_total:8.2f}")
    
    # 3. TAXES
    print("\n3. TAXES")
    print("-" * 40)
    for component, value in invoice.taxes.items():
        print(f"  {component:30s} €{value:8.2f}")
    taxes_total = sum(invoice.taxes.values())
    print(f"  {'SUBTOTAL':30s} €{taxes_total:8.2f}")
    
    # 4. VAT BREAKDOWN (Energy VAT only)
    print("\n4. VAT BREAKDOWN (Energy VAT only)")
    print("-" * 40)
    vb = invoice.vat_breakdown
    # Reduced VAT
    red = vb['reduced']
    print("Reduced VAT:")
    print(f"  kWh (without/with):        {red['kwh_without']:7.1f} / {red['kwh_with']:7.1f}")
    print(f"  Base € (without/with):     {red['base_without_eur']:8.2f} / {red['base_with_eur']:8.2f}")
    print(f"  VAT € (without/with):      {red['vat_without_eur']:8.2f} / {red['vat_with_eur']:8.2f}")
    # Standard VAT
    std = vb['standard']
    print("Standard VAT:")
    print(f"  kWh (without/with):        {std['kwh_without']:7.1f} / {std['kwh_with']:7.1f}")
    print(f"  Base € (without/with):     {std['base_without_eur']:8.2f} / {std['base_with_eur']:8.2f}")
    print(f"  VAT € (without/with):      {std['vat_without_eur']:8.2f} / {std['vat_with_eur']:8.2f}")

    # 5. Period breakdown if available (exact VAT allocation)
    if getattr(invoice, "period_breakdown", None):
        pb_all = invoice.period_breakdown
        def _print_period_table(title: str, pb: dict):
            print(f"\n5. {title}")
            print("-" * 40)
            header = (
                f"{'Period':10s} {'kWh':>8s} {'avg €/kWh':>10s} {'Base €':>8s} "
                f"{'Red kWh':>8s} {'Std kWh':>8s} {'VAT (red €)':>12s} {'VAT (std €)':>13s} {'Total €':>9s}"
            )
            print(header)
            for period in sorted(pb.keys()):
                entry = pb[period]
                print(
                    f"{period:10s} "
                    f"{entry['kwh']:8.1f} "
                    f"{entry['avg_unit_price_eur_kwh']:10.4f} "
                    f"{entry['base_eur']:8.2f} "
                    f"{entry['reduced_kwh']:8.1f} "
                    f"{entry['standard_kwh']:8.1f} "
                    f"{entry['vat_reduced_eur']:12.2f} "
                    f"{entry['vat_standard_eur']:13.2f} "
                    f"{entry['total_with_vat_eur']:9.2f}"
                )

        if isinstance(pb_all, dict):
            if 'without' in pb_all and pb_all['without']:
                _print_period_table("Bi-hourly / Period Consumption (house, WITHOUT battery) — exact VAT", pb_all['without'])
            if 'with' in pb_all and pb_all['with']:
                _print_period_table("Bi-hourly / Period Consumption (WITH battery) — exact VAT", pb_all['with'])

    # TOTAL SUMMARY
    print("\n" + "="*60)
    print("TOTAL INVOICE")
    print("-" * 60)
    print(f"Without Battery:               €{invoice.total_without_battery:8.2f}")
    print(f"With Battery:                  €{invoice.total_with_battery:8.2f}")
    print(f"{'TOTAL SAVINGS:':30s} €{invoice.savings:8.2f}")
    print(f"Savings percentage:            {(invoice.savings/invoice.total_without_battery*100):.1f}%")
    print("="*60)



# ========== UPDATED SECTION 4: Updated plot function for modular invoice ==========
def plot_modular_cost_breakdown(invoice, config, start_date, end_date, n_days):
    """
    Create visualization for modular invoice structure.
    Shows the 3 main components separately.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    plots_dir = config['output']['plots_dir']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Modular Energy Cost Breakdown\n{start_date} to {end_date} ({n_days} days)', 
                 fontsize=16, fontweight='bold')
    
    # Colors for components
    electricity_colors = {
        'OMIE Market (adjusted)': '#1f77b4',
        'Network Access (K2)': '#ff7f0e',
        'Time-of-Use Tariff': '#2ca02c',
        'Energy VAT': '#d62728'
    }
    
    power_colors = {
        'K3 Daily Fee': '#9467bd',
        'Contracted Power': '#8c564b'
    }
    
    taxes_colors = {
        'IEC Tax': '#e377c2',
        'IEC VAT': '#7f7f7f',
        'CAV (incl. VAT)': '#bcbd22',
        'DGEG (incl. VAT)': '#17becf'
    }
    
    # 1. ELECTRICITY TERM COMPARISON
    ax1 = axes[0, 0]
    categories = list(invoice.electricity_term_without.keys())
    without_vals = [invoice.electricity_term_without[cat] for cat in categories]
    with_vals = [invoice.electricity_term_with[cat] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, without_vals, width, label='Without Battery', 
                    color='coral', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, with_vals, width, label='With Battery',
                    color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Cost (EUR)')
    ax1.set_title('Electricity Term (Variable Costs)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'€{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. POWER TERM (Fixed)
    ax2 = axes[0, 1]
    power_components = list(invoice.power_term.keys())
    power_values = [invoice.power_term[comp] for comp in power_components]
    colors = [power_colors.get(comp, '#666666') for comp in power_components]
    
    bars = ax2.bar(power_components, power_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, power_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'€{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Cost (EUR)')
    ax2.set_title('Power Term (Fixed Costs)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. TAXES
    ax3 = axes[1, 0]
    tax_components = list(invoice.taxes.keys())
    tax_values = [invoice.taxes[comp] for comp in tax_components]
    colors = [taxes_colors.get(comp, '#666666') for comp in tax_components]
    
    bars = ax3.bar(tax_components, tax_values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, tax_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'€{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_ylabel('Cost (EUR)')
    ax3.set_title('Taxes')
    ax3.set_xticklabels(tax_components, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. TOTAL COMPARISON (Pie chart or stacked bar)
    ax4 = axes[1, 1]
    
    # Calculate totals for each category
    electricity_without = sum(invoice.electricity_term_without.values())
    electricity_with = sum(invoice.electricity_term_with.values())
    power_total = sum(invoice.power_term.values())
    taxes_total = sum(invoice.taxes.values())
    
    # Stacked bar comparison
    categories = ['Without Battery', 'With Battery']
    electricity_vals = [electricity_without, electricity_with]
    power_vals = [power_total, power_total]  # Same for both
    taxes_vals = [taxes_total, taxes_total]  # Approximately same
    
    x = np.arange(len(categories))
    width = 0.5
    
    p1 = ax4.bar(x, electricity_vals, width, label='Electricity', color='#ff9999')
    p2 = ax4.bar(x, power_vals, width, bottom=electricity_vals, label='Power', color='#66b3ff')
    p3 = ax4.bar(x, taxes_vals, width, bottom=np.array(electricity_vals) + np.array(power_vals),
                 label='Taxes', color='#99ff99')
    
    ax4.set_ylabel('Cost (EUR)')
    ax4.set_title('Total Cost Breakdown')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Add total values
    totals = [invoice.total_without_battery, invoice.total_with_battery]
    for i, total in enumerate(totals):
        ax4.text(i, total + 2, f'€{total:.2f}', ha='center', fontweight='bold')
    
    # Add savings annotation
    savings = invoice.total_without_battery - invoice.total_with_battery
    ax4.annotate(f'Savings: €{savings:.2f}', xy=(0.5, max(totals)/2),
                xytext=(0.5, max(totals)*0.7), ha='center', fontsize=12,
                fontweight='bold', color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/modular_cost_breakdown.png", dpi=150, bbox_inches='tight')
    
    return fig



def plot_results(results_df: pd.DataFrame, price_df: pd.DataFrame, config: dict):
    """Generate visualization plots. FIXED: Returns figure without showing immediately."""
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
    
    # Adjust layout and save
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/simulation_overview.png", dpi=150, bbox_inches='tight')
    
    # Return figure for later display
    return fig


def main():
    """Main simulation function with real consumption support."""
    print("="*60)
    print("ESS ENERGY ARBITRAGE SIMULATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    create_output_dirs(config)

    # Instantiate tariff processor and fixed costs
    tariff_processor = create_tariff_processor(config['tariff'])
    fixed_costs = calculate_daily_fixed_costs(config, tariff_processor)
    
    # Parse dates
    start_date = datetime.strptime(config['period']['start_date'], "%Y-%m-%d")
    if 'end_date' in config['period']:
        end_date = datetime.strptime(config['period']['end_date'], "%Y-%m-%d")
    else:
        end_date = start_date + timedelta(days=config['period']['num_days'] - 1)
    
    print(f"\nSimulation period: {start_date.date()} to {end_date.date()}")
    
    # Check consumption data type
    real_consumption = config['consumption'].get('real_consumption', False)
    consumption_type = "REAL DATA" if real_consumption else "E-REDES PROFILE"
    print(f"Consumption data type: {consumption_type}")
    
    if real_consumption:
        real_file = config['consumption'].get('real_consumption_file')
        print(f"Real consumption file: {real_file}")
    
    # Print tariff summary
    print_tariff_summary(config, fixed_costs, tariff_processor)
    
    # Load and prepare data
    print("\nPreparing simulation data...")
    
    # Prepare arguments for data loading
    prepare_args = {
        'consumption_profile_path': config['consumption']['profile_file'],
        'annual_consumption_kwh': config['consumption']['annual_consumption_kwh'],
        'start_date': start_date,
        'end_date': end_date,
        'profile_column': config['consumption']['profile_column'],
        'consumption_model': config['consumption'].get('consumption_model', False),
        'real_consumption': real_consumption
    }
    
    # Add real consumption file if needed
    if real_consumption:
        prepare_args['real_consumption_file'] = config['consumption']['real_consumption_file']
    
    consumption_df, prices_df = prepare_simulation_data(**prepare_args)

    print(consumption_df)
    print((prices_df))

    # Rename price column for clarity
    if 'price_eur_per_kwh' in prices_df.columns:
        prices_df = prices_df.rename(columns={'price_eur_per_kwh': 'price_omie_eur_kwh'})

    # Apply tariff to compute final prices
    prices_df = tariff_processor.apply(prices_df)

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

    print(f"\nRunning simulation with {consumption_type}...")
    results_df = simulator.run(
        consumption_df,
        prices_df,
        start_date,
        end_date,
    )
    total_consumed = results_df['house_consumption_kwh'].sum()
    print(f"\nTotal house consumption over simulation: {total_consumed:.2f} kWh")

    # Billing
    n_days = (end_date - start_date).days + 1


    print("\nRunning billing calculations...")
    ledger, invoice, metrics = run_billing(results_df, prices_df, config, fixed_costs)
    
    # Print modular invoice
    print_modular_invoice(invoice)
    
    # Save results
    if config['output']['save_timeline']:
        # Add ledger columns to results for complete view
        _candidate_cols = [
            'omie_adjusted_without_eur', 'k2_without_eur', 'tariff_without_eur',
            'energy_vat_without_eur', 'iec_without_eur',
            'omie_adjusted_with_eur', 'k2_with_eur', 'tariff_with_eur',
            'energy_vat_with_eur', 'iec_with_eur',
            'electricity_base_without_eur', 'electricity_base_with_eur'
        ]
        _export_cols = [c for c in _candidate_cols if c in ledger.columns]
        results_with_ledger = results_df.join(ledger[_export_cols], how='left')
        save_results(results_with_ledger, config['output']['timeline_file'])
    
    if config['output']['save_summary']:
        summary = {
            **metrics,
            'fixed_costs_breakdown': fixed_costs,
            'consumption_data_type': consumption_type,
            'real_consumption_used': real_consumption,
            # Add modular breakdown to summary
            'modular_invoice': {
                'electricity_term': {
                    'without_battery': invoice.electricity_term_without,
                    'with_battery': invoice.electricity_term_with
                },
                'power_term': invoice.power_term,
                'taxes': invoice.taxes
            }
        }
        with open(config['output']['summary_file'], 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {config['output']['summary_file']}")
    
    # Generate plots
    if config['output']['generate_plots']:
        print("\nGenerating plots...")
        
        # Original plots
        fig_results = plot_results(results_df, prices_df, config)
        
        # New modular breakdown plot
        n_days = (end_date - start_date).days + 1
        fig_modular = plot_modular_cost_breakdown(
            invoice, config, start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'), n_days
        )
        
        # Show all plots
        plt.ion()
        fig_results.show()
        fig_modular.show()
        
        print("\nAll plots displayed!")
        input("Press Enter to continue...")


    return results_df, metrics

if __name__ == "__main__":
    results, metrics = main()