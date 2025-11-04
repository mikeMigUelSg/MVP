"""
Visualization and reporting utilities
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict


def plot_system_behavior(df: pd.DataFrame, save_path: str = None):
    """
    Create comprehensive plots showing system behavior

    Args:
        df: Simulation results DataFrame
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Power flows
    ax = axes[0]
    ax.plot(df.index, df['solar_power'], label='Solar Production', color='orange', linewidth=1.5)
    ax.plot(df.index, df['load_power'], label='Load', color='red', linewidth=1.5)
    ax.plot(df.index, df['battery_power'], label='Battery Power', color='blue', linewidth=1.5)
    ax.plot(df.index, df['grid_power'], label='Grid Power', color='green', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Power (kW)')
    ax.set_title('Power Flows')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Battery State of Charge
    ax = axes[1]
    ax.plot(df.index, df['battery_soc'] * 100, color='purple', linewidth=2)
    ax.set_ylabel('SOC (%)')
    ax.set_title('Battery State of Charge')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Plot 3: Grid import/export
    ax = axes[2]
    ax.fill_between(df.index, 0, df['grid_import'], label='Grid Import', color='red', alpha=0.5)
    ax.fill_between(df.index, 0, -df['grid_export'], label='Grid Export', color='green', alpha=0.5)
    ax.set_ylabel('Power (kW)')
    ax.set_title('Grid Import/Export')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Electricity price and costs
    ax = axes[3]
    ax_twin = ax.twinx()

    ax.plot(df.index, df['electricity_price'], label='Price', color='black', linewidth=1.5)
    ax.set_ylabel('Price (€/kWh)', color='black')
    ax.tick_params(axis='y', labelcolor='black')

    ax_twin.fill_between(df.index, 0, df['cost'], label='Cost', color='blue', alpha=0.3)
    ax_twin.set_ylabel('Cost (€)', color='blue')
    ax_twin.tick_params(axis='y', labelcolor='blue')

    ax.set_xlabel('Time')
    ax.set_title('Electricity Price and Costs')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_daily_analysis(df: pd.DataFrame, save_path: str = None):
    """
    Create daily analysis plots

    Args:
        df: Simulation results DataFrame
        save_path: Optional path to save figure
    """
    # Aggregate by hour of day
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    df_copy['hour'] = df_copy.index.hour

    hourly_avg = df_copy.groupby('hour').mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average power by hour
    ax = axes[0, 0]
    ax.plot(hourly_avg.index, hourly_avg['solar_power'], label='Solar', marker='o')
    ax.plot(hourly_avg.index, hourly_avg['load_power'], label='Load', marker='o')
    ax.plot(hourly_avg.index, hourly_avg['battery_power'], label='Battery', marker='o')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Power (kW)')
    ax.set_title('Average Power by Hour')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Average SOC by hour
    ax = axes[0, 1]
    ax.plot(hourly_avg.index, hourly_avg['battery_soc'] * 100, marker='o', color='purple')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average SOC (%)')
    ax.set_title('Average Battery SOC by Hour')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Average grid interaction by hour
    ax = axes[1, 0]
    ax.bar(hourly_avg.index, hourly_avg['grid_import'], label='Import', color='red', alpha=0.6)
    ax.bar(hourly_avg.index, -hourly_avg['grid_export'], label='Export', color='green', alpha=0.6)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Power (kW)')
    ax.set_title('Average Grid Import/Export by Hour')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Price and cost by hour
    ax = axes[1, 1]
    ax_twin = ax.twinx()

    ax.plot(hourly_avg.index, hourly_avg['electricity_price'], marker='o', color='black', label='Price')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price (€/kWh)', color='black')
    ax.tick_params(axis='y', labelcolor='black')

    ax_twin.bar(hourly_avg.index, hourly_avg['cost'], color='blue', alpha=0.3, label='Cost')
    ax_twin.set_ylabel('Average Cost (€)', color='blue')
    ax_twin.tick_params(axis='y', labelcolor='blue')

    ax.set_title('Price and Cost by Hour')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def print_invoice(summary: Dict, savings: Dict):
    """
    Print formatted invoice/bill with detailed comparisons

    Args:
        summary: System summary statistics
        savings: Savings information with baseline comparisons
    """
    print("\n" + "=" * 80)
    print("ENERGY MANAGEMENT SYSTEM - DETAILED COMPARISON".center(80))
    print("=" * 80)

    print(f"\nController: {summary['controller']}")
    print("-" * 80)

    # Three-way comparison table
    print("\nSCENARIO COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<30} {'PV+Battery':<15} {'PV Only':<15} {'Grid Only':<15}")
    print(f"{'(Current)':<30} {'(Baseline)':<15} {'(No PV)':<15}")
    print("-" * 80)

    # Grid Import
    print(f"{'Grid Import (kWh)':<30} {savings['system_import_kwh']:>14.2f} "
          f"{savings['baseline_pv_import_kwh']:>14.2f} "
          f"{savings['baseline_no_pv_import_kwh']:>14.2f}")

    # Grid Export
    print(f"{'Grid Export (kWh)':<30} {savings['system_export_kwh']:>14.2f} "
          f"{savings['baseline_pv_export_kwh']:>14.2f} "
          f"{'0.00':>14}")

    # Net Import (Import - Export)
    net_import_sys = savings['system_import_kwh'] - savings['system_export_kwh']
    net_import_pv = savings['baseline_pv_import_kwh'] - savings['baseline_pv_export_kwh']
    net_import_no_pv = savings['baseline_no_pv_import_kwh']

    print(f"{'Net Import (kWh)':<30} {net_import_sys:>14.2f} "
          f"{net_import_pv:>14.2f} "
          f"{net_import_no_pv:>14.2f}")

    print("-" * 80)

    # Costs
    print(f"{'Total Cost (€)':<30} {savings['system_cost']:>14.2f} "
          f"{savings['baseline_pv_cost']:>14.2f} "
          f"{savings['baseline_no_pv_cost']:>14.2f}")

    print("\n" + "=" * 80)

    # Savings vs PV-only
    print("\nSAVINGS (Battery Benefit):")
    print(f"  vs. PV Only (no battery):  {savings['savings_vs_pv']:>10.2f} € ({savings['savings_pct_vs_pv']:>6.1f}%)")

    # Savings vs no PV
    print("\nSAVINGS (PV + Battery Benefit):")
    print(f"  vs. Grid Only (no PV):     {savings['savings_vs_no_pv']:>10.2f} € ({savings['savings_pct_vs_no_pv']:>6.1f}%)")

    print("\n" + "-" * 80)

    # Solar and Battery performance
    print("\nSOLAR & BATTERY PERFORMANCE:")
    print(f"  Solar Production:        {summary['total_solar_production_kwh']:>10.2f} kWh")
    print(f"  House Consumption:       {summary['total_consumption_kwh']:>10.2f} kWh")
    print(f"  Self-Consumption Rate:   {summary['self_consumption_rate']*100:>10.1f} %")
    print(f"  Self-Sufficiency Rate:   {summary['self_sufficiency_rate']*100:>10.1f} %")
    print(f"  Battery Cycles:          {summary['battery_cycles']:>10.2f}")
    print(f"  Degradation Cost:        {summary['battery_degradation_cost']:>10.2f} €")

    print("\n" + "=" * 80 + "\n")


def compare_controllers(results_dict: Dict[str, pd.DataFrame], save_path: str = None):
    """
    Compare results from different controllers

    Args:
        results_dict: Dictionary mapping controller name to results DataFrame
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot cumulative costs
    ax = axes[0, 0]
    for name, df in results_dict.items():
        cumulative_cost = df['cost'].cumsum()
        ax.plot(df.index, cumulative_cost, label=name, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Cost (€)')
    ax.set_title('Cumulative Cost Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot average SOC
    ax = axes[0, 1]
    for name, df in results_dict.items():
        ax.plot(df.index, df['battery_soc'] * 100, label=name, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('SOC (%)')
    ax.set_title('Battery SOC Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot grid import comparison
    ax = axes[1, 0]
    hourly_import = {}
    for name, df in results_dict.items():
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['hour'] = df_copy.index.hour
        hourly_import[name] = df_copy.groupby('hour')['grid_import'].mean()

    for name, data in hourly_import.items():
        ax.plot(data.index, data.values, marker='o', label=name)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Grid Import (kW)')
    ax.set_title('Grid Import by Hour')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary bar chart
    ax = axes[1, 1]
    names = list(results_dict.keys())
    total_costs = [df['cost'].sum() for df in results_dict.values()]

    ax.bar(names, total_costs, color=['blue', 'green', 'red'][:len(names)])
    ax.set_ylabel('Total Cost (€)')
    ax.set_title('Total Cost Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate labels if needed
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()
