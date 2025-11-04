#!/usr/bin/env python3
"""
Main simulation script for Energy Management System
"""
import yaml
import os
from datetime import datetime
from pathlib import Path

from src.components.battery import Battery
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import SimpleTariff, BiHorariaTariff
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.system import EnergyManagementSystem
from src.visualization import plot_system_behavior, plot_daily_analysis, print_invoice, compare_controllers


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_battery(config: dict) -> Battery:
    """Create battery from configuration"""
    battery_config = config['battery']
    return Battery(
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


def create_solar(config: dict) -> SolarPanel:
    """Create solar panel from configuration"""
    solar_config = config['solar']
    data_file = solar_config.get('data_file')

    if data_file:
        # Make path relative to script location
        script_dir = Path(__file__).parent
        data_file = str(script_dir / data_file)

    return SolarPanel(
        capacity_kw=solar_config['capacity_kw'],
        data_file=data_file
    )


def create_house(config: dict) -> House:
    """Create house from configuration"""
    house_config = config['house']
    data_file = house_config.get('data_file')

    if data_file:
        # Make path relative to script location
        script_dir = Path(__file__).parent
        data_file = str(script_dir / data_file)

    return House(data_file=data_file)


def create_tariff(config: dict):
    """Create tariff from configuration"""
    tariff_config = config['tariff']
    tariff_type = tariff_config['type']

    if tariff_type == 'simple':
        return SimpleTariff(price=tariff_config['simple']['price'])

    elif tariff_type == 'bihoraria':
        bi_config = tariff_config['bihoraria']
        tariff = BiHorariaTariff(
            peak_price=bi_config['peak_price'],
            off_peak_price=bi_config['off_peak_price']
        )
        # Override peak hours if specified
        if 'peak_hours_weekday' in bi_config:
            tariff.peak_hours_weekday = bi_config['peak_hours_weekday']
        if 'peak_hours_weekend' in bi_config:
            tariff.peak_hours_weekend = bi_config['peak_hours_weekend']
        return tariff

    else:
        raise ValueError(f"Unknown tariff type: {tariff_type}")


def create_controller(config: dict, controller_type: str = None):
    """Create controller from configuration"""
    controller_config = config['controller']

    if controller_type is None:
        controller_type = controller_config['type']

    if controller_type == 'rule_based':
        rb_config = controller_config['rule_based']
        return RuleBasedController(
            high_price_threshold=rb_config['high_price_threshold'],
            low_soc_threshold=rb_config['low_soc_threshold'],
            high_soc_threshold=rb_config['high_soc_threshold']
        )

    elif controller_type == 'mpc':
        mpc_config = controller_config['mpc']
        grid_config = config.get('grid', {})
        return MPCController(
            horizon_steps=mpc_config['horizon_steps'],
            dt_minutes=config['simulation']['timestep_minutes'],
            export_price=mpc_config.get('export_price', 0.05),  # Default 0.05 €/kWh
            contracted_power_kw=grid_config.get('contracted_power_kw', None),  # No limit if not specified
            use_perfect_forecast=mpc_config.get('use_perfect_forecast', False)  # Perfect forecast mode (for testing)
        )

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def main():
    """Main simulation function"""
    print("=" * 60)
    print("ENERGY MANAGEMENT SYSTEM SIMULATION".center(60))
    print("=" * 60)

    # Load configuration
    config = load_config("config.yaml")

    # Parse simulation settings
    sim_config = config['simulation']
    start_time = datetime.strptime(sim_config['start_date'], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(sim_config['end_date'], "%Y-%m-%d %H:%M:%S")
    dt_minutes = sim_config['timestep_minutes']

    # Create output directories
    output_config = config['output']
    if output_config['save_plots']:
        os.makedirs(output_config['plots_directory'], exist_ok=True)
    if output_config['save_results']:
        os.makedirs(output_config['results_directory'], exist_ok=True)

    # Check if comparison mode is enabled
    comparison_config = config.get('comparison', {})
    comparison_enabled = comparison_config.get('enabled', False)

    if comparison_enabled:
        print("\nCOMPARISON MODE ENABLED")
        print("=" * 60)

        controllers_to_compare = comparison_config.get('controllers', ['rule_based', 'mpc'])
        results_dict = {}

        for controller_type in controllers_to_compare:
            print(f"\nRunning simulation with {controller_type} controller...")

            # Create components
            battery = create_battery(config)
            solar = create_solar(config)
            house = create_house(config)
            tariff = create_tariff(config)
            controller = create_controller(config, controller_type)

            # Create system
            system = EnergyManagementSystem(battery, solar, house, tariff, controller)

            # Run simulation
            df = system.simulate(start_time, end_time, dt_minutes)

            # Save results
            results_dict[controller_type] = df

            # Get summary and savings
            summary = system.get_summary()
            savings = system.get_savings(df)

            # Print invoice
            print_invoice(summary, savings)

            # Save results if requested
            if output_config['save_results']:
                results_file = f"{output_config['results_directory']}/results_{controller_type}.csv"
                df.to_csv(results_file)
                print(f"Results saved to {results_file}")

        # Create comparison plots
        if output_config['save_plots']:
            comparison_plot_file = f"{output_config['plots_directory']}/comparison.png"
            compare_controllers(results_dict, comparison_plot_file)

    else:
        # Single controller simulation
        print("\nSINGLE CONTROLLER MODE")
        print("=" * 60)

        # Create components
        battery = create_battery(config)
        solar = create_solar(config)
        house = create_house(config)
        tariff = create_tariff(config)
        controller = create_controller(config)

        # Create system
        system = EnergyManagementSystem(battery, solar, house, tariff, controller)

        # Run simulation
        df = system.simulate(start_time, end_time, dt_minutes)

        # Get summary and savings
        summary = system.get_summary()
        savings = system.get_savings(df)

        # Print results
        print_invoice(summary, savings)

        # Create plots
        if output_config['save_plots']:
            plot_file = f"{output_config['plots_directory']}/system_behavior.png"
            plot_system_behavior(df, plot_file)

            daily_plot_file = f"{output_config['plots_directory']}/daily_analysis.png"
            plot_daily_analysis(df, daily_plot_file)

        # Save results
        if output_config['save_results']:
            results_file = f"{output_config['results_directory']}/results.csv"
            df.to_csv(results_file)
            print(f"Results saved to {results_file}")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE".center(60))
    print("=" * 60)


if __name__ == "__main__":
    main()
