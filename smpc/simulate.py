"""
Energy Management System Simulator

This script runs a complete simulation of the energy management system
with support for different controller types: Rule-Based, MPC, and SDP.

Configuration is loaded from config.yaml
"""

import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import components
from src.components.battery import Battery
from src.components.solar import SolarPanel
from src.components.house import House
from src.components.tariff import SimpleTariff, BiHorariaTariff

# Import controllers
from src.controllers.rule_based import RuleBasedController
from src.controllers.mpc_controller import MPCController
from src.controllers.sdp_controller import SDPController, SDPParams, PlantModel, ResidualModel

# Import forecasters for SDP
from src.forecasters.profile_persistence_forecaster import ProfilePersistenceForecaster

# Import system and visualization
from src.system import EnergyManagementSystem
from src.visualization import (
    plot_simulation_results,
    plot_daily_analysis,
    print_summary
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_battery(config: dict) -> Battery:
    """Create battery component from config."""
    bat_config = config['battery']
    return Battery(
        capacity_kwh=bat_config['capacity_kwh'],
        max_power_kw=bat_config['max_power_kw'],
        max_charge_current_kw=bat_config['max_charge_current_kw'],
        max_discharge_current_kw=bat_config['max_discharge_current_kw'],
        efficiency_charge=bat_config['efficiency_charge'],
        efficiency_discharge=bat_config['efficiency_discharge'],
        initial_soc=bat_config['initial_soc'],
        min_soc=bat_config['min_soc'],
        max_soc=bat_config['max_soc'],
        degradation_cost_per_kwh=bat_config['degradation_cost_per_kwh']
    )


def create_solar(config: dict) -> SolarPanel:
    """Create solar panel component from config."""
    solar_config = config['solar']
    return SolarPanel(
        capacity_kw=solar_config['capacity_kw'],
        data_file=solar_config['data_file']
    )


def create_house(config: dict) -> House:
    """Create house component from config."""
    house_config = config['house']
    return House(
        data_file=house_config['data_file']
    )


def create_tariff(config: dict):
    """Create tariff component from config."""
    tariff_config = config['tariff']
    tariff_type = tariff_config['type']

    if tariff_type == 'simple':
        return SimpleTariff(
            price=tariff_config['simple']['price']
        )
    elif tariff_type == 'bihoraria':
        tariff = BiHorariaTariff(
            peak_price=tariff_config['bihoraria']['peak_price'],
            off_peak_price=tariff_config['bihoraria']['off_peak_price']
        )
        # Set custom peak hours if provided
        tariff.peak_hours_weekday = tariff_config['bihoraria']['peak_hours_weekday']
        tariff.peak_hours_weekend = tariff_config['bihoraria']['peak_hours_weekend']
        return tariff
    else:
        raise ValueError(f"Unknown tariff type: {tariff_type}")


def load_historical_data(solar: SolarPanel, house: House,
                         start_date: datetime, days: int = 30) -> pd.DataFrame:
    """
    Load and merge historical data from solar and house components.

    Since the components store data indexed by (month, day, hour, minute) without year,
    we create a synthetic historical dataset by querying the components.

    Args:
        solar: SolarPanel object
        house: House object
        start_date: Starting date for historical data
        days: Number of days of historical data to create

    Returns:
        DataFrame with columns: ['timestamp', 'P_pv', 'P_load']
    """
    from datetime import timedelta

    # Create timestamps going back from start_date
    historical_start = start_date - timedelta(days=days)
    timestamps = pd.date_range(
        start=historical_start,
        end=start_date,
        freq='15min'
    )

    # Query both components for each timestamp
    pv_values = []
    load_values = []

    for ts in timestamps:
        pv_values.append(solar.get_production(ts))
        load_values.append(house.get_consumption(ts))

    # Create DataFrame
    historical_data = pd.DataFrame({
        'timestamp': timestamps,
        'P_pv': pv_values,
        'P_load': load_values
    })

    return historical_data


def create_sdp_controller(config: dict, battery: Battery,
                          solar: SolarPanel, house: House,
                          tariff) -> SDPController:
    """Create SDP controller from config."""
    sdp_config = config['controller']['sdp']
    sim_config = config['simulation']

    print("\n" + "="*70)
    print("CREATING SDP CONTROLLER")
    print("="*70)

    # 1. Create SDP parameters
    print("\n1. Creating SDP parameters...")
    params = SDPParams(
        N=sdp_config['horizon_steps'],
        dt_minutes=sim_config['timestep_minutes'],
        n_x=sdp_config['n_soc_points'],
        n_R=sdp_config['n_residual_points'],
        t_half_minutes=sdp_config['half_life_minutes'],
        sigma_R=sdp_config['sigma_residual_kw'],
        policy_update_hours=sdp_config['policy_update_hours'],
        soc_target=sdp_config['soc_target'],
        terminal_cost_weight=sdp_config['terminal_cost_weight']
    )
    print(f"   Horizon: {params.N} steps = {params.N * params.dt_minutes / 60:.1f}h")
    print(f"   Discretization: {params.n_x} SOC points, {params.n_R} residual points")
    print(f"   Half-life: {params.t_half_minutes} min, σ: {params.sigma_R} kW")

    # 2. Create plant model
    print("\n2. Creating plant model...")
    plant = PlantModel(
        C_bat=config['battery']['capacity_kwh'],
        P_nom=config['battery']['max_power_kw'],
        P_lim=sdp_config['injection_limit_kw'],
        eta_charge=config['battery']['efficiency_charge'],
        eta_discharge=config['battery']['efficiency_discharge'],
        dt_minutes=sim_config['timestep_minutes'],
        soc_min=config['battery']['min_soc'],
        soc_max=config['battery']['max_soc']
    )
    print(f"   Battery: {plant.C_bat} kWh, {plant.P_nom} kW")
    print(f"   Injection limit: {plant.P_lim} kW")

    # 3. Create residual model
    print("\n3. Creating residual model...")
    residual_model = ResidualModel(
        t_half_minutes=params.t_half_minutes,
        dt_minutes=params.dt_minutes,
        sigma=params.sigma_R
    )
    print(f"   κ (persistence coef): {residual_model.kappa:.4f}")
    print(f"   Stationary variance: {residual_model.stationary_variance():.4f} kW²")

    # 4. Load historical data and create forecasters
    print("\n4. Creating forecasters...")

    # Get simulation start date to create historical data before it
    sim_start = datetime.strptime(config['simulation']['start_date'], '%Y-%m-%d %H:%M:%S')

    # Create historical data (30 days before simulation start for forecasters)
    print("   Loading historical data...")
    historical_data = load_historical_data(solar, house, sim_start, days=30)
    print(f"   Historical data created: {len(historical_data)} records")
    print(f"   Date range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")

    # Create PV forecaster
    pv_forecaster = ProfilePersistenceForecaster(
        n_weeks=sdp_config['forecaster']['n_weeks'],
        dt_minutes=sim_config['timestep_minutes']
    )
    pv_data = historical_data[['timestamp', 'P_pv']].copy()
    pv_forecaster.set_data(pv_data, value_column='P_pv')
    print(f"   PV forecaster created (profile persistence, {sdp_config['forecaster']['n_weeks']} weeks)")

    # Create load forecaster
    load_forecaster = ProfilePersistenceForecaster(
        n_weeks=sdp_config['forecaster']['n_weeks'],
        dt_minutes=sim_config['timestep_minutes']
    )
    load_data = historical_data[['timestamp', 'P_load']].copy()
    load_forecaster.set_data(load_data, value_column='P_load')
    print(f"   Load forecaster created (profile persistence, {sdp_config['forecaster']['n_weeks']} weeks)")

    # 5. Optional: Calibration
    if sdp_config['calibration']['enabled']:
        print("\n5. Calibrating SDP parameters...")
        print("   (Calibration not yet implemented - using config values)")
        # TODO: Implement calibration using src.controllers.sdp_calibration

    # 6. Determine buy/sell prices
    print("\n6. Setting tariff prices...")
    if config['tariff']['type'] == 'bihoraria':
        # Use average of peak and off-peak as buy price
        buy_price = (config['tariff']['bihoraria']['peak_price'] +
                    config['tariff']['bihoraria']['off_peak_price']) / 2
        print(f"   Buy price (average): {buy_price:.3f} €/kWh")
    else:
        buy_price = config['tariff']['simple']['price']
        print(f"   Buy price: {buy_price:.3f} €/kWh")

    sell_price = sdp_config['sell_price']
    print(f"   Sell price: {sell_price:.3f} €/kWh")

    # 7. Create SDP controller
    print("\n7. Creating SDP controller...")
    controller = SDPController(
        params=params,
        plant=plant,
        residual_model=residual_model,
        c_s=buy_price,
        c_f=sell_price
    )

    # Store forecasters in controller for use during simulation
    controller.pv_forecaster = pv_forecaster
    controller.load_forecaster = load_forecaster

    print("   SDP controller created successfully!")
    print("="*70 + "\n")

    return controller


def create_controller(config: dict, battery: Battery,
                     solar: SolarPanel, house: House,
                     tariff):
    """Create controller based on config type."""
    controller_type = config['controller']['type']

    if controller_type == 'rule_based':
        print("\nCreating Rule-Based controller...")
        rb_config = config['controller']['rule_based']
        return RuleBasedController(
            high_price_threshold=rb_config['high_price_threshold'],
            low_soc_threshold=rb_config['low_soc_threshold'],
            high_soc_threshold=rb_config['high_soc_threshold']
        )

    elif controller_type == 'mpc':
        print("\nCreating MPC controller...")
        mpc_config = config['controller']['mpc']
        return MPCController(
            horizon_steps=mpc_config['horizon_steps'],
            export_price=mpc_config['export_price']
        )

    elif controller_type == 'sdp':
        return create_sdp_controller(config, battery, solar, house, tariff)

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def run_simulation(config: dict):
    """Run the complete simulation."""

    print("\n" + "="*70)
    print("ENERGY MANAGEMENT SYSTEM SIMULATION")
    print("="*70)

    # Parse simulation dates
    start_date = datetime.strptime(config['simulation']['start_date'], '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(config['simulation']['end_date'], '%Y-%m-%d %H:%M:%S')
    timestep = config['simulation']['timestep_minutes']

    print(f"\nSimulation period: {start_date} to {end_date}")
    print(f"Time step: {timestep} minutes")
    print(f"Controller type: {config['controller']['type'].upper()}")

    # Create components
    print("\n" + "-"*70)
    print("Creating system components...")
    print("-"*70)

    battery = create_battery(config)
    print(f"✓ Battery: {battery.capacity_kwh} kWh, {battery.max_power_kw} kW")

    solar = create_solar(config)
    print(f"✓ Solar: {solar.capacity_kw} kW")

    house = create_house(config)
    print(f"✓ House: Load data loaded")

    tariff = create_tariff(config)
    print(f"✓ Tariff: {tariff.name}")

    # Create controller
    controller = create_controller(config, battery, solar, house, tariff)
    print(f"✓ Controller: {controller.name}")

    # Create system
    print("\n" + "-"*70)
    print("Initializing energy management system...")
    print("-"*70)

    system = EnergyManagementSystem(
        battery=battery,
        solar=solar,
        house=house,
        tariff=tariff,
        controller=controller
    )

    # Run simulation
    print("\n" + "-"*70)
    print("Running simulation...")
    print("-"*70 + "\n")

    results_df = system.simulate(
        start_time=start_date,
        end_time=end_date,
        dt_minutes=timestep
    )

    # Get summary
    print("\n" + "-"*70)
    print("SIMULATION RESULTS")
    print("-"*70 + "\n")

    summary = system.get_summary()
    print_summary(summary)

    # Calculate additional energy metrics
    dt_hours = timestep / 60.0

    total_solar_produced = results_df['solar_power'].sum() * dt_hours
    total_load_consumed = results_df['load_power'].sum() * dt_hours
    total_battery_charged = results_df[results_df['battery_power'] > 0]['battery_power'].sum() * dt_hours
    total_battery_discharged = -results_df[results_df['battery_power'] < 0]['battery_power'].sum() * dt_hours

    print("\n" + "-"*70)
    print("ENERGY BALANCE")
    print("-"*70)

    print(f"\nTotal Solar Produced:     {total_solar_produced:>10.2f} kWh")
    print(f"Total Load Consumed:      {total_load_consumed:>10.2f} kWh")
    print(f"Total Grid Import:        {summary['total_grid_import_kwh']:>10.2f} kWh")
    print(f"Total Grid Export:        {summary['total_grid_export_kwh']:>10.2f} kWh")
    print(f"Total Battery Charged:    {total_battery_charged:>10.2f} kWh")
    print(f"Total Battery Discharged: {total_battery_discharged:>10.2f} kWh")

    # Calculate ratios
    solar_to_load_ratio = (total_solar_produced / total_load_consumed * 100) if total_load_consumed > 0 else 0
    solar_used_locally = total_solar_produced - summary['total_grid_export_kwh']
    solar_utilization = (solar_used_locally / total_solar_produced * 100) if total_solar_produced > 0 else 0

    print(f"\nSolar/Load Ratio:         {solar_to_load_ratio:>10.1f} %")
    print(f"Solar Self-Consumption:   {solar_utilization:>10.1f} %")
    print(f"Load Self-Sufficiency:    {summary['self_sufficiency_rate']*100:>10.1f} %")

    # Battery efficiency
    battery_roundtrip_eff = (total_battery_discharged / total_battery_charged * 100) if total_battery_charged > 0 else 0
    print(f"Battery Round-Trip Eff:   {battery_roundtrip_eff:>10.1f} %")

    # Calculate savings
    savings = system.get_savings(results_df)

    print("\n" + "-"*70)
    print("COST COMPARISON")
    print("-"*70)

    print(f"\nSystem with battery: {savings['system_cost']:.2f} €")
    print(f"PV without battery:  {savings['baseline_pv_cost']:.2f} €")
    print(f"No PV, no battery:   {savings['baseline_no_pv_cost']:.2f} €")

    print(f"\nSavings vs PV only:  {savings['savings_vs_pv']:.2f} € ({savings['savings_pct_vs_pv']:.1f}%)")
    print(f"Savings vs no PV:    {savings['savings_vs_no_pv']:.2f} € ({savings['savings_pct_vs_no_pv']:.1f}%)")

    # Save results
    if config['output']['save_results']:
        output_dir = Path(config['output']['results_directory'])
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f'results_{controller.name.lower()}.csv'
        results_df.to_csv(output_file)
        print(f"\n✓ Results saved to: {output_file}")

        # Save summary with additional metrics
        summary_file = output_dir / f'summary_{controller.name.lower()}.yaml'

        # Combine all metrics
        complete_summary = {
            **summary,
            **savings,
            'energy_balance': {
                'total_solar_produced_kwh': float(total_solar_produced),
                'total_load_consumed_kwh': float(total_load_consumed),
                'total_battery_charged_kwh': float(total_battery_charged),
                'total_battery_discharged_kwh': float(total_battery_discharged),
                'solar_to_load_ratio_pct': float(solar_to_load_ratio),
                'solar_utilization_pct': float(solar_utilization),
                'battery_roundtrip_efficiency_pct': float(battery_roundtrip_eff)
            }
        }

        with open(summary_file, 'w') as f:
            yaml.dump(complete_summary, f, default_flow_style=False)
        print(f"✓ Summary saved to: {summary_file}")

    # Generate plots
    if config['output']['save_plots']:
        plots_dir = Path(config['output']['plots_directory'])
        plots_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating plots...")

        # Main simulation plot
        plot_file = plots_dir / f'simulation_{controller.name.lower()}.png'
        plot_simulation_results(results_df, controller.name, str(plot_file))
        print(f"✓ Simulation plot saved to: {plot_file}")

        # Daily analysis plot
        daily_file = plots_dir / f'daily_analysis_{controller.name.lower()}.png'
        plot_daily_analysis(results_df, controller.name, str(daily_file))
        print(f"✓ Daily analysis plot saved to: {daily_file}")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70 + "\n")

    return results_df, summary, savings


def main():
    """Main entry point."""
    # Load configuration
    config_file = "config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)

    # Run simulation
    results_df, summary, savings = run_simulation(config)

    return results_df, summary, savings


if __name__ == "__main__":
    results, summary, savings = main()
