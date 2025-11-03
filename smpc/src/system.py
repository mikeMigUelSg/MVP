"""
Energy Management System - integrates all components
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union

from .components.battery import Battery
from .components.solar import SolarPanel
from .components.house import House
from .components.tariff import Tariff
from .controllers.rule_based import RuleBasedController
from .controllers.mpc_controller import MPCController
from .controllers.sdp_controller import SDPController


class EnergyManagementSystem:
    """
    Complete energy management system with power balance

    Power balance equation:
    solar + battery_discharge + grid_import = load + battery_charge + grid_export
    """

    def __init__(self,
                 battery: Battery,
                 solar: SolarPanel,
                 house: House,
                 tariff: Tariff,
                 controller: Union[RuleBasedController, MPCController, SDPController]):
        """
        Args:
            battery: Battery component
            solar: Solar panel component
            house: House component
            tariff: Electricity tariff
            controller: Control strategy (rule-based or MPC)
        """
        self.battery = battery
        self.solar = solar
        self.house = house
        self.tariff = tariff
        self.controller = controller

        # Tracking variables
        self.history = {
            'timestamp': [],
            'solar_power': [],
            'load_power': [],
            'battery_power': [],
            'battery_soc': [],
            'grid_power': [],
            'electricity_price': [],
            'grid_import': [],
            'grid_export': [],
            'cost': [],
            'degradation_cost': []
        }

        self.total_cost = 0.0
        self.total_grid_import_kwh = 0.0
        self.total_grid_export_kwh = 0.0

    def reset(self):
        """Reset system to initial state"""
        self.battery.reset()
        self.solar.reset()
        self.house.reset()

        for key in self.history:
            self.history[key] = []

        self.total_cost = 0.0
        self.total_grid_import_kwh = 0.0
        self.total_grid_export_kwh = 0.0

    def step(self, timestamp: datetime, dt_minutes: int = 15) -> dict:
        """
        Execute one simulation time step with power balance

        Args:
            timestamp: Current timestamp
            dt_minutes: Time step duration in minutes

        Returns:
            Dictionary with step results
        """
        dt_hours = dt_minutes / 60.0

        # Get solar production and house load
        solar_result = self.solar.step(timestamp, dt_hours)
        house_result = self.house.step(timestamp, dt_hours)

        solar_power = solar_result['power_kw']
        load_power = house_result['power_kw']

        # Get electricity price
        price = self.tariff.get_price(timestamp)

        # Compute controller action
        if isinstance(self.controller, MPCController):
            battery_power_cmd = self.controller.compute_action(
                timestamp, solar_power, load_power,
                self.battery, self.tariff, self.solar, self.house
            )
        elif isinstance(self.controller, SDPController):
            # SDP controller needs forecasters
            pv_forecaster = getattr(self.controller, 'pv_forecaster', None)
            load_forecaster = getattr(self.controller, 'load_forecaster', None)
            battery_power_cmd = self.controller.compute_action(
                timestamp, solar_power, load_power,
                self.battery, self.tariff, self.solar, self.house,
                pv_forecaster, load_forecaster
            )
        else:
            battery_power_cmd = self.controller.compute_action(
                timestamp, solar_power, load_power,
                self.battery, self.tariff, dt_hours
            )

        # Execute battery action
        battery_result = self.battery.step(battery_power_cmd, dt_hours)
        battery_power = battery_result['power_kw']

        # Power balance: grid_power = load - solar + battery_power
        # When battery_power > 0 (charging), it's a load
        # When battery_power < 0 (discharging), it's a source
        # Positive grid_power = import from grid
        # Negative grid_power = export to grid
        grid_power = load_power - solar_power + battery_power

        # Split into import and export
        grid_import = max(0.0, grid_power)
        grid_export = max(0.0, -grid_power)

        # Calculate costs
        # Import cost
        import_cost = grid_import * dt_hours * price

        # Export revenue (fixed price, e.g., 0.05 €/kWh)
        export_price_value = getattr(self.controller, 'export_price', 0.05)
        export_revenue = grid_export * dt_hours * export_price_value

        # Net cost for this time step
        step_cost = import_cost - export_revenue

        # Add degradation cost from battery
        degradation_cost = 0.0
        if battery_power > 0:
            # Charging
            degradation_cost = abs(battery_power) * dt_hours * self.battery.efficiency_charge * self.battery.degradation_cost_per_kwh
        elif battery_power < 0:
            # Discharging
            degradation_cost = abs(battery_power) * dt_hours / self.battery.efficiency_discharge * self.battery.degradation_cost_per_kwh

        total_step_cost = step_cost + degradation_cost

        # Update totals
        self.total_cost += total_step_cost
        self.total_grid_import_kwh += grid_import * dt_hours
        self.total_grid_export_kwh += grid_export * dt_hours

        # Record history
        self.history['timestamp'].append(timestamp)
        self.history['solar_power'].append(solar_power)
        self.history['load_power'].append(load_power)
        self.history['battery_power'].append(battery_power)
        self.history['battery_soc'].append(self.battery.get_soc())
        self.history['grid_power'].append(grid_power)
        self.history['electricity_price'].append(price)
        self.history['grid_import'].append(grid_import)
        self.history['grid_export'].append(grid_export)
        self.history['cost'].append(total_step_cost)
        self.history['degradation_cost'].append(degradation_cost)

        # Verify power balance
        power_balance_error = abs(
            solar_power + grid_import - load_power - grid_export - battery_power
        )

        return {
            'timestamp': timestamp,
            'solar_power': solar_power,
            'load_power': load_power,
            'battery_power': battery_power,
            'battery_soc': self.battery.get_soc(),
            'grid_power': grid_power,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'price': price,
            'step_cost': total_step_cost,
            'power_balance_error': power_balance_error
        }

    def simulate(self,
                start_time: datetime,
                end_time: datetime,
                dt_minutes: int = 15) -> pd.DataFrame:
        """
        Run full simulation

        Args:
            start_time: Simulation start time
            end_time: Simulation end time
            dt_minutes: Time step in minutes

        Returns:
            DataFrame with simulation results
        """
        self.reset()

        current_time = start_time
        step_count = 0

        print(f"Starting simulation: {start_time} to {end_time}")
        print(f"Controller: {self.controller.name}")
        print(f"Time step: {dt_minutes} minutes")

        while current_time < end_time:
            self.step(current_time, dt_minutes)
            current_time += timedelta(minutes=dt_minutes)
            step_count += 1

            if step_count % 1000 == 0:
                print(f"Progress: {step_count} steps, {current_time}")

        print(f"\nSimulation complete: {step_count} steps")

        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        df.set_index('timestamp', inplace=True)

        return df

    def get_summary(self) -> dict:
        """Get simulation summary statistics"""
        df = pd.DataFrame(self.history)

        return {
            'controller': self.controller.name,
            'total_cost': self.total_cost,
            'total_grid_import_kwh': self.total_grid_import_kwh,
            'total_grid_export_kwh': self.total_grid_export_kwh,
            'total_solar_production_kwh': self.solar.total_production_kwh,
            'total_consumption_kwh': self.house.total_consumption_kwh,
            'battery_cycles': (self.battery.total_charge_kwh + self.battery.total_discharge_kwh) / (2 * self.battery.capacity_kwh),
            'battery_degradation_cost': self.battery.total_degradation_cost,
            # Self-consumption rate = how much of solar production is used locally (not exported)
            'self_consumption_rate': (self.solar.total_production_kwh - self.total_grid_export_kwh) / self.solar.total_production_kwh if self.solar.total_production_kwh > 0 else 0,
            # Self-sufficiency rate = how much of consumption is covered without grid import
            'self_sufficiency_rate': 1 - (self.total_grid_import_kwh / self.house.total_consumption_kwh) if self.house.total_consumption_kwh > 0 else 0,
        }

    def calculate_baseline_cost(self, df: pd.DataFrame) -> dict:
        """
        Calculate costs for different baseline scenarios

        Args:
            df: Simulation results DataFrame

        Returns:
            Dictionary with baseline costs and energy flows
        """
        export_price_value = getattr(self.controller, 'export_price', 0.05)
        dt_hours = 15 / 60.0  # Assuming 15 min timestep

        # Scenario 1: With PV, without battery
        baseline_pv_cost = 0.0
        baseline_pv_import = 0.0
        baseline_pv_export = 0.0

        # Scenario 2: No PV, no battery (only grid)
        baseline_no_pv_cost = 0.0
        baseline_no_pv_import = 0.0

        for idx, row in df.iterrows():
            price = row['electricity_price']

            # Scenario 1: With PV, without battery
            net_demand = row['load_power'] - row['solar_power']

            if net_demand > 0:
                # Import from grid
                baseline_pv_import += net_demand * dt_hours
                baseline_pv_cost += net_demand * dt_hours * price
            else:
                # Export excess solar to grid (negative cost = revenue)
                export_power = -net_demand
                baseline_pv_export += export_power * dt_hours
                baseline_pv_cost -= export_power * dt_hours * export_price_value

            # Scenario 2: No PV, no battery (only grid)
            load = row['load_power']
            baseline_no_pv_import += load * dt_hours
            baseline_no_pv_cost += load * dt_hours * price

        return {
            'pv_no_battery': {
                'cost': baseline_pv_cost,
                'import_kwh': baseline_pv_import,
                'export_kwh': baseline_pv_export
            },
            'no_pv_no_battery': {
                'cost': baseline_no_pv_cost,
                'import_kwh': baseline_no_pv_import,
                'export_kwh': 0.0
            }
        }

    def get_savings(self, df: pd.DataFrame) -> dict:
        """
        Calculate savings compared to baselines

        Args:
            df: Simulation results DataFrame

        Returns:
            Dictionary with detailed savings information
        """
        baselines = self.calculate_baseline_cost(df)

        # Savings vs PV without battery
        baseline_pv_cost = baselines['pv_no_battery']['cost']
        savings_vs_pv = baseline_pv_cost - self.total_cost
        savings_pct_vs_pv = (savings_vs_pv / baseline_pv_cost * 100) if baseline_pv_cost > 0 else 0

        # Savings vs no PV, no battery (total grid)
        baseline_no_pv_cost = baselines['no_pv_no_battery']['cost']
        savings_vs_no_pv = baseline_no_pv_cost - self.total_cost
        savings_pct_vs_no_pv = (savings_vs_no_pv / baseline_no_pv_cost * 100) if baseline_no_pv_cost > 0 else 0

        return {
            'system_cost': self.total_cost,
            'system_import_kwh': self.total_grid_import_kwh,
            'system_export_kwh': self.total_grid_export_kwh,

            'baseline_pv_cost': baseline_pv_cost,
            'baseline_pv_import_kwh': baselines['pv_no_battery']['import_kwh'],
            'baseline_pv_export_kwh': baselines['pv_no_battery']['export_kwh'],
            'savings_vs_pv': savings_vs_pv,
            'savings_pct_vs_pv': savings_pct_vs_pv,

            'baseline_no_pv_cost': baseline_no_pv_cost,
            'baseline_no_pv_import_kwh': baselines['no_pv_no_battery']['import_kwh'],
            'savings_vs_no_pv': savings_vs_no_pv,
            'savings_pct_vs_no_pv': savings_pct_vs_no_pv,
        }
