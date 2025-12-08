"""
Energy Management System - integrates all components
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union
import time
import logging
from tqdm import tqdm

from .components.battery import Battery
from .components.solar import SolarPanel
from .components.house import House
from .components.tariff import Tariff
from .components.inverter import Inverter
from .controllers.rule_based import RuleBasedController
from .controllers.mpc_controller import MPCController
from .controllers.sdp_controller import SDPController

# Configure logging
logger = logging.getLogger(__name__)


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
                 inverter: Inverter,
                 controller: Union[RuleBasedController, MPCController, SDPController],
                 hvac = None,
                 export_price: float = 0.05,
                 max_export_kw: float = float('inf')):
        """
        Args:
            battery: Battery component
            solar: Solar panel component
            house: House component
            tariff: Electricity tariff
            inverter: Inverter component
            controller: Control strategy (rule-based or MPC)
            hvac: HVAC component (optional)
            export_price: Price for exporting energy to grid in €/kWh
            max_export_kw: Maximum power export to grid in kW (can cause curtailment)
        """
        self.battery = battery
        self.solar = solar
        self.house = house
        self.tariff = tariff
        self.inverter = inverter
        self.controller = controller
        self.hvac = hvac
        self.export_price = export_price
        self.max_export_kw = max_export_kw

        # Tracking variables
        self.history = {
            'timestamp': [],
            'solar_power': [],
            'solar_available': [],
            'solar_curtailed': [],
            'load_power': [],
            'battery_power': [],
            'battery_soc': [],
            'grid_power': [],
            'electricity_price': [],
            'grid_import': [],
            'grid_export': [],
            'cost': [],
            'degradation_cost': [],
            'inverter_utilization': [],
            'hvac_power': [],
            'hvac_mode': [],
            'T_in': [],
            'T_out': []
        }

        self.total_cost = 0.0
        self.total_grid_import_kwh = 0.0
        self.total_grid_export_kwh = 0.0
        self.total_solar_curtailed_kwh = 0.0
        self._curtailment_events = []

    def reset(self):
        """Reset system to initial state"""
        self.battery.reset()
        self.solar.reset()
        self.house.reset()
        self.inverter.reset()
        if self.hvac:
            self.hvac.reset()

        for key in self.history:
            self.history[key] = []

        self.total_cost = 0.0
        self.total_grid_import_kwh = 0.0
        self.total_grid_export_kwh = 0.0
        self.total_solar_curtailed_kwh = 0.0
        self._curtailment_events = []

    def step(self, timestamp: datetime, dt_minutes: int = 15) -> dict:
        """
        Execute one simulation time step with power balance

        Args:
            timestamp: Current timestamp
            dt_minutes: Time step duration in minutes

        Returns:
            Dictionary with step results
        """
        step_start_time = time.time()

        dt_hours = dt_minutes / 60.0

        # Get solar production and house load
        component_start = time.time()
        solar_result = self.solar.step(timestamp, dt_hours)
        house_result = self.house.step(timestamp, dt_hours)

        solar_available = solar_result['power_kw']  # Available solar power (before curtailment)
        load_power = house_result['power_kw']

        # Get HVAC power (if enabled)
        hvac_power = 0.0
        hvac_mode = 'off'
        T_in = None
        T_out = None
        if self.hvac:
            hvac_result = self.hvac.step(timestamp, dt_hours)
            hvac_power = hvac_result['power_kw']
            hvac_mode = hvac_result['hvac_mode']
            T_in = hvac_result['T_in']
            T_out = hvac_result['T_out']

            # Add HVAC to total load
            load_power += hvac_power

        # Get electricity price
        price = self.tariff.get_price(timestamp)
        component_time = time.time() - component_start

        # Compute controller action
        controller_start = time.time()
        if isinstance(self.controller, MPCController):
            battery_power_cmd = self.controller.compute_action(
                timestamp, solar_available, load_power,
                self.battery, self.tariff, self.solar, self.house
            )
        elif isinstance(self.controller, SDPController):
            # SDP controller needs forecasters
            pv_forecaster = getattr(self.controller, 'pv_forecaster', None)
            load_forecaster = getattr(self.controller, 'load_forecaster', None)
            battery_power_cmd = self.controller.compute_action(
                timestamp, solar_available, load_power,
                self.battery, self.tariff, self.solar, self.house,
                pv_forecaster, load_forecaster
            )
        else:
            battery_power_cmd = self.controller.compute_action(
                timestamp, solar_available, load_power,
                self.battery, self.tariff, dt_hours
            )
        controller_time = time.time() - controller_start

        # Constraint: Battery cannot discharge to export to grid
        # Only solar excess can be exported
        if battery_power_cmd < 0:  # If battery wants to discharge
            net_demand = load_power - solar_available
            if net_demand < 0:
                # There's already excess solar, battery should not discharge
                battery_power_cmd = 0.0
            else:
                # Battery can only discharge to meet demand, not to export
                # Limit discharge to net_demand
                battery_power_cmd = max(battery_power_cmd, -net_demand)

        # Apply inverter limit to DC→AC power flow
        # The inverter limits: solar (DC) + battery discharge (DC) → AC side (load + grid)
        # Battery charging (AC→DC) is NOT limited by the inverter

        # Apply inverter limit to solar + potential battery discharge
        if battery_power_cmd < 0:  # Battery wants to discharge
            # Check if solar + battery discharge exceeds inverter limit
            total_dc_to_ac = solar_available + abs(battery_power_cmd)
            if total_dc_to_ac > self.inverter.max_power_kw:
                # Inverter is limiting, reduce battery discharge command
                # Prioritize solar, then battery
                available_for_battery = self.inverter.max_power_kw - solar_available
                battery_power_cmd = -max(0.0, available_for_battery)


        if battery_power_cmd > 0:  # pedir carga
            # 1) usa excedente PV primeiro (DC->DC, não consome AC)
            pv_surplus = max(0.0, solar_available - load_power)
            charge_from_pv = min(battery_power_cmd, pv_surplus)

            # 2) restante vem da rede (AC->DC) e DEVE ser limitado
            charge_from_grid_cmd = battery_power_cmd - charge_from_pv

            if charge_from_grid_cmd > 0:
                # limites configuráveis do teu modelo/inversor
                grid_charge_limit = getattr(self.inverter, "grid_charge_max_kw", float("inf"))
                ac_budget = self.inverter.max_power_kw  # refina se modelares fluxos AC em detalhe
                charge_from_grid = min(charge_from_grid_cmd, grid_charge_limit, max(0.0, ac_budget))
            else:
                charge_from_grid = 0.0

            battery_power_cmd = charge_from_pv + charge_from_grid

        # Execute battery action with potentially limited command
        battery_start = time.time()
        battery_result = self.battery.step(battery_power_cmd, dt_hours)
        battery_power = battery_result['power_kw']
        battery_time = time.time() - battery_start

        # Calculate curtailment
        # Curtailment occurs when available solar exceeds what can be used considering:
        # 1. Inverter capacity limits DC→AC conversion
        # 2. Load demand
        # 3. Battery charging capacity
        # 4. Grid export limit (max_export_kw)

        # Step 1: Determine solar allocation priorities
        # Priority 1: Direct load consumption (through inverter)
        solar_to_load = min(solar_available, load_power)

        # Priority 2: Battery charging (DC→DC, bypasses inverter)
        solar_to_battery_dc = 0.0
        if battery_power > 0:  # Charging
            pv_surplus_after_load = max(0.0, solar_available - solar_to_load)
            solar_to_battery_dc = min(battery_power, pv_surplus_after_load)

        # Priority 3: Grid export (through inverter, limited by max_export_kw and inverter capacity)
        solar_available_for_export = solar_available - solar_to_load - solar_to_battery_dc

        # Check inverter constraint for export
        # Battery discharge also uses inverter capacity
        battery_discharge = abs(min(0.0, battery_power))
        inverter_used_by_load_and_battery = solar_to_load + battery_discharge
        inverter_available_for_export = max(0.0, self.inverter.max_power_kw - inverter_used_by_load_and_battery)

        # Export is limited by: available solar, inverter capacity, and grid export limit
        max_possible_export = min(
            solar_available_for_export,
            inverter_available_for_export,
            self.max_export_kw
        )

        solar_to_grid = max(0.0, max_possible_export)

        # Calculate actual usable solar and curtailment
        solar_power = solar_to_load + solar_to_battery_dc + solar_to_grid
        solar_curtailed = max(0.0, solar_available - solar_power)

        # Log and track curtailment events if significant
        if solar_curtailed > 0.01:  # More than 10W curtailed
            if not hasattr(self, '_curtailment_events'):
                self._curtailment_events = []

            reasons = []
            if inverter_available_for_export < solar_available_for_export:
                reasons.append(f"inverter limit ({self.inverter.max_power_kw}kW)")
            if self.max_export_kw < solar_available_for_export:
                reasons.append(f"grid export limit ({self.max_export_kw}kW)")

            # Log with INFO level so it shows
            logger.info(f"[Curtailment] {solar_curtailed:.3f}kW at {timestamp.strftime('%Y-%m-%d %H:%M')} - Reasons: {', '.join(reasons)}")

            self._curtailment_events.append({
                'timestamp': timestamp,
                'power_kw': solar_curtailed,
                'reasons': reasons
            })

        # Calculate actual AC power through inverter (only DC→AC conversion)
        # This is solar to load + solar to grid + battery discharge (if discharging)
        solar_through_inverter = solar_to_load + solar_to_grid
        inverter_ac_power = solar_through_inverter + battery_discharge
        inverter_result = self.inverter.step(inverter_ac_power, dt_hours)

        # Power balance: grid_power = load - solar + battery_power
        # When battery_power > 0 (charging), it's a load (AC from grid → DC to battery, not through inverter)
        # When battery_power < 0 (discharging), it's a source (DC from battery → AC through inverter)
        # Positive grid_power = import from grid
        # Negative grid_power = export to grid
        grid_power = load_power - solar_power + battery_power

        # Split into import and export
        grid_import = max(0.0, grid_power)
        grid_export = max(0.0, -grid_power)

        # Apply grid export limit (should already be handled by curtailment calculation, but double-check)
        if grid_export > self.max_export_kw:
            logger.warning(f"[System] Grid export {grid_export:.3f}kW exceeds limit {self.max_export_kw}kW at {timestamp}")
            grid_export = self.max_export_kw
            # Recalculate grid_power with limited export
            grid_power = -grid_export

        # Calculate costs
        # Import cost
        import_cost = grid_import * dt_hours * price

        # Export revenue
        export_revenue = grid_export * dt_hours * self.export_price

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

        total_step_cost = step_cost #+ degradation_cost

        # Update totals
        self.total_cost += total_step_cost
        self.total_grid_import_kwh += grid_import * dt_hours
        self.total_grid_export_kwh += grid_export * dt_hours
        self.total_solar_curtailed_kwh += solar_curtailed * dt_hours

        # Record history
        self.history['timestamp'].append(timestamp)
        self.history['solar_power'].append(solar_power)
        self.history['solar_available'].append(solar_available)
        self.history['solar_curtailed'].append(solar_curtailed)
        self.history['load_power'].append(load_power)
        self.history['battery_power'].append(battery_power)
        self.history['battery_soc'].append(self.battery.get_soc())
        self.history['grid_power'].append(grid_power)
        self.history['electricity_price'].append(price)
        self.history['grid_import'].append(grid_import)
        self.history['grid_export'].append(grid_export)
        self.history['cost'].append(total_step_cost)
        self.history['degradation_cost'].append(degradation_cost)
        self.history['inverter_utilization'].append(inverter_result['utilization'])
        self.history['hvac_power'].append(hvac_power)
        self.history['hvac_mode'].append(hvac_mode)
        self.history['T_in'].append(T_in if T_in is not None else np.nan)
        self.history['T_out'].append(T_out if T_out is not None else np.nan)

        # Verify power balance
        power_balance_error = abs(
            solar_power + grid_import - load_power - grid_export - battery_power
        )

        total_step_time = time.time() - step_start_time
        logger.debug(f"[System] Step time: {total_step_time:.6f}s (controller: {controller_time:.6f}s, components: {component_time:.6f}s, battery: {battery_time:.6f}s)")

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
                dt_minutes: int = 15,
                show_progress: bool = True) -> pd.DataFrame:
        """
        Run full simulation

        Args:
            start_time: Simulation start time
            end_time: Simulation end time
            dt_minutes: Time step in minutes
            show_progress: Whether to display progress bar

        Returns:
            DataFrame with simulation results
        """
        simulation_start_time = time.time()

        self.reset()

        current_time = start_time
        step_count = 0
        step_times = []

        # Calculate total number of steps
        total_steps = int((end_time - start_time).total_seconds() / (dt_minutes * 60))

        logger.info(f"[System] Starting simulation: {start_time} to {end_time} with controller {self.controller.name}")

        # Create progress bar only if requested
        if show_progress:
            with tqdm(total=total_steps, desc=f"{self.controller.name}",
                      unit="steps", ncols=100, leave=True) as pbar:

                while current_time < end_time:
                    step_start = time.time()
                    self.step(current_time, dt_minutes)
                    step_time = time.time() - step_start
                    step_times.append(step_time)

                    current_time += timedelta(minutes=dt_minutes)
                    step_count += 1
                    pbar.update(1)
        else:
            # No progress bar - just run the simulation
            while current_time < end_time:
                step_start = time.time()
                self.step(current_time, dt_minutes)
                step_time = time.time() - step_start
                step_times.append(step_time)

                current_time += timedelta(minutes=dt_minutes)
                step_count += 1

        total_simulation_time = time.time() - simulation_start_time

        # Calculate statistics
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        min_step_time = np.min(step_times)
        p95_step_time = np.percentile(step_times, 95)

        logger.info(f"[System] Simulation complete: {step_count} steps in {total_simulation_time:.2f}s")
        logger.info(f"[System] Step time stats - Avg: {avg_step_time*1000:.2f}ms, Min: {min_step_time*1000:.2f}ms, Max: {max_step_time*1000:.2f}ms, P95: {p95_step_time*1000:.2f}ms")

        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        df.set_index('timestamp', inplace=True)

        return df

    def get_summary(self) -> dict:
        """Get simulation summary statistics"""
        df = pd.DataFrame(self.history)

        # Calculate usable capacity (between min_soc and max_soc)
        usable_capacity_kwh = self.battery.capacity_kwh * (self.battery.max_soc - self.battery.min_soc)

        # Calculate total available solar (including curtailed)
        total_solar_available_kwh = self.solar.total_production_kwh + self.total_solar_curtailed_kwh

        return {
            'controller': self.controller.name,
            'total_cost': self.total_cost,
            'total_grid_import_kwh': self.total_grid_import_kwh,
            'total_grid_export_kwh': self.total_grid_export_kwh,
            'total_solar_production_kwh': self.solar.total_production_kwh,
            'total_solar_available_kwh': total_solar_available_kwh,
            'total_solar_curtailed_kwh': self.total_solar_curtailed_kwh,
            'curtailment_rate': (self.total_solar_curtailed_kwh / total_solar_available_kwh) if total_solar_available_kwh > 0 else 0,
            'total_consumption_kwh': self.house.total_consumption_kwh,
            'battery_cycles': (self.battery.total_charge_kwh + self.battery.total_discharge_kwh) / (2 * self.battery.capacity_kwh),
            'battery_cycles_usable': (self.battery.total_charge_kwh + self.battery.total_discharge_kwh) / (2 * usable_capacity_kwh),
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
                baseline_pv_cost -= export_power * dt_hours * self.export_price

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

    def get_energy_balance_comparison(self, df: pd.DataFrame) -> dict:
        """
        Calculate complete energy balance for all three scenarios

        Args:
            df: Simulation results DataFrame

        Returns:
            Dictionary with energy balance for all scenarios
        """
        baselines = self.calculate_baseline_cost(df)
        dt_hours = 15 / 60.0  # Assuming 15 min timestep

        # Total solar produced and load consumed (same for all scenarios with PV)
        total_solar_kwh = df['solar_power'].sum() * dt_hours
        total_load_kwh = df['load_power'].sum() * dt_hours

        # Get curtailment data if available
        total_solar_available_kwh = df['solar_available'].sum() * dt_hours if 'solar_available' in df.columns else total_solar_kwh
        system_curtailed_kwh = self.total_solar_curtailed_kwh

        # Scenario 1: System with battery (current system)
        system_import = self.total_grid_import_kwh
        system_export = self.total_grid_export_kwh
        system_solar_used_locally = total_solar_kwh - system_export
        system_solar_self_consumption = (system_solar_used_locally / total_solar_kwh * 100) if total_solar_kwh > 0 else 0
        system_load_self_sufficiency = ((total_load_kwh - system_import) / total_load_kwh * 100) if total_load_kwh > 0 else 0

        # Scenario 2: PV without battery - calculate curtailment due to inverter/export limits
        # Without battery, solar can only go to: (1) load, (2) grid export
        # Curtailment occurs if solar > load and export is limited
        pv_import = baselines['pv_no_battery']['import_kwh']
        pv_export = baselines['pv_no_battery']['export_kwh']
        pv_curtailed_kwh = 0.0
        if 'solar_available' in df.columns:
            # Calculate curtailment for PV-only scenario
            for idx, row in df.iterrows():
                solar_avail = row['solar_available']
                load = row['load_power']
                excess = max(0.0, solar_avail - load)
                # Export is limited by inverter and max_export_kw
                max_export = min(self.inverter.max_power_kw - min(solar_avail, load), self.max_export_kw)
                curtailed = max(0.0, excess - max_export)
                pv_curtailed_kwh += curtailed * dt_hours

        pv_solar_used_locally = total_solar_kwh - pv_export
        pv_solar_self_consumption = (pv_solar_used_locally / total_solar_kwh * 100) if total_solar_kwh > 0 else 0
        pv_load_self_sufficiency = ((total_load_kwh - pv_import) / total_load_kwh * 100) if total_load_kwh > 0 else 0

        # Scenario 3: No PV, no battery
        no_pv_import = baselines['no_pv_no_battery']['import_kwh']

        # Solar to load ratio (only for scenarios with PV)
        solar_to_load_ratio = (total_solar_kwh / total_load_kwh * 100) if total_load_kwh > 0 else 0

        return {
            'system_with_battery': {
                'solar_produced_kwh': total_solar_kwh,
                'solar_curtailed_kwh': system_curtailed_kwh,
                'load_consumed_kwh': total_load_kwh,
                'grid_import_kwh': system_import,
                'grid_export_kwh': system_export,
                'solar_to_load_ratio_pct': solar_to_load_ratio,
                'solar_self_consumption_pct': system_solar_self_consumption,
                'load_self_sufficiency_pct': system_load_self_sufficiency
            },
            'pv_without_battery': {
                'solar_produced_kwh': total_solar_kwh,
                'solar_curtailed_kwh': pv_curtailed_kwh,
                'load_consumed_kwh': total_load_kwh,
                'grid_import_kwh': pv_import,
                'grid_export_kwh': pv_export,
                'solar_to_load_ratio_pct': solar_to_load_ratio,
                'solar_self_consumption_pct': pv_solar_self_consumption,
                'load_self_sufficiency_pct': pv_load_self_sufficiency
            },
            'no_pv_no_battery': {
                'solar_produced_kwh': 0.0,
                'solar_curtailed_kwh': 0.0,
                'load_consumed_kwh': total_load_kwh,
                'grid_import_kwh': no_pv_import,
                'grid_export_kwh': 0.0,
                'solar_to_load_ratio_pct': 0.0,
                'solar_self_consumption_pct': 0.0,
                'load_self_sufficiency_pct': 0.0
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
