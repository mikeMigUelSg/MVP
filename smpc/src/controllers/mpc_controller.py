"""
MPC controller - Model Predictive Control for battery management
"""
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import linprog
from typing import TYPE_CHECKING, Optional
import time
import logging

if TYPE_CHECKING:
    from ..components.battery import Battery
    from ..components.solar import SolarPanel
    from ..components.house import House
    from ..components.tariff import Tariff

from ..forecasters import LoadForecaster, SolarForecaster

# Configure logging
logger = logging.getLogger(__name__)


class MPCController:
    """
    Model Predictive Control for optimal battery scheduling

    Minimizes total cost over prediction horizon considering:
    - Electricity purchase costs
    - Battery degradation costs
    - Export revenue (if applicable)
    """

    def __init__(self,
                 horizon_steps: int = 96,
                 dt_minutes: int = 15,
                 export_price: float = 0.05,
                 contracted_power_kw: float = None,
                 load_forecaster: Optional[LoadForecaster] = None,
                 solar_forecaster: Optional[SolarForecaster] = None,
                 forecast_method: str = 'historical',
                 use_perfect_forecast: bool = False):
        """
        Args:
            horizon_steps: Prediction horizon length (default: 96 steps = 24h with 15min resolution)
            dt_minutes: Time step duration in minutes
            export_price: Fixed export price in €/kWh (default: 0.05)
            contracted_power_kw: Maximum contracted power from grid in kW (None = no limit)
            load_forecaster: Optional LoadForecaster instance (will be created if None)
            solar_forecaster: Optional SolarForecaster instance (will be created if None)
            forecast_method: Forecasting method ('historical', 'mean', 'naive', 'moving_average')
            use_perfect_forecast: If True, uses actual future data (for testing MPC performance)
        """
        self.horizon_steps = horizon_steps
        self.dt_minutes = dt_minutes
        self.dt_hours = dt_minutes / 60.0
        self.export_price = export_price
        self.contracted_power_kw = contracted_power_kw
        self.use_perfect_forecast = use_perfect_forecast
        # Keep this for backward compatibility with other code
        self.export_price_factor = 0.7

        # Forecasters
        self.load_forecaster = load_forecaster
        self.solar_forecaster = solar_forecaster
        self.forecast_method = forecast_method

        self.name = "MPC" if not use_perfect_forecast else "MPC (Perfect Forecast)"

    def compute_action(self,
                      timestamp: datetime,
                      solar_power: float,
                      load_power: float,
                      battery: 'Battery',
                      tariff: 'Tariff',
                      solar_panel: 'SolarPanel',
                      house: 'House') -> float:
        """
        Compute optimal battery action using MPC

        Args:
            timestamp: Current timestamp
            solar_power: Current solar production in kW
            load_power: Current house consumption in kW
            battery: Battery object
            tariff: Tariff object
            solar_panel: Solar panel object (for forecast)
            house: House object (for forecast)

        Returns:
            Battery power in kW (positive = charge, negative = discharge)
        """
        action_start_time = time.time()

        # Get forecasts
        forecast_start = time.time()
        if self.use_perfect_forecast:
            # PERFECT FORECAST MODE: Use actual future data (for testing only!)
            solar_forecast = np.zeros(self.horizon_steps)
            load_forecast = np.zeros(self.horizon_steps)

            current_time = timestamp
            for i in range(self.horizon_steps):
                solar_forecast[i] = solar_panel.get_production(current_time)
                load_forecast[i] = house.get_consumption(current_time)
                current_time += timedelta(minutes=self.dt_minutes)
        else:
            # NORMAL MODE: Use forecasters
            # Initialize forecasters if not already done
            if self.load_forecaster is None:
                self.load_forecaster = LoadForecaster(house=house)

            if self.solar_forecaster is None:
                self.solar_forecaster = SolarForecaster(solar_panel=solar_panel)

            # Get forecasts using forecasters
            # Try using forecasters first, fallback to direct methods if they fail
            try:
                solar_forecast = self.solar_forecaster.get_forecast(
                    start_time=timestamp,
                    n_steps=self.horizon_steps,
                    dt_minutes=self.dt_minutes,
                    method=self.forecast_method,
                    context_hours=24,
                    apply_night_constraint=True
                )
            except Exception as e:
                print(f"Warning: Solar forecaster failed ({e}), using fallback method")
                solar_forecast = solar_panel.get_production_forecast(
                    timestamp, self.horizon_steps, self.dt_minutes
                )

            try:
                load_forecast = self.load_forecaster.get_forecast(
                    start_time=timestamp,
                    n_steps=self.horizon_steps,
                    dt_minutes=self.dt_minutes,
                    method=self.forecast_method,
                    context_hours=24
                )
            except Exception as e:
                print(f"Warning: Load forecaster failed ({e}), using fallback method")
                load_forecast = house.get_consumption_forecast(
                    timestamp, self.horizon_steps, self.dt_minutes
                )

        # Get price forecast
        if hasattr(tariff, 'get_prices_for_horizon'):
            price_forecast = tariff.get_prices_for_horizon(
                timestamp, self.horizon_steps, self.dt_minutes
            )
        else:
            # Fallback: constant price
            price_forecast = np.array([tariff.get_price(timestamp)] * self.horizon_steps)

        forecast_time = time.time() - forecast_start
        logger.debug(f"  [MPC] Forecast generation: {forecast_time:.6f}s")

        # Solve optimization problem
        optimization_start = time.time()
        solution = self._solve_optimization(
            solar_forecast,
            load_forecast,
            price_forecast,
            battery
        )
        optimization_time = time.time() - optimization_start
        logger.debug(f"  [MPC] Optimization solve: {optimization_time:.6f}s")

        total_action_time = time.time() - action_start_time
        logger.debug(f"  [MPC] compute_action total: {total_action_time:.6f}s")

        if solution is not None:
            # Return first action (receding horizon)
            return solution[0]
        else:
            # Fallback to simple rule if optimization fails
            return 0.0

    def _solve_optimization(self,
                           solar_forecast: np.ndarray,
                           load_forecast: np.ndarray,
                           price_forecast: np.ndarray,
                           battery: 'Battery') -> np.ndarray:
        """
        Solve MPC optimization problem using linear programming

        Variables for each time step:
        - p_bat[k]: battery power (can be positive or negative)
        - p_grid[k]: grid power (positive = import, negative = export)
        - p_charge[k]: battery charging power (>= 0)
        - p_discharge[k]: battery discharging power (>= 0)
        - soc[k]: state of charge

        We'll use a simplified formulation with split charge/discharge variables.
        """
        setup_start = time.time()

        N = self.horizon_steps
        dt = self.dt_hours

        # Battery parameters
        E_cap = battery.capacity_kwh
        P_max = battery.max_power_kw
        P_charge_max = battery.max_charge_power_kw
        P_discharge_max = battery.max_discharge_power_kw
        eta_c = battery.efficiency_charge
        eta_d = battery.efficiency_discharge
        soc_0 = battery.get_soc()
        soc_min = battery.min_soc
        soc_max = battery.max_soc
        deg_cost = battery.degradation_cost_per_kwh

        # Net demand forecast (load - solar)
        net_demand = load_forecast - solar_forecast

        # Decision variables: [p_charge[0..N-1], p_discharge[0..N-1], p_import[0..N-1], p_export[0..N-1]]
        # Total: 4*N variables
        # Separated import/export to handle different prices correctly
        n_vars = 4 * N

        # Objective: minimize total cost
        # Cost = sum(price * p_import * dt - export_price * p_export * dt + deg_cost * (p_charge + p_discharge) * dt * eta)
        # Add penalty to discourage simultaneous import/export
        # Terminal cost (Article Eq. 9): g_N(x_N) = -x_N * (c_s + c_f) / 2
        # This values the final battery energy as the average of buy and sell prices
        c = np.zeros(n_vars)

        # Calculate average price for terminal cost
        # c_s = buy price (average over horizon)
        # c_f = sell price (export_price)
        c_s = np.mean(price_forecast)  # Average buy price
        c_f = self.export_price  # Sell price
        terminal_price = (c_s + c_f) / 2.0

        for k in range(N):
            idx_charge = k
            idx_discharge = N + k
            idx_import = 2 * N + k
            idx_export = 3 * N + k

            # Grid import cost (buying from grid)
            c[idx_import] = price_forecast[k] * dt

            # Grid export revenue (selling to grid) - negative cost = revenue
            # Using fixed export price (e.g., 0.05 €/kWh)
            c[idx_export] = -self.export_price * dt

            # Add small penalty to both to discourage simultaneous import/export
            # This penalty is much smaller than prices but helps break ties
            penalty = 0.001 * dt
            c[idx_import] += penalty
            c[idx_export] += penalty

            # Degradation cost for charging
            c[idx_charge] = deg_cost * dt * eta_c

            # Degradation cost for discharging
            # Note: Battery discharge is cheaper than grid import (no purchase cost)
            # Only pay for degradation + efficiency losses
            c[idx_discharge] = deg_cost * dt / eta_d

        # Add terminal cost: -SOC_N * E_cap * terminal_price
        # SOC_N = SOC_0 + sum_{k=0}^{N-1} (eta_c * p_charge[k] - p_discharge[k]/eta_d) * dt / E_cap
        # Terminal cost contribution to each variable:
        for k in range(N):
            idx_charge = k
            idx_discharge = N + k

            # Charging increases final SOC -> increases final value -> reduces cost
            c[idx_charge] -= eta_c * dt * terminal_price

            # Discharging decreases final SOC -> decreases final value -> increases cost
            c[idx_discharge] += dt / eta_d * terminal_price

        # Inequality constraints: A_ub @ x <= b_ub
        # We need many constraints, let's build them systematically
        A_ub = []
        b_ub = []

        for k in range(N):
            idx_charge = k
            idx_discharge = N + k
            idx_import = 2 * N + k
            idx_export = 3 * N + k

            # Constraint: p_charge <= P_charge_max (maximum charging current)
            row = np.zeros(n_vars)
            row[idx_charge] = 1.0
            A_ub.append(row)
            b_ub.append(P_charge_max)

            # Constraint: p_discharge <= P_discharge_max (maximum discharging current)
            row = np.zeros(n_vars)
            row[idx_discharge] = 1.0
            A_ub.append(row)
            b_ub.append(P_discharge_max)

            # Constraint: Can only export solar (not battery discharge or grid import)
            # This prevents: buy cheap -> charge -> discharge -> export at loss
            # export <= solar
            row = np.zeros(n_vars)
            row[idx_export] = 1.0
            A_ub.append(row)
            b_ub.append(solar_forecast[k])

            # Constraint: Discharge limited to net demand (load - solar)
            # This ensures battery only discharges to meet actual consumption deficit
            # discharge <= max(0, load - solar)
            max_discharge_need = max(0.0, load_forecast[k] - solar_forecast[k])
            row = np.zeros(n_vars)
            row[idx_discharge] = 1.0
            A_ub.append(row)
            b_ub.append(max_discharge_need)

            # Constraint: Total supply (discharge + import + solar) should not exceed load
            # This prevents simultaneous import+discharge when solar already covers load
            # discharge + import <= max(0, load - solar)
            row = np.zeros(n_vars)
            row[idx_discharge] = 1.0
            row[idx_import] = 1.0
            A_ub.append(row)
            b_ub.append(max_discharge_need)

            # Constraint: Contracted power limit (maximum grid import)
            # import <= contracted_power_kw
            if self.contracted_power_kw is not None:
                row = np.zeros(n_vars)
                row[idx_import] = 1.0
                A_ub.append(row)
                b_ub.append(self.contracted_power_kw)

            # Allow grid charging for price arbitrage
            # The optimizer will decide when it's economical

            # SOC constraints will be handled in equality constraints

        # Bounds for variables
        bounds = []
        for k in range(N):
            bounds.append((0, None))  # p_charge >= 0
        for k in range(N):
            bounds.append((0, None))  # p_discharge >= 0
        for k in range(N):
            bounds.append((0, None))  # p_import >= 0
        for k in range(N):
            bounds.append((0, None))  # p_export >= 0

        # Equality constraints: A_eq @ x = b_eq
        # Power balance: (p_import - p_export) + solar - load = p_charge - p_discharge
        # Rearranged: p_import - p_export - p_charge + p_discharge = net_demand
        # SOC dynamics: soc[k+1] = soc[k] + (eta_c * p_charge[k] - p_discharge[k]/eta_d) * dt / E_cap
        A_eq = []
        b_eq = []

        for k in range(N):
            idx_charge = k
            idx_discharge = N + k
            idx_import = 2 * N + k
            idx_export = 3 * N + k

            # Power balance: p_import - p_export = (p_charge - p_discharge) + (load - solar)
            # Rearranged: p_import - p_export - p_charge + p_discharge = net_demand
            row = np.zeros(n_vars)
            row[idx_charge] = -1.0
            row[idx_discharge] = 1.0
            row[idx_import] = 1.0
            row[idx_export] = -1.0
            A_eq.append(row)
            b_eq.append(net_demand[k])

        # SOC constraints via inequality (to keep within bounds)
        # We need to track SOC evolution
        # soc[0] = soc_0
        # soc[k] = soc_0 + sum_{i=0}^{k-1} (eta_c * p_charge[i] - p_discharge[i]/eta_d) * dt / E_cap

        for k in range(N):
            # Upper bound: soc[k] <= soc_max
            # soc_0 + sum_{i=0}^{k} (...) <= soc_max
            row = np.zeros(n_vars)
            for i in range(k + 1):
                row[i] = eta_c * dt / E_cap  # p_charge contribution
                row[N + i] = -dt / (eta_d * E_cap)  # p_discharge contribution
            A_ub.append(row)
            b_ub.append(soc_max - soc_0)

            # Lower bound: soc[k] >= soc_min
            # Rearranged: -soc[k] <= -soc_min
            row = np.zeros(n_vars)
            for i in range(k + 1):
                row[i] = -eta_c * dt / E_cap
                row[N + i] = dt / (eta_d * E_cap)
            A_ub.append(row)
            b_ub.append(-soc_min + soc_0)

        # Convert to numpy arrays
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None

        setup_time = time.time() - setup_start
        logger.debug(f"    [MPC] LP problem setup: {setup_time:.6f}s")

        # Solve linear program
        solver_start = time.time()
        try:
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            solver_time = time.time() - solver_start
            logger.debug(f"    [MPC] LP solver time: {solver_time:.6f}s")

            if result.success:
                # Extract battery power schedule (charge - discharge)
                p_charge = result.x[0:N]
                p_discharge = result.x[N:2*N]
                p_import = result.x[2*N:3*N]
                p_export = result.x[3*N:4*N]
                p_battery = p_charge - p_discharge

                # DEBUG: Check first timestep for violations
                if p_export[0] > solar_forecast[0] + p_discharge[0] + 0.001:
                    print(f"WARNING: Export constraint violated at t=0!")
                    print(f"  export={p_export[0]:.3f}, solar={solar_forecast[0]:.3f}, discharge={p_discharge[0]:.3f}")

                # Check for real problems: importing AND exporting (buying high, selling low)
                if p_export[0] > 0.01 and p_import[0] > 0.01:
                    print(f"WARNING: Importing AND exporting simultaneously at t=0!")
                    print(f"  Import:   {p_import[0]:.3f} kW")
                    print(f"  Export:   {p_export[0]:.3f} kW")
                    print(f"  This causes economic loss (buy high, sell low)!")

                # Check if charging while importing (may be OK for arbitrage, but worth noting)
                if p_charge[0] > 0.01 and p_import[0] > 0.01:
                    # Only warn if export is also happening (definitely wrong)
                    if p_export[0] > 0.01:
                        print(f"WARNING: Import + Charge + Export at t=0 (illogical)!")
                        print(f"  Solar:    {solar_forecast[0]:.3f} kW")
                        print(f"  Load:     {load_forecast[0]:.3f} kW")
                        print(f"  Import:   {p_import[0]:.3f} kW")
                        print(f"  Charge:   {p_charge[0]:.3f} kW")
                        print(f"  Export:   {p_export[0]:.3f} kW")

                return p_battery
            else:
                print(f"Optimization failed: {result.message}")
                return None

        except Exception as e:
            print(f"Error in optimization: {e}")
            return None
