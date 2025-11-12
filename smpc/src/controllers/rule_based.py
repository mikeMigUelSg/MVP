"""
Rule-based controller - simple heuristic battery control
"""
from datetime import datetime
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..components.battery import Battery
    from ..components.tariff import Tariff


class RuleBasedController:
    """
    Simple rule-based controller for battery management

    Strategies:
    - 'simple': Original heuristic based on a fixed high price threshold.
    - 'dynamic_threshold': Uses price forecasts for the next 24 hours to set dynamic thresholds.

    Rules for 'dynamic_threshold':
    1. If excess solar production -> charge battery.
    2. If net demand:
        - Price > 75th percentile: Discharge to cover demand.
        - Price < 25th percentile: Charge from grid (if SOC is low).
        - Otherwise: Use grid.
    3. Try to maintain battery between min/max SOC.
    """

    def __init__(self,
                 strategy: str = 'simple',
                 high_price_threshold: float = 0.15,
                 low_soc_threshold: float = 0.3,
                 high_soc_threshold: float = 0.8,
                 charge_soc_min: float = 0.5):
        """
        Args:
            strategy: Control strategy ('simple' or 'dynamic_threshold').
            high_price_threshold: Price threshold for 'simple' strategy (€/kWh).
            low_soc_threshold: SOC below which to avoid discharge.
            high_soc_threshold: SOC above which to avoid charge.
            charge_soc_min: Minimum SOC to start charging from grid in 'dynamic_threshold' strategy.
        """
        if strategy not in ['simple', 'dynamic_threshold']:
            raise ValueError("Strategy must be 'simple' or 'dynamic_threshold'")

        self.strategy = strategy
        self.high_price_threshold = high_price_threshold
        self.low_soc_threshold = low_soc_threshold
        self.high_soc_threshold = high_soc_threshold
        self.charge_soc_min = charge_soc_min

        self.name = f"Rule-Based ({strategy})"

    def compute_action(self,
                      timestamp: datetime,
                      solar_power: float,
                      load_power: float,
                      battery: 'Battery',
                      tariff: 'Tariff',
                      dt_hours: float) -> float:
        """
        Compute battery power action based on the selected strategy.
        """
        if self.strategy == 'simple':
            return self._compute_simple_action(timestamp, solar_power, load_power, battery, tariff, dt_hours)
        elif self.strategy == 'dynamic_threshold':
            return self._compute_dynamic_action(timestamp, solar_power, load_power, battery, tariff, dt_hours)

    def _compute_simple_action(self, timestamp: datetime, solar_power: float, load_power: float,
                               battery: 'Battery', tariff: 'Tariff', dt_hours: float) -> float:
        """Original simple heuristic."""
        soc = battery.get_soc()
        current_price = tariff.get_price(timestamp)
        net_power = solar_power - load_power

        if net_power > 0:
            if soc < self.high_soc_threshold:
                max_charge = battery.get_available_charge_power(dt_hours)
                battery_power = min(net_power, max_charge)
            else:
                battery_power = 0.0
        else:
            demand = -net_power
            if current_price >= self.high_price_threshold and soc > self.low_soc_threshold:
                max_discharge = battery.get_available_discharge_power(dt_hours)
                battery_power = -min(demand, max_discharge)
            else:
                battery_power = 0.0
        return battery_power

    def _compute_dynamic_action(self, timestamp: datetime, solar_power: float, load_power: float,
                                battery: 'Battery', tariff: 'Tariff', dt_hours: float) -> float:
        """Dynamic threshold heuristic based on 24h price forecast."""
        soc = battery.get_soc()
        current_price = tariff.get_price(timestamp)
        net_power = solar_power - load_power

        # Get price forecast for the next 24 hours
        n_steps_24h = int(24 / dt_hours)
        try:
            # Assumes tariff object has this method, which is true for BiHorariaTariff
            future_prices = tariff.get_prices_for_horizon(timestamp, n_steps=n_steps_24h, dt_minutes=int(dt_hours * 60))
        except AttributeError:
            # Fallback to simple strategy if the method is not available
            return self._compute_simple_action(timestamp, solar_power, load_power, battery, tariff, dt_hours)

        # Calculate price quartiles
        q1_price, q3_price = np.percentile(future_prices, [25, 75])

        # Rule 1: Excess solar production
        if net_power > 0:
            if soc < self.high_soc_threshold:
                max_charge = battery.get_available_charge_power(dt_hours)
                battery_power = min(net_power, max_charge)
            else:
                battery_power = 0.0  # Export to grid
        # Rule 2: Net demand
        else:
            demand = -net_power
            max_discharge = battery.get_available_discharge_power(dt_hours)
            max_charge = battery.get_available_charge_power(dt_hours)

            # High price period -> discharge
            if current_price >= q3_price and soc > self.low_soc_threshold:
                battery_power = -min(demand, max_discharge)
            # Low price period -> charge from grid
            elif current_price <= q1_price and soc < self.charge_soc_min:
                # Charge with power equivalent to demand, up to max charge rate
                charge_power = min(demand, max_charge)
                battery_power = charge_power
            # Mid-price period -> use grid
            else:
                battery_power = 0.0

        return battery_power