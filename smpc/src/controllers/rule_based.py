"""
Rule-based controller - simple heuristic battery control
"""
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.battery import Battery
    from ..components.tariff import Tariff


class RuleBasedController:
    """
    Simple rule-based controller for battery management

    Rules:
    1. If excess solar production -> charge battery
    2. If net demand and high price -> discharge battery (if available)
    3. If net demand and low price -> use grid
    4. Try to maintain battery between min/max SOC
    """

    def __init__(self,
                 high_price_threshold: float = 0.15,
                 low_soc_threshold: float = 0.3,
                 high_soc_threshold: float = 0.8):
        """
        Args:
            high_price_threshold: Price threshold to consider "high" (€/kWh)
            low_soc_threshold: SOC below which to avoid discharge
            high_soc_threshold: SOC above which to avoid charge
        """
        self.high_price_threshold = high_price_threshold
        self.low_soc_threshold = low_soc_threshold
        self.high_soc_threshold = high_soc_threshold

        self.name = "Rule-Based"

    def compute_action(self,
                      timestamp: datetime,
                      solar_power: float,
                      load_power: float,
                      battery: 'Battery',
                      tariff: 'Tariff',
                      dt_hours: float) -> float:
        """
        Compute battery power action based on rules

        Args:
            timestamp: Current timestamp
            solar_power: Solar production in kW
            load_power: House consumption in kW
            battery: Battery object
            tariff: Tariff object
            dt_hours: Time step in hours

        Returns:
            Battery power in kW (positive = charge, negative = discharge)
        """
        # Get current state
        soc = battery.get_soc()
        current_price = tariff.get_price(timestamp)

        # Net power: positive = excess solar, negative = net demand
        net_power = solar_power - load_power

        # Rule 1: Excess solar production
        if net_power > 0:
            # Charge battery if not full
            if soc < self.high_soc_threshold:
                # Try to charge with all excess
                max_charge = battery.get_available_charge_power(dt_hours)
                battery_power = min(net_power, max_charge)
            else:
                # Battery full, export to grid
                battery_power = 0.0

        # Rule 2: Net demand
        else:
            demand = -net_power

            # High price period -> discharge battery if possible
            if current_price >= self.high_price_threshold and soc > self.low_soc_threshold:
                max_discharge = battery.get_available_discharge_power(dt_hours)
                # Discharge to cover as much demand as possible
                battery_power = -min(demand, max_discharge)

            # Low price period -> use grid, keep battery for later
            else:
                battery_power = 0.0

        return battery_power
