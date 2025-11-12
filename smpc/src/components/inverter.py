"""
Inverter module - handles DC to AC power conversion with power limits
"""


class Inverter:
    """
    Bidirectional inverter component for DC↔AC power conversion

    The inverter has a maximum power rating that limits:
    - DC→AC: Power from solar (DC) + battery discharge (DC) to AC side (load + grid)
    - AC→DC: Power from grid (AC) to battery charge (DC)

    Modern hybrid inverters typically support bidirectional operation, allowing both
    power export (DC→AC) and battery charging (AC→DC) through the same device.
    """

    def __init__(self, max_power_kw: float, price: float = 0.0):
        """
        Args:
            max_power_kw: Maximum AC output power in kW (e.g., 3.0 or 5.0)
            price: Purchase price of inverter in €
        """
        self.max_power_kw = max_power_kw
        self.price = price

        # Tracking
        self.total_energy_converted_kwh = 0.0
        self.peak_power_used_kw = 0.0

    def limit_ac_power(self, requested_power_kw: float) -> float:
        """
        Limit AC power output to inverter rating

        Args:
            requested_power_kw: Requested AC power in kW (positive)

        Returns:
            Limited AC power in kW
        """
        return min(requested_power_kw, self.max_power_kw)

    def step(self, ac_power_kw: float, dt_hours: float) -> dict:
        """
        Execute one time step

        Args:
            ac_power_kw: AC power flowing through inverter in kW (positive)
            dt_hours: Time step duration in hours

        Returns:
            Dictionary with step results
        """
        # Track energy and peak power
        self.total_energy_converted_kwh += ac_power_kw * dt_hours
        self.peak_power_used_kw = max(self.peak_power_used_kw, ac_power_kw)

        return {
            'power_kw': ac_power_kw,
            'utilization': ac_power_kw / self.max_power_kw if self.max_power_kw > 0 else 0.0
        }

    def reset(self):
        """Reset inverter statistics"""
        self.total_energy_converted_kwh = 0.0
        self.peak_power_used_kw = 0.0

    def get_stats(self) -> dict:
        """Get inverter statistics"""
        return {
            'max_power_kw': self.max_power_kw,
            'total_energy_converted_kwh': self.total_energy_converted_kwh,
            'peak_power_used_kw': self.peak_power_used_kw,
            'peak_utilization': self.peak_power_used_kw / self.max_power_kw if self.max_power_kw > 0 else 0.0,
            'price': self.price
        }
