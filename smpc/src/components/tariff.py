"""
Tariff module - handles different electricity pricing schemes
"""
from datetime import datetime
from typing import Dict, List
import numpy as np


class Tariff:
    """Base class for electricity tariffs"""

    def __init__(self, name: str):
        self.name = name

    def get_price(self, timestamp: datetime) -> float:
        """Get electricity price at given timestamp (€/kWh)"""
        raise NotImplementedError


class SimpleTariff(Tariff):
    """Simple tariff with constant price"""

    def __init__(self, price: float):
        """
        Args:
            price: Constant electricity price (€/kWh)
        """
        super().__init__("Simple")
        self.price = price

    def get_price(self, timestamp: datetime) -> float:
        return self.price


class BiHorariaTariff(Tariff):
    """Bi-horária tariff with peak and off-peak hours"""

    def __init__(self,
                 peak_price: float,
                 off_peak_price: float,
                 peak_hours: List[int] = None):
        """
        Args:
            peak_price: Price during peak hours (€/kWh)
            off_peak_price: Price during off-peak hours (€/kWh)
            peak_hours: List of hours considered peak (default: 9-24 on weekdays)
        """
        super().__init__("Bi-horária")
        self.peak_price = peak_price
        self.off_peak_price = off_peak_price

        # Default peak hours: 9h-24h on weekdays, 9h-14h and 20h-22h on weekends
        if peak_hours is None:
            self.peak_hours_weekday = list(range(9, 24))
            self.peak_hours_weekend = list(range(9, 14)) + list(range(20, 22))
        else:
            self.peak_hours_weekday = peak_hours
            self.peak_hours_weekend = peak_hours

    def get_price(self, timestamp: datetime) -> float:
        """Get price based on time of day and day of week"""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5  # Saturday=5, Sunday=6

        if is_weekend:
            is_peak = hour in self.peak_hours_weekend
        else:
            is_peak = hour in self.peak_hours_weekday

        return self.peak_price if is_peak else self.off_peak_price

    def get_prices_for_horizon(self, start_time: datetime, n_steps: int, dt_minutes: int = 15) -> np.ndarray:
        """
        Get price forecast for planning horizon

        Args:
            start_time: Starting timestamp
            n_steps: Number of time steps
            dt_minutes: Time step duration in minutes

        Returns:
            Array of prices for each time step
        """
        from datetime import timedelta

        prices = np.zeros(n_steps)
        current_time = start_time

        for i in range(n_steps):
            prices[i] = self.get_price(current_time)
            current_time += timedelta(minutes=dt_minutes)

        return prices
