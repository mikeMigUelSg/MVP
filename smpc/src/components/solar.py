"""
Solar PV module - handles solar panel production
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class SolarPanel:
    """Solar photovoltaic panel system"""

    def __init__(self,
                 capacity_kw: float,
                 data_file: Optional[str] = None):
        """
        Args:
            capacity_kw: Installed solar capacity in kW
            data_file: Path to CSV file with historical production data
        """
        self.capacity_kw = capacity_kw
        self.production_data = None
        self.total_production_kwh = 0.0

        if data_file:
            self.load_production_data(data_file)

    def load_production_data(self, file_path: str):
        """
        Load historical production data from CSV (timestamp;pv_1;pv_2;...)
        Year is ignored for matching.
        """
        try:
            df = pd.read_csv(file_path, sep=';', header=None)
            df.columns = ['timestamp'] + [f'pv_{i}' for i in range(len(df.columns) - 1)]
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df.loc[(df['month'] == 2) & (df['day'] == 29), 'day'] = 28

            self.production_data = df[['month', 'day', 'hour', 'minute', 'pv_1']].copy()

            # === NOVO: índice O(1) (em kW já convertido) ===
            self._pv_lookup = {
                (int(r.month), int(r.day), int(r.hour), int(r.minute)): float(r.pv_1) / 1000.0
                for r in self.production_data.itertuples(index=False)
            }

            print(f"Loaded solar data: {len(self.production_data)} time steps")
            print(f"Original date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Data will be matched by month/day/hour, ignoring year")

        except Exception as e:
            print(f"Error loading solar data: {e}")
            self.production_data = None
            self._pv_lookup = {}


    def get_production(self, timestamp: datetime) -> float:
        """
        O(1) lookup via dicionário, limitado pela capacidade instalada.
        """
        if self.production_data is None or not hasattr(self, "_pv_lookup"):
            hour = timestamp.hour
            if hour < 7 or hour > 19:
                return 0.0
            hour_angle = (hour - 7) / 12 * np.pi
            return max(0.0, min(self.capacity_kw * np.sin(hour_angle), self.capacity_kw))

        month = timestamp.month
        day = 28 if (timestamp.month == 2 and timestamp.day == 29) else timestamp.day
        key = (month, day, timestamp.hour, timestamp.minute)

        v = self._pv_lookup.get(key, 0.0)
        return max(0.0, min(v, self.capacity_kw))


    def get_production_forecast(self,
                                start_time: datetime,
                                n_steps: int,
                                dt_minutes: int = 15) -> np.ndarray:
        """
        Get production forecast for planning horizon

        Args:
            start_time: Starting timestamp
            n_steps: Number of time steps
            dt_minutes: Time step duration in minutes

        Returns:
            Array of production forecast in kW
        """
        forecast = np.zeros(n_steps)
        current_time = start_time

        for i in range(n_steps):
            forecast[i] = self.get_production(current_time)
            current_time += timedelta(minutes=dt_minutes)

        return forecast

    def step(self, timestamp: datetime, dt_hours: float) -> dict:
        """
        Simulate one time step

        Args:
            timestamp: Current timestamp
            dt_hours: Time step duration in hours

        Returns:
            Dictionary with step results
        """
        production_kw = self.get_production(timestamp)
        energy_kwh = production_kw * dt_hours

        self.total_production_kwh += energy_kwh

        return {
            'power_kw': production_kw,
            'energy_kwh': energy_kwh,
            'timestamp': timestamp
        }

    def reset(self):
        """Reset solar panel state"""
        self.total_production_kwh = 0.0

    def get_stats(self) -> dict:
        """Get solar panel statistics"""
        return {
            'total_production_kwh': self.total_production_kwh,
            'capacity_kw': self.capacity_kw
        }
