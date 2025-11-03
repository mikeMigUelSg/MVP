"""
House module - handles residential electricity consumption
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class House:
    """Residential house with electricity consumption"""

    def __init__(self, data_file: Optional[str] = None):
        """
        Args:
            data_file: Path to Excel/CSV file with historical consumption data
        """
        self.consumption_data = None
        self.total_consumption_kwh = 0.0

        if data_file:
            self.load_consumption_data(data_file)

    def load_consumption_data(self, file_path: str):
        """
        Load historical consumption data from Excel or CSV

        Expected columns: 'Data', 'Hora', 'Consumo registado (kW)'
        Note: Year is ignored - only month/day/hour/minute are used for matching
        """
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Create datetime index
            if 'Data' in df.columns and 'Hora' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                raise ValueError("Data must have 'Data' and 'Hora' columns or 'timestamp' column")

            # Get consumption column
            consumption_col = 'Consumo registado (kW)'
            if consumption_col not in df.columns:
                consumption_col = [col for col in df.columns if 'consumo' in col.lower()][0]

            # Lookup key
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute

            # Feb 29 -> Feb 28
            df.loc[(df['month'] == 2) & (df['day'] == 29), 'day'] = 28

            self.consumption_data = df[['month', 'day', 'hour', 'minute', consumption_col]].copy()
            self.consumption_data.rename(columns={consumption_col: 'consumption'}, inplace=True)

            # === NOVO: índices O(1) ===
            self._cons_lookup = {
                (int(r.month), int(r.day), int(r.hour), int(r.minute)): float(r.consumption)
                for r in self.consumption_data.itertuples(index=False)
            }
            # média por (mês, dia) como fallback
            from collections import defaultdict
            _day_buckets = defaultdict(list)
            for (m, d, h, mi), v in self._cons_lookup.items():
                _day_buckets[(m, d)].append(v)
            self._cons_day_mean = {k: (sum(vs)/len(vs)) for k, vs in _day_buckets.items()}

            print(f"Loaded consumption data: {len(self.consumption_data)} time steps")
            print(f"Original date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Average consumption: {self.consumption_data['consumption'].mean():.3f} kW")
            print(f"Data will be matched by month/day/hour, ignoring year")

        except Exception as e:
            print(f"Error loading consumption data: {e}")
            self.consumption_data = None
            self._cons_lookup = {}
            self._cons_day_mean = {}


    def get_consumption(self, timestamp: datetime) -> float:
        """
        O(1) lookup via dicionário com fallback média do dia.
        """
        if self.consumption_data is None or not hasattr(self, "_cons_lookup"):
            hour = timestamp.hour
            if 7 <= hour < 23:
                return 0
            else:
                return 0

        month = timestamp.month
        day = 28 if (timestamp.month == 2 and timestamp.day == 29) else timestamp.day
        key = (month, day, timestamp.hour, timestamp.minute)

        v = self._cons_lookup.get(key)
        if v is not None:
            return max(0.0, v)

        v = self._cons_day_mean.get((month, day))
        if v is not None:
            return max(0.0, v)

        return float(self.consumption_data["consumption"].mean())


    def get_consumption_forecast(self,
                                 start_time: datetime,
                                 n_steps: int,
                                 dt_minutes: int = 15) -> np.ndarray:
        """
        Get consumption forecast for planning horizon

        Args:
            start_time: Starting timestamp
            n_steps: Number of time steps
            dt_minutes: Time step duration in minutes

        Returns:
            Array of consumption forecast in kW
        """
        forecast = np.zeros(n_steps)
        current_time = start_time

        for i in range(n_steps):
            forecast[i] = self.get_consumption(current_time)
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
        consumption_kw = self.get_consumption(timestamp)
        energy_kwh = consumption_kw * dt_hours

        self.total_consumption_kwh += energy_kwh

        return {
            'power_kw': consumption_kw,
            'energy_kwh': energy_kwh,
            'timestamp': timestamp
        }

    def reset(self):
        """Reset house state"""
        self.total_consumption_kwh = 0.0

    def get_stats(self) -> dict:
        """Get house statistics"""
        return {
            'total_consumption_kwh': self.total_consumption_kwh
        }
