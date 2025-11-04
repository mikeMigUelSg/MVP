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
                # Try alternative column names
                consumption_col = [col for col in df.columns if 'consumo' in col.lower()][0]

            # Create lookup key: (month, day, hour, minute)
            # This allows matching regardless of year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute

            # Handle leap year (Feb 29 -> Feb 28)
            df.loc[(df['month'] == 2) & (df['day'] == 29), 'day'] = 28

            self.consumption_data = df[['month', 'day', 'hour', 'minute', consumption_col]].copy()
            self.consumption_data.rename(columns={consumption_col: 'consumption'}, inplace=True)

            # Optimize lookups: Create multi-index for O(1) access
            self.consumption_data.set_index(['month', 'day', 'hour', 'minute'], inplace=True)
            self.consumption_data.sort_index(inplace=True)

            # Pre-compute mean for fallback
            self._mean_consumption = self.consumption_data['consumption'].mean()

            print(f"Loaded consumption data: {len(self.consumption_data)} time steps")
            print(f"Original date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Average consumption: {self.consumption_data['consumption'].mean():.3f} kW")
            print(f"Data will be matched by month/day/hour, ignoring year")

        except Exception as e:
            print(f"Error loading consumption data: {e}")
            self.consumption_data = None

    def get_consumption(self, timestamp: datetime) -> float:
        """
        Get consumption at given timestamp

        Args:
            timestamp: Datetime to query (year is ignored)

        Returns:
            Consumption in kW
        """
        if self.consumption_data is None:
            # Simple model: average consumption with daily pattern
            hour = timestamp.hour
            # Higher consumption during day, lower at night
            if 7 <= hour < 23:
                base_load = 0.5
                variable_load = 1.0
            else:
                base_load = 0.3
                variable_load = 0.2

            return base_load + np.random.uniform(0, variable_load)

        try:
            # Extract lookup key from timestamp (ignoring year)
            month = timestamp.month
            day = timestamp.day if not (timestamp.month == 2 and timestamp.day == 29) else 28
            hour = timestamp.hour
            minute = timestamp.minute

            # O(1) lookup using multi-index
            try:
                consumption = self.consumption_data.loc[(month, day, hour, minute), 'consumption']
                # If multiple matches (shouldn't happen), take first
                if hasattr(consumption, 'iloc'):
                    consumption = consumption.iloc[0]
            except KeyError:
                # If no exact match, try to find closest time on same day
                try:
                    day_data = self.consumption_data.loc[(month, day)]
                    # Calculate time difference for all entries on this day
                    time_minutes = hour * 60 + minute
                    day_index = day_data.index
                    time_diffs = [abs((h * 60 + m) - time_minutes) for h, m in day_index]
                    min_idx = np.argmin(time_diffs)
                    consumption = day_data.iloc[min_idx]['consumption']
                except (KeyError, IndexError):
                    # Fallback to pre-computed mean
                    consumption = self._mean_consumption

            return max(0.0, consumption)

        except Exception as e:
            print(f"Error getting consumption at {timestamp}: {e}")
            return 0.0

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
