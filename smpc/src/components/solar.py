"""
Solar PV module - handles solar panel production
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SolarPanel:
    """Solar photovoltaic panel system"""

    def __init__(self,
                 capacity_kw: float,
                 data_file: Optional[str] = None,
                 scale_factor: float = 1.0,
                 preloaded_data: Optional[pd.DataFrame] = None):
        """
        Args:
            capacity_kw: Installed solar capacity in kW
            data_file: Path to CSV file with historical production data
            scale_factor: Scaling factor for solar production (1.0 = 100%, 0.5 = 50%, etc.)
            preloaded_data: Pre-loaded DataFrame (for performance optimization)
        """
        self.capacity_kw = capacity_kw
        self.scale_factor = scale_factor
        self.production_data = None
        self.total_production_kwh = 0.0

        if preloaded_data is not None:
            # Use pre-loaded data directly (should already be indexed)
            self.production_data = preloaded_data
        elif data_file:
            self.load_production_data(data_file)

    def load_production_data(self, file_path: str):
        """
        Load historical production data from CSV

        Expected format: timestamp;production_values
        Note: Year is ignored - only month/day/hour/minute are used for matching
        """
        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(file_path, sep=';', header=None)
            df.columns = ['timestamp'] + [f'pv_{i}' for i in range(len(df.columns) - 1)]

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

            # Create lookup key: (month, day, hour, minute)
            # This allows matching regardless of year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute

            # Handle leap year (Feb 29 -> Feb 28)
            df.loc[(df['month'] == 2) & (df['day'] == 29), 'day'] = 28

            # Use first PV system as default
            self.production_data = df[['month', 'day', 'hour', 'minute', 'pv_1']].copy()

            # Optimize lookups: Create multi-index for O(1) access
            self.production_data.set_index(['month', 'day', 'hour', 'minute'], inplace=True)
            self.production_data.sort_index(inplace=True)

            logger.debug(f"Loaded solar data: {len(self.production_data)} time steps")
            logger.debug(f"Original date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.debug(f"Data will be matched by month/day/hour, ignoring year")

        except Exception as e:
            print(f"Error loading solar data: {e}")
            self.production_data = None

    def get_production(self, timestamp: datetime) -> float:
        """
        Get solar production at given timestamp

        Args:
            timestamp: Datetime to query (year is ignored)

        Returns:
            Production in kW (scaled by capacity)
        """
        if self.production_data is None:
            # Simple model: no production at night, peak at noon
            hour = timestamp.hour
            if hour < 7 or hour > 19:
                return 0.0
            else:
                # Simple sine wave model
                hour_angle = (hour - 7) / 12 * np.pi
                return self.capacity_kw * self.scale_factor * np.sin(hour_angle)

        try:
            # Extract lookup key from timestamp (ignoring year)
            month = timestamp.month
            day = timestamp.day if not (timestamp.month == 2 and timestamp.day == 29) else 28
            hour = timestamp.hour
            minute = timestamp.minute

            # O(1) lookup using multi-index
            try:
                production = self.production_data.loc[(month, day, hour, minute), 'pv_1']
                # If multiple matches (shouldn't happen), take first
                if hasattr(production, 'iloc'):
                    production = production.iloc[0]
            except KeyError:
                # If no exact match, try to find closest time on same day
                try:
                    day_data = self.production_data.loc[(month, day)]
                    # Calculate time difference for all entries on this day
                    time_minutes = hour * 60 + minute
                    day_index = day_data.index
                    time_diffs = [abs((h * 60 + m) - time_minutes) for h, m in day_index]
                    min_idx = np.argmin(time_diffs)
                    production = day_data.iloc[min_idx]['pv_1']
                except (KeyError, IndexError):
                    return 0.0

            # Scale by capacity (assuming data is in W, converting to kW)
            scaled_production = production / 1000.0  # W to kW

            # Apply scale factor
            scaled_production *= self.scale_factor

            return max(0.0, min(scaled_production, self.capacity_kw * self.scale_factor))

        except Exception as e:
            print(f"Error getting production at {timestamp}: {e}")
            return 0.0

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
