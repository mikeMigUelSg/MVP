"""
ess/strategies.py - Control strategies for battery operation
Focus: Energy arbitrage with D+1 lookahead
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
from .battery import Battery


class ArbitrageStrategy:
    """
    Energy arbitrage strategy with D+1 price visibility.
    Charges during cheap hours and discharges during expensive hours.
    """
    
    def __init__(self, 
                 charge_threshold_percentile: float = 30,
                 discharge_threshold_percentile: float = 70,
                 min_price_spread: float = 20):  # EUR/MWh
        """
        Parameters
        ----------
        charge_threshold_percentile : float
            Charge when price is below this percentile of daily prices (0-100)
        discharge_threshold_percentile : float
            Discharge when price is above this percentile of daily prices (0-100)
        min_price_spread : float
            Minimum price difference (EUR/MWh) to trigger action
        """
        self.charge_threshold_percentile = charge_threshold_percentile
        self.discharge_threshold_percentile = discharge_threshold_percentile
        self.min_price_spread = min_price_spread / 1000  # Convert to EUR/kWh
        
    def decide_action(self,
                     current_time: datetime,
                     current_price: float,
                     consumption_kw: float,
                     battery: Battery,
                     prices_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Decide battery action based on current and future prices.
        
        Parameters
        ----------
        current_time : datetime
            Current timestamp
        current_price : float
            Current electricity price (EUR/kWh)
        consumption_kw : float
            Current consumption (kW)
        battery : Battery
            Battery object with current state
        prices_df : pd.DataFrame
            DataFrame with price forecasts (must include D+1)
        
        Returns
        -------
        Tuple[str, float]
            (action, power_kw) where action is 'charge', 'discharge', or 'idle'
        """
        # Get prices for current day and next day (D+1)
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_end = today_start + timedelta(days=2)
        
        # Filter prices for lookahead window
        lookahead_prices = prices_df.loc[current_time:tomorrow_end, 'price_eur_per_kwh']
        
        if len(lookahead_prices) < 24:  # Need at least 24 hours lookahead
            # Fallback to simple threshold if not enough data
            return self._simple_decision(current_price, consumption_kw, battery)
        
        # Calculate price thresholds based on lookahead window
        price_low = lookahead_prices.quantile(self.charge_threshold_percentile / 100)
        price_high = lookahead_prices.quantile(self.discharge_threshold_percentile / 100)
        
        # Check if spread is worth it
        if (price_high - price_low) < self.min_price_spread:
            return ('idle', 0)
        
        # Get battery limits
        max_charge_kw, max_discharge_kw = battery.get_max_power()
        
        # Decision logic
        if current_price <= price_low and max_charge_kw > 0:
            # Cheap hour - charge the battery
            # Consider future expensive hours to decide how much to charge
            future_high_hours = lookahead_prices[lookahead_prices >= price_high]
            if len(future_high_hours) > 0:
                # Charge aggressively if we see expensive hours coming
                charge_power = max_charge_kw
            else:
                # Charge moderately
                charge_power = max_charge_kw * 0.5
            return ('charge', charge_power)
        
        elif current_price >= price_high and max_discharge_kw > 0:
            # Expensive hour - discharge to cover consumption
            # Discharge to cover consumption, but not more than needed
            discharge_power = min(consumption_kw, max_discharge_kw)
            
            # Check if we have enough cheap hours ahead to recharge
            future_low_hours = lookahead_prices[lookahead_prices <= price_low]
            if len(future_low_hours) < 4:  # Less than 1 hour of cheap prices ahead
                # Conservative discharge - save some energy
                discharge_power *= 0.7
            
            return ('discharge', discharge_power)
        
        else:
            # Mid-price or battery constraints prevent action
            return ('idle', 0)
    
    def _simple_decision(self, current_price: float, consumption_kw: float, battery: Battery) -> Tuple[str, float]:
        """Fallback decision when not enough lookahead data."""
        max_charge_kw, max_discharge_kw = battery.get_max_power()
        
        # Simple thresholds (EUR/kWh)
        if current_price < 0.05 and max_charge_kw > 0:  # Below 50 EUR/MWh
            return ('charge', max_charge_kw)
        elif current_price > 0.15 and max_discharge_kw > 0:  # Above 150 EUR/MWh
            return ('discharge', min(consumption_kw, max_discharge_kw))
        else:
            return ('idle', 0)


class OptimalArbitrageStrategy:
    """
    Optimal arbitrage using perfect foresight for current and next day.
    Finds the best charge/discharge schedule to maximize profit.
    """
    
    def __init__(self, efficiency_penalty: float = 0.01):
        """
        Parameters
        ----------
        efficiency_penalty : float
            Small penalty to discourage unnecessary cycling (EUR/kWh)
        """
        
        self.efficiency_penalty = efficiency_penalty
        self.daily_schedule = None
        self.last_schedule_date = None
    
    def _compute_daily_schedule(self, 
                                start_time: datetime,
                                battery: Battery,
                                prices_df: pd.DataFrame,
                                consumption_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute optimal schedule for the next 48 hours using dynamic programming.
        Simplified version - could be enhanced with linear programming.
        """
        # Get 48 hours of data
        end_time = start_time + timedelta(hours=47, minutes=45)
        
        index = pd.date_range(start_time, end_time, freq='15min')
        prices = prices_df.loc[start_time:end_time, 'price_eur_per_kwh'].values
        consumption_kw = consumption_df.loc[start_time:end_time, 'kw'].values

        n_periods = len(prices)
        if n_periods == 0:
            return pd.DataFrame()

        # Pad consumption array if shorter than prices
        if len(consumption_kw) < n_periods:
            consumption_kw = np.pad(
                consumption_kw, (0, n_periods - len(consumption_kw)),
                constant_values=0
            )

        # Dynamic programming over discrete SOC levels
        dt = 0.25  # hours per period
        soc_min = battery.soc_min * battery.capacity_kwh
        soc_max = battery.soc_max * battery.capacity_kwh
        soc_step = battery.capacity_kwh * 0.05  # 5% increments
        soc_levels = np.arange(soc_min, soc_max + 1e-6, soc_step)
        n_levels = len(soc_levels)

        values = np.full((n_periods + 1, n_levels), -np.inf)
        actions = np.full((n_periods, n_levels), 'idle', dtype=object)
        powers = np.zeros((n_periods, n_levels))
        next_idx = np.zeros((n_periods, n_levels), dtype=int)

        start_soc = battery.soc_kwh
        start_level = int(np.clip(np.round((start_soc - soc_min) / soc_step), 0, n_levels - 1))
        values[-1, start_level] = 0

        for t in range(n_periods - 1, -1, -1):
            price = prices[t]
            cons_kw = consumption_kw[t]
            cons_energy = cons_kw * dt

            for i, soc in enumerate(soc_levels):
                best_val = values[t + 1, i]
                best_action = 'idle'
                best_power = 0.0
                best_next = i

                # Charge at maximum power
                max_charge = min(
                    battery.max_charge_kw * dt,
                    (soc_max - soc) / battery.efficiency_charge
                )
                if max_charge > 1e-6:
                    next_soc = soc + max_charge * battery.efficiency_charge
                    j = int(np.clip(np.round((next_soc - soc_min) / soc_step), 0, n_levels - 1))
                    profit = -price * max_charge - self.efficiency_penalty * max_charge
                    val = profit + values[t + 1, j]
                    if val > best_val:
                        best_val = val
                        best_action = 'charge'
                        best_power = max_charge / dt
                        best_next = j

                # Discharge at maximum power (limited by consumption)
                max_discharge = min(
                    battery.max_discharge_kw * dt,
                    (soc - soc_min) * battery.efficiency_discharge,
                    cons_energy
                )
                if max_discharge > 1e-6:
                    next_soc = soc - max_discharge / battery.efficiency_discharge
                    j = int(np.clip(np.round((next_soc - soc_min) / soc_step), 0, n_levels - 1))
                    profit = price * max_discharge - self.efficiency_penalty * max_discharge
                    val = profit + values[t + 1, j]
                    if val > best_val:
                        best_val = val
                        best_action = 'discharge'
                        best_power = max_discharge / dt
                        best_next = j

                values[t, i] = best_val
                actions[t, i] = best_action
                powers[t, i] = best_power
                next_idx[t, i] = best_next

        # Build schedule
        schedule = pd.DataFrame(index=index)
        schedule['price'] = prices
        schedule['consumption_kw'] = consumption_kw
        schedule['action'] = 'idle'
        schedule['power_kw'] = 0.0

        level = start_level
        for t in range(n_periods):
            schedule.iloc[t, schedule.columns.get_loc('action')] = actions[t, level]
            schedule.iloc[t, schedule.columns.get_loc('power_kw')] = powers[t, level]
            level = next_idx[t, level]

        return schedule
    
    def decide_action(self,
                     current_time: datetime,
                     current_price: float,
                     consumption_kw: float,
                     battery: Battery,
                     prices_df: pd.DataFrame,
                     consumption_df: Optional[pd.DataFrame] = None) -> Tuple[str, float]:
        """
        Get action from pre-computed optimal schedule.
        """
        # Recompute schedule at midnight or if not computed yet
        current_date = current_time.date()
        if self.daily_schedule is None or self.last_schedule_date != current_date:
            if consumption_df is not None:
                self.daily_schedule = self._compute_daily_schedule(
                    current_time.replace(hour=0, minute=0, second=0, microsecond=0),
                    battery, prices_df, consumption_df
                )
                self.last_schedule_date = current_date
            else:
                # Fallback to simple arbitrage
                return ArbitrageStrategy().decide_action(
                    current_time, current_price, consumption_kw, battery, prices_df
                )
        
        # Get action from schedule
        if current_time in self.daily_schedule.index:
            row = self.daily_schedule.loc[current_time]
            
            # Validate action against current battery state
            max_charge_kw, max_discharge_kw = battery.get_max_power()
            
            if row['action'] == 'charge':
                power = min(row['power_kw'], max_charge_kw)
                return ('charge', power) if power > 0 else ('idle', 0)
            elif row['action'] == 'discharge':
                power = min(row['power_kw'], max_discharge_kw)
                return ('discharge', power) if power > 0 else ('idle', 0)
        
        return ('idle', 0)