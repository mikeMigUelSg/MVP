"""
ess/strategies.py - FIXED VERSION - Handles time zone transitions and index issues
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta
from .battery import Battery
import warnings

# Try to import optimization libraries
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("WARNING: PuLP not available. Install with: pip install pulp")


class ArbitrageStrategy:
    """
    Enhanced energy arbitrage strategy with D+1 price visibility.
    """
    
    def __init__(self, 
                 charge_threshold_percentile: float = 30,
                 discharge_threshold_percentile: float = 70,
                 min_price_spread: float = 20,  # EUR/MWh
                 lookahead_hours: int = 24):
        """
        Parameters
        ----------
        charge_threshold_percentile : float
            Charge when price is below this percentile (0-100)
        discharge_threshold_percentile : float
            Discharge when price is above this percentile (0-100)
        min_price_spread : float
            Minimum price difference (EUR/MWh) to trigger action
        lookahead_hours : int
            Hours to look ahead for price analysis
        """
        self.charge_threshold_percentile = charge_threshold_percentile
        self.discharge_threshold_percentile = discharge_threshold_percentile
        self.min_price_spread = min_price_spread / 1000  # Convert to EUR/kWh
        self.lookahead_hours = lookahead_hours
        
    def decide_action(self,
                     current_time: datetime,
                     current_price: float,
                     consumption_kw: float,
                     battery: Battery,
                     prices_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Decide battery action based on current and future prices.
        """
        # Get lookahead window
        lookahead_end = current_time + timedelta(hours=self.lookahead_hours)
        
        try:
            # Filter prices for lookahead window - handle index issues
            lookahead_prices = self._safe_slice_prices(prices_df, current_time, lookahead_end)
            
            if len(lookahead_prices) < 8:  # Need at least 2 hours of data
                return self._simple_decision(current_price, consumption_kw, battery)
            
            # Calculate dynamic thresholds
            price_low = lookahead_prices.quantile(self.charge_threshold_percentile / 100)
            price_high = lookahead_prices.quantile(self.discharge_threshold_percentile / 100)
            
            # Check if spread is worth it
            if (price_high - price_low) < self.min_price_spread:
                return ('idle', 0)
            
            # Get battery constraints
            max_charge_kw, max_discharge_kw = battery.get_max_power(0.25)
            
            # Enhanced decision logic
            if current_price <= price_low and max_charge_kw > 0.1:
                # Low price - consider charging
                return self._decide_charge(current_price, price_low, price_high, 
                                         lookahead_prices, battery, max_charge_kw)
            
            elif current_price >= price_high and max_discharge_kw > 0.1:
                # High price - consider discharging
                return self._decide_discharge(current_price, price_low, price_high,
                                            lookahead_prices, battery, consumption_kw, max_discharge_kw)
            
            else:
                return ('idle', 0)
                
        except Exception as e:
            print(f"Error in arbitrage decision at {current_time}: {e}")
            return self._simple_decision(current_price, consumption_kw, battery)
    
    def _safe_slice_prices(self, prices_df: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.Series:
        """Safely slice price data handling index issues."""
        try:
            # Ensure index is sorted and has no duplicates
            if not prices_df.index.is_monotonic_increasing:
                prices_df = prices_df.sort_index()
            
            # Remove duplicates
            prices_df = prices_df[~prices_df.index.duplicated(keep='first')]
            
            # Get the slice
            mask = (prices_df.index >= start_time) & (prices_df.index <= end_time)
            result = prices_df.loc[mask, 'price_eur_per_kwh']
            
            return result
            
        except Exception as e:
            print(f"Error in _safe_slice_prices: {e}")
            # Return empty series as fallback
            return pd.Series(dtype=float)
    
    def _decide_charge(self, current_price: float, price_low: float, price_high: float,
                      lookahead_prices: pd.Series, battery: Battery, max_charge_kw: float) -> Tuple[str, float]:
        """Enhanced charging decision logic."""
        
        # Count future expensive hours
        future_high_hours = lookahead_prices[lookahead_prices >= price_high]
        
        # Calculate charging urgency based on price position
        price_range = lookahead_prices.max() - lookahead_prices.min()
        if price_range > 0:
            price_percentile = (current_price - lookahead_prices.min()) / price_range
        else:
            price_percentile = 0.5
        
        # Charging power based on multiple factors
        if len(future_high_hours) >= 4:  # Plenty of expensive hours ahead
            charge_factor = 1.0
        elif len(future_high_hours) >= 2:
            charge_factor = 0.8
        else:
            charge_factor = 0.5
            
        # Adjust based on price position (lower price = more aggressive charging)
        charge_factor *= (1 - price_percentile * 0.5)
        
        # Adjust based on current SOC (lower SOC = more aggressive charging)
        soc_factor = 1 + (1 - battery.soc) * 0.5
        charge_factor *= soc_factor
        
        charge_power = max_charge_kw * charge_factor
        return ('charge', max(0.1, charge_power))
    
    def _decide_discharge(self, current_price: float, price_low: float, price_high: float,
                         lookahead_prices: pd.Series, battery: Battery, 
                         consumption_kw: float, max_discharge_kw: float) -> Tuple[str, float]:
        """Enhanced discharging decision logic."""
        
        # Count future cheap hours for recharging
        future_low_hours = lookahead_prices[lookahead_prices <= price_low]
        
        # Base discharge to cover consumption
        base_discharge = min(consumption_kw * 1.2, max_discharge_kw)  # 20% buffer
        
        # Adjust based on recharge opportunities
        if len(future_low_hours) >= 6:  # Plenty of cheap hours to recharge
            discharge_factor = 1.0
        elif len(future_low_hours) >= 3:
            discharge_factor = 0.8
        else:
            discharge_factor = 0.6  # Conservative - preserve battery
        
        # Adjust based on price position (higher price = more aggressive discharge)
        price_range = lookahead_prices.max() - lookahead_prices.min()
        if price_range > 0:
            price_percentile = (current_price - lookahead_prices.min()) / price_range
            discharge_factor *= (0.5 + price_percentile * 0.5)
        
        discharge_power = base_discharge * discharge_factor
        return ('discharge', max(0.1, min(discharge_power, max_discharge_kw)))
    
    def _simple_decision(self, current_price: float, consumption_kw: float, battery: Battery) -> Tuple[str, float]:
        """Fallback decision when not enough lookahead data."""
        max_charge_kw, max_discharge_kw = battery.get_max_power(0.25)
        
        # Simple thresholds (EUR/kWh)
        if current_price < 0.05 and max_charge_kw > 0.1:  # Below 50 EUR/MWh
            return ('charge', max_charge_kw * 0.8)
        elif current_price > 0.15 and max_discharge_kw > 0.1:  # Above 150 EUR/MWh
            return ('discharge', min(consumption_kw * 1.1, max_discharge_kw))
        else:
            return ('idle', 0)


class OptimalArbitrageStrategy:
    """
    Fixed optimal arbitrage strategy with robust handling of time series issues.
    """
    
    def __init__(self, optimization_window_hours: int = 24, use_simple_optimization: bool = False):
        """
        Parameters
        ----------
        optimization_window_hours : int
            Optimization horizon in hours (default: 48h)
        use_simple_optimization : bool
            Use simplified optimization if PuLP is not available
        """
        self.optimization_window_hours = optimization_window_hours
        self.use_simple_optimization = use_simple_optimization or not HAS_PULP
        self.current_schedule = None
        self.schedule_start_time = None
        
        if not HAS_PULP and not use_simple_optimization:
            print("WARNING: PuLP not available, falling back to simple optimization")
            self.use_simple_optimization = True
    
    def _clean_and_prepare_data(self, start_time: datetime, 
                              prices_df: pd.DataFrame,
                              consumption_df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """
        Clean and prepare data for optimization, handling time zone issues.
        """
        try:
            # Create clean time index avoiding DST issues
            end_time = start_time + timedelta(hours=self.optimization_window_hours - 0.25)
            
            # Create time index with proper frequency
            time_index = pd.date_range(start_time, end_time, freq='15min')
            n_periods = len(time_index)
            
            if n_periods == 0:
                raise ValueError("Empty time index")
            
            # Clean price data
            prices_clean = prices_df.copy()
            if not prices_clean.index.is_monotonic_increasing:
                prices_clean = prices_clean.sort_index()
            prices_clean = prices_clean[~prices_clean.index.duplicated(keep='first')]
            
            # Clean consumption data
            consumption_clean = consumption_df.copy()
            if not consumption_clean.index.is_monotonic_increasing:
                consumption_clean = consumption_clean.sort_index()
            consumption_clean = consumption_clean[~consumption_clean.index.duplicated(keep='first')]
            
            # Reindex both series to our clean time index
            prices_series = prices_clean['price_eur_per_kwh'].reindex(time_index, method='ffill')
            consumption_series = consumption_clean['kw'].reindex(time_index, method='ffill')
            
            # Fill any remaining NaN values
            prices_series = prices_series.fillna(method='ffill').fillna(0.1)  # Default price
            consumption_series = consumption_series.fillna(method='ffill').fillna(0)  # Default consumption
            
            # Convert to numpy arrays
            prices = prices_series.values
            consumption_kw = consumption_series.values
            
            # Validate arrays
            if len(prices) != n_periods:
                prices = np.resize(prices, n_periods)
            if len(consumption_kw) != n_periods:
                consumption_kw = np.resize(consumption_kw, n_periods)
            
            # Replace any invalid values
            prices = np.where(np.isfinite(prices), prices, 0.1)
            consumption_kw = np.where(np.isfinite(consumption_kw), consumption_kw, 0)
            
            return time_index, prices, consumption_kw
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            raise
    
    def _solve_with_pulp(self, time_index: pd.DatetimeIndex, prices: np.ndarray, 
                        consumption_kw: np.ndarray, battery: Battery) -> Optional[Dict]:
        """Solve optimization using PuLP linear programming."""
        try:
            n_periods = len(time_index)
            dt = 0.25  # 15 minutes in hours
            
            # Create optimization problem
            prob = pulp.LpProblem("Battery_Arbitrage", pulp.LpMaximize)
            
            # Decision variables
            charge_power = pulp.LpVariable.dicts("charge", range(n_periods), 
                                               lowBound=0, upBound=battery.max_charge_kw)
            discharge_power = pulp.LpVariable.dicts("discharge", range(n_periods),
                                                  lowBound=0, upBound=battery.max_discharge_kw)
            soc_kwh = pulp.LpVariable.dicts("soc", range(n_periods + 1),
                                          lowBound=battery.min_energy_kwh,
                                          upBound=battery.max_energy_kwh)
            grid_import = pulp.LpVariable.dicts("grid", range(n_periods), lowBound=0)
            
            # Objective: Minimize electricity cost
            total_cost = pulp.lpSum([grid_import[t] * prices[t] * dt for t in range(n_periods)])
            prob += -total_cost  # Maximize negative cost (minimize cost)
            
            # Constraints
            
            # Initial SOC
            prob += soc_kwh[0] == battery.soc_kwh
            
            # SOC dynamics
            for t in range(n_periods):
                prob += soc_kwh[t+1] == (soc_kwh[t] + 
                                       charge_power[t] * dt * battery.efficiency_charge - 
                                       discharge_power[t] * dt / battery.efficiency_discharge)
            
            # Energy balance
            for t in range(n_periods):
                prob += grid_import[t] == (consumption_kw[t] * dt + 
                                         charge_power[t] * dt - 
                                         discharge_power[t] * dt)
            
            # Cannot discharge more than available
            for t in range(n_periods):
                prob += discharge_power[t] * dt <= consumption_kw[t] * dt + charge_power[t] * dt
            
            # Solve with timeout
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)  # 30 second timeout
            status = prob.solve(solver)
            
            if status != pulp.LpStatusOptimal:
                print(f"Optimization status: {pulp.LpStatus[status]}")
                return None
            
            # Extract solution
            schedule = {
                'time_index': time_index,
                'charge_power': [max(0, charge_power[t].varValue or 0) for t in range(n_periods)],
                'discharge_power': [max(0, discharge_power[t].varValue or 0) for t in range(n_periods)],
                'soc_kwh': [max(battery.min_energy_kwh, min(battery.max_energy_kwh, soc_kwh[t].varValue or battery.soc_kwh)) for t in range(n_periods + 1)],
                'grid_import': [max(0, grid_import[t].varValue or 0) for t in range(n_periods)],
                'prices': prices,
                'consumption_kw': consumption_kw,
                'optimization_status': 'optimal'
            }
            
            return schedule
            
        except Exception as e:
            print(f"PuLP optimization error: {e}")
            return None
    
    def _solve_with_simple_algorithm(self, time_index: pd.DatetimeIndex, prices: np.ndarray,
                                   consumption_kw: np.ndarray, battery: Battery) -> Dict:
        """Simple greedy optimization as fallback."""
        n_periods = len(time_index)
        dt = 0.25
        
        # Sort periods by price for greedy approach
        price_order = np.argsort(prices)
        
        # Initialize solution
        charge_power = np.zeros(n_periods)
        discharge_power = np.zeros(n_periods)
        
        # Simulate battery state
        soc_kwh = np.zeros(n_periods + 1)
        soc_kwh[0] = battery.soc_kwh
        
        # Greedy charging in cheapest periods
        available_capacity = battery.max_energy_kwh - battery.soc_kwh
        for t in price_order[:n_periods//3]:  # Charge in cheapest 1/3 of periods
            if available_capacity > 0:
                max_charge = min(battery.max_charge_kw, available_capacity / dt)
                charge_power[t] = max_charge
                available_capacity -= max_charge * dt * battery.efficiency_charge
        
        # Greedy discharging in most expensive periods
        available_discharge = battery.soc_kwh - battery.min_energy_kwh
        for t in price_order[-n_periods//3:]:  # Discharge in most expensive 1/3 of periods
            if available_discharge > 0:
                max_discharge = min(
                    battery.max_discharge_kw,
                    consumption_kw[t],
                    available_discharge * battery.efficiency_discharge / dt
                )
                discharge_power[t] = max_discharge
                available_discharge -= max_discharge * dt / battery.efficiency_discharge
        
        # Simulate SOC evolution
        for t in range(n_periods):
            soc_kwh[t+1] = soc_kwh[t] + (charge_power[t] * dt * battery.efficiency_charge - 
                                       discharge_power[t] * dt / battery.efficiency_discharge)
            soc_kwh[t+1] = np.clip(soc_kwh[t+1], battery.min_energy_kwh, battery.max_energy_kwh)
        
        return {
            'time_index': time_index,
            'charge_power': charge_power.tolist(),
            'discharge_power': discharge_power.tolist(),
            'soc_kwh': soc_kwh.tolist(),
            'grid_import': (consumption_kw * dt + charge_power * dt - discharge_power * dt).tolist(),
            'prices': prices,
            'consumption_kw': consumption_kw,
            'optimization_status': 'simple_greedy'
        }
    
    def _solve_optimal_schedule(self, start_time: datetime, battery: Battery,
                              prices_df: pd.DataFrame, consumption_df: pd.DataFrame) -> Optional[Dict]:
        """Main optimization function with robust error handling."""
        try:
            # Clean and prepare data
            time_index, prices, consumption_kw = self._clean_and_prepare_data(
                start_time, prices_df, consumption_df
            )
            
            # Try PuLP optimization first
            if not self.use_simple_optimization:
                schedule = self._solve_with_pulp(time_index, prices, consumption_kw, battery)
                if schedule is not None:
                    return schedule
                else:
                    print("PuLP optimization failed, falling back to simple algorithm")
            
            # Fallback to simple optimization
            schedule = self._solve_with_simple_algorithm(time_index, prices, consumption_kw, battery)
            return schedule
            
        except Exception as e:
            print(f"Complete optimization failure: {e}")
            return None
    
    def decide_action(self, current_time: datetime, current_price: float, consumption_kw: float,
                     battery: Battery, prices_df: pd.DataFrame, 
                     consumption_df: Optional[pd.DataFrame] = None) -> Tuple[str, float]:
        """Get action with improved reliability."""
        
        # Check if we need to recompute the schedule
        need_recompute = (
            self.current_schedule is None or
            self.schedule_start_time is None or
            current_time < self.schedule_start_time or
            current_time >= self.schedule_start_time + timedelta(hours=12) or  # Recompute more frequently
            consumption_df is None
        )
        
        if need_recompute and consumption_df is not None:
            # Compute new schedule with more conservative start time
            schedule_start = current_time.replace(second=0, microsecond=0)
            # Round to nearest 15 minutes
            minute = (schedule_start.minute // 15) * 15
            schedule_start = schedule_start.replace(minute=minute)
            
            try:
                print(f"Computing optimal schedule starting from {schedule_start}")
                self.current_schedule = self._solve_optimal_schedule(
                    schedule_start, battery, prices_df, consumption_df
                )
                self.schedule_start_time = schedule_start
                
                if self.current_schedule is None:
                    print("Optimization failed completely")
                else:
                    status = self.current_schedule.get('optimization_status', 'unknown')
                    print(f"Optimization completed with status: {status}")
                    
            except Exception as e:
                print(f"Error computing schedule: {e}")
                self.current_schedule = None
        
        # Fallback to arbitrage if optimization failed
        if self.current_schedule is None:
            return ArbitrageStrategy().decide_action(
                current_time, current_price, consumption_kw, battery, prices_df
            )
        
        # Extract action from schedule
        try:
            time_index = self.current_schedule['time_index']
            
            # Find closest time point
            time_diffs = [(abs((t - current_time).total_seconds()), i) for i, t in enumerate(time_index)]
            if not time_diffs:
                raise ValueError("Empty time index in schedule")
            
            _, closest_idx = min(time_diffs)
            
            # Get planned actions with bounds checking
            if closest_idx >= len(self.current_schedule['charge_power']):
                closest_idx = len(self.current_schedule['charge_power']) - 1
            
            planned_charge = self.current_schedule['charge_power'][closest_idx]
            planned_discharge = self.current_schedule['discharge_power'][closest_idx]
            
            # Validate against current battery constraints
            max_charge_kw, max_discharge_kw = battery.get_max_power(0.25)
            
            # Decide action with validation
            if planned_charge > 0.01 and max_charge_kw > 0.01:
                power = min(planned_charge, max_charge_kw)
                return ('charge', power)
            elif planned_discharge > 0.01 and max_discharge_kw > 0.01:
                power = min(planned_discharge, max_discharge_kw)
                return ('discharge', power)
            else:
                return ('idle', 0)
                
        except Exception as e:
            print(f"Error extracting action from schedule: {e}")
            # Final fallback to arbitrage
            return ArbitrageStrategy().decide_action(
                current_time, current_price, consumption_kw, battery, prices_df
            )


class PeakShavingStrategy:
    """
    Peak shaving strategy to reduce maximum demand.
    """
    
    def __init__(self, target_peak_kw: float, peak_window_hours: int = 4):
        """
        Parameters
        ----------
        target_peak_kw : float
            Target maximum power from grid
        peak_window_hours : int
            Rolling window for peak detection
        """
        self.target_peak_kw = target_peak_kw
        self.peak_window_hours = peak_window_hours
        self.recent_consumption = []
        
    def decide_action(self, current_time: datetime, current_price: float, consumption_kw: float,
                     battery: Battery, prices_df: pd.DataFrame) -> Tuple[str, float]:
        """Peak shaving logic."""
        max_charge_kw, max_discharge_kw = battery.get_max_power(0.25)
        
        # Track recent consumption for peak detection
        self.recent_consumption.append((current_time, consumption_kw))
        
        # Keep only recent data
        cutoff_time = current_time - timedelta(hours=self.peak_window_hours)
        self.recent_consumption = [(t, p) for t, p in self.recent_consumption if t >= cutoff_time]
        
        # Check if we're approaching peak
        if consumption_kw > self.target_peak_kw and max_discharge_kw > 0.1:
            # Discharge to reduce grid import
            required_discharge = min(consumption_kw - self.target_peak_kw, max_discharge_kw)
            return ('discharge', required_discharge)
        
        # Charge during low consumption if prices are reasonable
        elif (consumption_kw < self.target_peak_kw * 0.3 and 
              current_price < 0.1 and  # Below 100 EUR/MWh
              max_charge_kw > 0.1):
            # Gentle charging during low demand
            charge_power = min(max_charge_kw * 0.5, self.target_peak_kw - consumption_kw)
            return ('charge', max(0.1, charge_power))
        
        else:
            return ('idle', 0)