import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from .battery import Battery
from .strategies import ArbitrageStrategy, OptimalArbitrageStrategy
import warnings


class EnergyArbitrageSimulator:
    """
    CORRECTED simulator for battery energy arbitrage operations.
    Updated with simplified energy VAT logic - no 200 kWh limit.
    """
    
    def __init__(self,
                 battery: Battery,
                 strategy: Optional[object] = None,
                 time_step_minutes: int = 15,
                 validation_mode: bool = True):
        """
        Parameters
        ----------
        battery : Battery
            Battery model instance
        strategy : object
            Strategy instance (ArbitrageStrategy or OptimalArbitrageStrategy)
        time_step_minutes : int
            Simulation time step in minutes (default: 15)
        validation_mode : bool
            Enable data validation and consistency checks
        """
        self.battery = battery
        self.strategy = strategy or ArbitrageStrategy()
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60
        self.validation_mode = validation_mode
        
        # Results storage
        self.results = []
        self.validation_errors = []
        self.performance_stats = {
            'total_periods': 0,
            'successful_periods': 0,
            'strategy_errors': 0,
            'battery_constraint_violations': 0,
            'data_missing_periods': 0
        }
        
    def validate_input_data(self, consumption_df: pd.DataFrame, prices_df: pd.DataFrame,
                           start_date: datetime, end_date: datetime) -> bool:
        """Validate input data quality and consistency."""
        
        if not self.validation_mode:
            return True
            
        errors = []
        warnings_list = []
        
        # Check data availability
        required_consumption_periods = pd.date_range(start_date, end_date, freq='15min')
        required_price_periods = pd.date_range(start_date, end_date + timedelta(days=1), freq='15min')
        
        missing_consumption = len(required_consumption_periods) - len(consumption_df.loc[consumption_df.index.intersection(required_consumption_periods)])
        missing_prices = len(required_price_periods) - len(prices_df.loc[prices_df.index.intersection(required_price_periods)])
        
        if missing_consumption > len(required_consumption_periods) * 0.1:
            errors.append(f"More than 10% of consumption data missing ({missing_consumption} periods)")
        elif missing_consumption > 0:
            warnings_list.append(f"Some consumption data missing ({missing_consumption} periods)")
            
        if missing_prices > len(required_price_periods) * 0.05:
            errors.append(f"More than 5% of price data missing ({missing_prices} periods)")
        elif missing_prices > 0:
            warnings_list.append(f"Some price data missing ({missing_prices} periods)")
        
        # Check data ranges
        if 'kwh' in consumption_df.columns:
            max_consumption = consumption_df['kwh'].max()
            if max_consumption > 10:  # Reasonable limit for 15-min residential consumption
                warnings_list.append(f"Very high consumption values detected (max: {max_consumption:.2f} kWh/15min)")
                
        price_col = 'price_final_eur_kwh' if 'price_final_eur_kwh' in prices_df.columns else 'price_omie_eur_kwh'
        if price_col in prices_df.columns:
            price_range = prices_df[price_col]
            if price_range.min() < -0.1:
                warnings_list.append(f"Negative prices detected (min: {price_range.min():.4f} EUR/kWh)")
            if price_range.max() > 1.0:
                warnings_list.append(f"Very high prices detected (max: {price_range.max():.4f} EUR/kWh)")
        
        # Log warnings and errors
        for warning in warnings_list:
            warnings.warn(warning)
            
        if errors:
            self.validation_errors.extend(errors)
            for error in errors:
                print(f"VALIDATION ERROR: {error}")
            return False
            
        return True
    
    def run(self,
            consumption_df: pd.DataFrame,
            prices_df: pd.DataFrame,
            start_date: datetime,
            end_date: datetime) -> pd.DataFrame:
        """
        Run the simulation and return only physical energy flows.
        Monetary calculations are handled by the billing engine.
        """
        
        # Validate input data
        if not self.validate_input_data(consumption_df, prices_df, start_date, end_date):
            print("WARNING: Data validation failed. Proceeding with simulation anyway...")
        
        # Reset battery and results
        self.battery.reset()
        self.results = []
        self.performance_stats = {k: 0 for k in self.performance_stats.keys()}
        
        # Create time range
        current_time = start_date
        end_time = end_date.replace(hour=23, minute=45, second=0, microsecond=0)
        
        print(f"Starting CORRECTED simulation from {start_date} to {end_date}")
        print(f"Battery: {self.battery.capacity_kwh} kWh, "
              f"Power: {self.battery.max_charge_kw}/{self.battery.max_discharge_kw} kW, "
              f"SOC limits: {self.battery.soc_min*100:.0f}%-{self.battery.soc_max*100:.0f}%")
        print(f"Strategy: {type(self.strategy).__name__}")
        
        step_count = 0
        total_steps = int((end_time - current_time).total_seconds() / (self.time_step_minutes * 60)) + 1
        last_logged_date = None

        while current_time <= end_time:
            self.performance_stats['total_periods'] += 1
            
            # Progress indicator
            if current_time.date() != last_logged_date:
                progress = step_count / total_steps * 100
                print(f"Simulating {current_time.date()}... ({progress:.1f}% complete)")
                last_logged_date = current_time.date()
            
            # Get current consumption and price with error handling
            try:
                if current_time in consumption_df.index:
                    consumption_kwh = consumption_df.loc[current_time, 'kwh']
                    consumption_kw = consumption_df.loc[current_time, 'kw']
                else:
                    consumption_kwh = 0
                    consumption_kw = 0
                    self.performance_stats['data_missing_periods'] += 1
            except (KeyError, ValueError):
                consumption_kwh = 0
                consumption_kw = 0
                self.performance_stats['data_missing_periods'] += 1
            
            try:
                if current_time in prices_df.index:
                    base_price = prices_df.loc[current_time, 'price_omie_eur_kwh']
                    final_price = prices_df.loc[current_time, 'price_final_eur_kwh']
                else:
                    current_time += timedelta(minutes=self.time_step_minutes)
                    step_count += 1
                    self.performance_stats['data_missing_periods'] += 1
                    continue
            except (KeyError, ValueError):
                current_time += timedelta(minutes=self.time_step_minutes)
                step_count += 1
                self.performance_stats['data_missing_periods'] += 1
                continue
            
            # Validate price data
            if base_price < -0.5 or base_price > 2.0:
                print(f"WARNING: Extreme price {base_price:.4f} EUR/kWh at {current_time}")
            
            # Store battery state before action
            battery_state_before = self.battery.get_state()
            
            # Get strategy decision with error handling
            action = 'idle'
            power_kw = 0
            
            try:
                if isinstance(self.strategy, OptimalArbitrageStrategy):
                    action, power_kw = self.strategy.decide_action(
                        current_time, final_price, consumption_kw,
                        self.battery, prices_df, consumption_df
                    )
                else:
                    action, power_kw = self.strategy.decide_action(
                        current_time, final_price, consumption_kw,
                        self.battery, prices_df
                    )
            except Exception as e:
                print(f"Strategy error at {current_time}: {e}")
                self.performance_stats['strategy_errors'] += 1
                action, power_kw = 'idle', 0
            
            # Validate strategy output
            if power_kw < 0:
                print(f"WARNING: Negative power {power_kw} kW from strategy at {current_time}")
                power_kw = 0
            
            # Execute battery action with validation
            battery_charge_kwh = 0
            battery_discharge_kwh = 0
            action_success = True
            
            try:
                if action == 'charge' and power_kw > 0.001:
                    max_charge, _ = self.battery.get_max_power(self.time_step_hours)
                    if power_kw > max_charge + 0.01:  # Small tolerance
                        self.performance_stats['battery_constraint_violations'] += 1
                        power_kw = max_charge
                    
                    battery_charge_kwh = self.battery.charge(power_kw, self.time_step_hours)
                    
                elif action == 'discharge' and power_kw > 0.001:
                    _, max_discharge = self.battery.get_max_power(self.time_step_hours)
                    if power_kw > max_discharge + 0.01:  # Small tolerance
                        self.performance_stats['battery_constraint_violations'] += 1
                        power_kw = max_discharge
                    
                    battery_discharge_kwh = self.battery.discharge(power_kw, self.time_step_hours)
                    
            except Exception as e:
                print(f"Battery operation error at {current_time}: {e}")
                action_success = False
                action = 'idle'
                power_kw = 0
            
            # CORRECTED ENERGY FLOW CALCULATIONS
            # ====================================
            
            # House consumption is always the same
            house_consumption_kwh = consumption_kwh
            
            # Grid supply to house = house consumption - battery discharge (cannot be negative)
            grid_to_house_kwh = max(0, house_consumption_kwh - battery_discharge_kwh)

            # Total grid import = grid to house + battery charging
            total_grid_import_kwh = grid_to_house_kwh + battery_charge_kwh
            grid_export_kwh = max(0, battery_discharge_kwh - house_consumption_kwh)
            
            # Instantaneous powers (kW) - CORRECTED
            house_consumption_kw = consumption_kw  # This stays the same
            battery_charge_kw = battery_charge_kwh / self.time_step_hours if battery_charge_kwh > 0 else 0
            battery_discharge_kw = battery_discharge_kwh / self.time_step_hours if battery_discharge_kwh > 0 else 0
            
            # Net grid power = house consumption + battery charging - battery discharge
            # This can be negative if battery discharges more than house consumes, but we limit to 0 for import
            net_grid_power_kw = max(0, house_consumption_kw + battery_charge_kw - battery_discharge_kw)
            
            # Additional metrics
            battery_state_after = self.battery.get_state()
            
            # Track successful periods
            if action_success:
                self.performance_stats['successful_periods'] += 1
            
            # Store comprehensive results
            result = {
                # Time and basic data
                'datetime': current_time,
                'house_consumption_kwh': house_consumption_kwh,
                'house_consumption_kw': house_consumption_kw,
                
                # Battery actions
                'battery_action': action,
                'battery_power_kw': power_kw,
                'battery_charge_kwh': battery_charge_kwh,
                'battery_discharge_kwh': battery_discharge_kwh,
                'battery_charge_kw': battery_charge_kw,
                'battery_discharge_kw': battery_discharge_kw,
                'action_success': action_success,
                
                # Battery state
                'battery_soc': battery_state_after['soc'],
                'battery_soc_kwh': battery_state_after['soc_kwh'],
                'battery_soc_pct': battery_state_after['soc_pct'],
                'battery_available_charge_kwh': battery_state_after['available_to_charge_kwh'],
                'battery_available_discharge_kwh': battery_state_after['available_to_discharge_kwh'],
                
                # CORRECTED energy flows
                'grid_to_house_kwh': grid_to_house_kwh,
                'total_grid_import_kwh': total_grid_import_kwh,
                'grid_export_kwh': grid_export_kwh,
                'net_grid_power_kw': net_grid_power_kw,

                # Legacy columns for compatibility (but with correct values)
                'consumption_kwh': house_consumption_kwh,  # For backward compatibility
                'consumption_kw': house_consumption_kw,    # For backward compatibility
                'grid_consumption_kwh': total_grid_import_kwh,
                'grid_consumption_kw': net_grid_power_kw,

                # Performance tracking
                'battery_efficiency_charge': self.battery.efficiency_charge,
                'battery_efficiency_discharge': self.battery.efficiency_discharge,
                'cumulative_cycles': battery_state_after['cycles'],
                'degradation_cost_eur': battery_state_after.get('degradation_cost_eur', 0),
            }
            
            self.results.append(result)
            
            # Move to next time step
            current_time += timedelta(minutes=self.time_step_minutes)
            step_count += 1
        
        # Create enhanced results DataFrame
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            results_df.set_index('datetime', inplace=True)
        
        # Print performance statistics
        self._print_performance_stats()
        
        print(f"Simulation completed: {step_count} steps, {len(results_df)} results")
        
        return results_df
    
    def _print_performance_stats(self):
        """Print simulation performance statistics."""
        stats = self.performance_stats
        if stats['total_periods'] == 0:
            return
        
        print("\n--- SIMULATION PERFORMANCE ---")
        print(f"Total periods: {stats['total_periods']}")
        print(f"Successful periods: {stats['successful_periods']} ({stats['successful_periods']/stats['total_periods']*100:.1f}%)")
        if stats['strategy_errors'] > 0:
            print(f"Strategy errors: {stats['strategy_errors']} ({stats['strategy_errors']/stats['total_periods']*100:.1f}%)")
        if stats['battery_constraint_violations'] > 0:
            print(f"Battery constraint violations: {stats['battery_constraint_violations']}")
        if stats['data_missing_periods'] > 0:
            print(f"Missing data periods: {stats['data_missing_periods']} ({stats['data_missing_periods']/stats['total_periods']*100:.1f}%)")
    
    
    def print_summary(self, metrics: Dict):
        """Print CORRECTED comprehensive formatted summary."""
        if not metrics:
            print("No metrics to display.")
            return
            
        print("\n" + "="*70)
        print("CORRECTED SIMULATION SUMMARY - ENERGY ARBITRAGE")
        print("="*70)
        
        # Show data quality first
        quality = metrics.get('results_quality', 'unknown')
        if quality != 'good':
            print(f"⚠️  RESULTS QUALITY: {quality.upper()}")
            print("-" * 70)
        
        print(f"\nPeriod: {metrics['simulation_start'].strftime('%Y-%m-%d')} to {metrics['simulation_end'].strftime('%Y-%m-%d')} ({metrics['period_days']} days)")
        print(f"Total periods: {metrics['total_periods']} ({metrics['time_step_hours']:.2f}h intervals)")
        print(f"Data completeness: {metrics['data_completeness_pct']:.1f}%")
        print(f"Energy VAT rate used: {metrics.get('energy_vat_rate_used', 0.23):.0%}")
        
        print("\n--- CORRECTED ENERGY FLOWS ---")
        print(f"House consumption:        {metrics['total_house_consumption_kwh']:.1f} kWh")
        print(f"Grid import (with battery): {metrics['total_grid_import_kwh']:.1f} kWh")
        print(f"Grid reduction:           {metrics['grid_consumption_reduction_kwh']:.1f} kWh ({metrics['grid_consumption_reduction_pct']:.1f}%)")
        
        print("\n--- COSTS & SAVINGS ---")
        print(f"Cost without battery:  €{metrics['total_cost_without_battery_eur']:.2f}")
        print(f"Cost with battery:     €{metrics['total_cost_with_battery_eur']:.2f}")
        print(f"Total savings:         €{metrics['total_savings_eur']:.2f} ({metrics['savings_percentage']:.1f}%)")
        print(f"Daily avg savings:     €{metrics['daily_avg_savings_eur']:.3f}")
        print(f"Annual projection:     €{metrics['annual_projected_savings_eur']:.2f}")
        
        if metrics['simple_payback_years'] != float('inf'):
            print(f"Simple payback:        {metrics['simple_payback_years']:.1f} years")
        else:
            print(f"Simple payback:        Infinite (negative or zero savings)")
        
        print("\n--- BATTERY PERFORMANCE ---")
        print(f"Total charged:         {metrics['battery_total_charged_kwh']:.1f} kWh")
        print(f"Total discharged:      {metrics['battery_total_discharged_kwh']:.1f} kWh")
        print(f"Energy throughput:     {metrics['battery_energy_throughput_kwh']:.1f} kWh")
        print(f"Equivalent cycles:     {metrics['battery_cycles']:.2f}")
        print(f"Theoretical efficiency: {metrics['battery_theoretical_efficiency']:.1%}")
        print(f"Actual efficiency:     {metrics['battery_actual_efficiency']:.1%}")
        print(f"Utilization:           {metrics['battery_utilization_pct']:.1f}%")
        
        print("\n--- ARBITRAGE PERFORMANCE ---")
        print(f"Average price:         €{metrics['avg_price_eur_kwh']:.4f}/kWh ({metrics['avg_price_eur_kwh']*1000:.1f} EUR/MWh)")
        print(f"Avg charge price:      €{metrics['avg_charge_price_eur_kwh']:.4f}/kWh ({metrics['avg_charge_price_eur_kwh']*1000:.1f} EUR/MWh)")
        print(f"Avg discharge price:   €{metrics['avg_discharge_price_eur_kwh']:.4f}/kWh ({metrics['avg_discharge_price_eur_kwh']*1000:.1f} EUR/MWh)")
        print(f"Arbitrage spread:      €{metrics['arbitrage_spread_eur_kwh']:.4f}/kWh ({metrics['arbitrage_spread_eur_mwh']:.1f} EUR/MWh)")
        
        print("\n--- CORRECTED PEAK MANAGEMENT ---")
        print(f"Peak house consumption: {metrics['peak_house_consumption_kw']:.2f} kW")
        print(f"Peak grid import:       {metrics['peak_grid_import_kw']:.2f} kW")
        print(f"Peak reduction:         {metrics['peak_reduction_kw']:.2f} kW ({metrics['peak_reduction_pct']:.1f}%)")
        
        if metrics.get('degradation_cost_eur', 0) > 0:
            print("\n--- DEGRADATION ANALYSIS ---")
            print(f"Degradation cost:      €{metrics['degradation_cost_eur']:.2f}")
            print(f"Net savings:           €{metrics['net_savings_after_degradation_eur']:.2f}")
        
        print("\n--- SIMULATION QUALITY ---")
        print(f"Successful periods:    {metrics['successful_periods_pct']:.1f}%")
        if metrics['strategy_error_rate_pct'] > 0:
            print(f"Strategy error rate:   {metrics['strategy_error_rate_pct']:.1f}%")
        
        # Analysis and recommendations
        print("\n--- ANALYSIS & RECOMMENDATIONS ---")
        if metrics['arbitrage_spread_eur_mwh'] < 30:
            print("⚠️  Low arbitrage spread (<30 EUR/MWh) - may not be profitable")
        if metrics['simple_payback_years'] > 15:
            print("⚠️  Payback period too long for typical battery lifetime")
        if metrics['grid_consumption_reduction_pct'] < 5:
            print("⚠️  Very low grid consumption reduction - check strategy settings")
        if metrics['battery_utilization_pct'] > 90:
            print("⚠️  Very high battery utilization - may cause premature degradation")
        
        print("="*70)