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
            end_date: datetime,
            vat_rate: float = 0.23,

            reduced_vat_rate: float = 0.06,
            reduced_vat_kwh_per_cycle: float = 200.0,
            vat_cycle_days: int = 30,
            iec_vat_rate: float = 0.23,
            contracted_power_kva: float = 6.9,
            vat_reduced_power_threshold_kva: float = 6.9,

            daily_fixed_cost_eur: float = 0.0) -> pd.DataFrame:
        """
        Run the CORRECTED simulation with proper energy flow calculations.
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

        self.daily_fixed_cost_eur = daily_fixed_cost_eur

        cycle_consumption_with = 0.0
        cycle_consumption_without = 0.0
        cycle_start = start_date


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
                    energy_price = prices_df.loc[current_time, 'price_energy_pre_vat_eur_kwh']
                    iec_tax = prices_df.loc[current_time, 'iec_tax_eur_kwh']

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
            
            # Instantaneous powers (kW) - CORRECTED
            house_consumption_kw = consumption_kw  # This stays the same
            battery_charge_kw = battery_charge_kwh / self.time_step_hours if battery_charge_kwh > 0 else 0
            battery_discharge_kw = battery_discharge_kwh / self.time_step_hours if battery_discharge_kwh > 0 else 0
            
            # Net grid power = house consumption + battery charging - battery discharge
            # This can be negative if battery discharges more than house consumes, but we limit to 0 for import
            net_grid_power_kw = max(0, house_consumption_kw + battery_charge_kw - battery_discharge_kw)
            
            # CORRECTED COST CALCULATIONS
            # ===========================
            
            # Reset VAT cycle if needed
            if (current_time - cycle_start).days >= vat_cycle_days:
                cycle_start += timedelta(days=vat_cycle_days)
                cycle_consumption_with = 0.0
                cycle_consumption_without = 0.0

            def compute_cost(amount_kwh: float, cycle_consumption: float) -> Tuple[float, float]:
                reduced_kwh = 0.0
                if contracted_power_kva <= vat_reduced_power_threshold_kva:
                    reduced_remaining = max(0.0, reduced_vat_kwh_per_cycle - cycle_consumption)
                    reduced_kwh = min(amount_kwh, reduced_remaining)
                standard_kwh = amount_kwh - reduced_kwh
                cost_energy = (
                    energy_price * reduced_kwh * (1 + reduced_vat_rate)
                    + energy_price * standard_kwh * (1 + vat_rate)
                )
                cost_iec = iec_tax * amount_kwh * (1 + iec_vat_rate)
                cycle_consumption += amount_kwh
                return cost_energy + cost_iec, cycle_consumption

            cost_without_battery, cycle_consumption_without = compute_cost(house_consumption_kwh, cycle_consumption_without)
            cost_with_battery, cycle_consumption_with = compute_cost(total_grid_import_kwh, cycle_consumption_with)
            
            # Savings = difference
            savings = cost_without_battery - cost_with_battery
            
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
                
                # Prices
                'price_omie_eur_kwh': base_price,
                'price_final_eur_kwh': final_price,
                'vat_rate': vat_rate,
                
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
                'net_grid_power_kw': net_grid_power_kw,
                
                # Legacy columns for compatibility (but with correct values)
                'consumption_kwh': house_consumption_kwh,  # For backward compatibility
                'consumption_kw': house_consumption_kw,    # For backward compatibility
                'grid_consumption_kwh': total_grid_import_kwh,
                'grid_consumption_kw': net_grid_power_kw,
                
                # CORRECTED costs and savings
                'cost_without_battery_eur': cost_without_battery,
                'cost_with_battery_eur': cost_with_battery,
                'savings_eur': savings,
                'savings_pct': (savings / cost_without_battery * 100) if cost_without_battery > 0 else 0,
                
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
    
    def calculate_summary_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate CORRECTED summary metrics from simulation results.
        """
        if results_df.empty:
            return {}
        
        # Basic metrics
        n_periods = len(results_df)
        n_days = (results_df.index[-1] - results_df.index[0]).days + 1
        
        # CORRECTED energy and consumption metrics
        total_house_consumption_kwh = results_df['house_consumption_kwh'].sum()
        total_grid_import_kwh = results_df['total_grid_import_kwh'].sum()
        
        # Grid consumption REDUCTION (should be positive if battery helps)
        grid_consumption_reduction_kwh = total_house_consumption_kwh - total_grid_import_kwh
        
        # Cost metrics
        total_cost_without_battery = results_df['cost_without_battery_eur'].sum()
        total_cost_with_battery = results_df['cost_with_battery_eur'].sum()
        fixed_cost_total = self.daily_fixed_cost_eur * n_days
        total_cost_without_battery += fixed_cost_total
        total_cost_with_battery += fixed_cost_total
        total_savings = total_cost_without_battery - total_cost_with_battery
        
        # Battery metrics
        battery_final_state = self.battery.get_state()
        total_charged = battery_final_state['total_charged_kwh']
        total_discharged = battery_final_state['total_discharged_kwh']
        battery_cycles = battery_final_state['cycles']
        
        # Efficiency calculations
        if total_charged > 0:
            actual_round_trip_efficiency = total_discharged / total_charged
        else:
            actual_round_trip_efficiency = 0
        
        # CORRECTED peak analysis - use instantaneous power correctly
        peak_house_consumption = results_df['house_consumption_kw'].max()
        peak_grid_import = results_df['net_grid_power_kw'].max()
        peak_reduction_kw = max(0, peak_house_consumption - peak_grid_import)
        
        # Time-based analysis
        daily_avg_savings = total_savings / n_days if n_days > 0 else 0
        annual_projected_savings = daily_avg_savings * 365
        
        # Battery utilization
        battery_active_periods = (results_df['battery_action'] != 'idle').sum()
        battery_utilization_pct = battery_active_periods / n_periods * 100 if n_periods > 0 else 0
        
        # Price analysis
        charge_periods = results_df[results_df['battery_action'] == 'charge']
        discharge_periods = results_df[results_df['battery_action'] == 'discharge']
        
        avg_charge_price = charge_periods['price_final_eur_kwh'].mean() if len(charge_periods) > 0 else 0
        avg_discharge_price = discharge_periods['price_final_eur_kwh'].mean() if len(discharge_periods) > 0 else 0
        price_spread = avg_discharge_price - avg_charge_price
        
        # Advanced metrics
        energy_throughput = total_charged + total_discharged
        capacity_factor_charge = (total_charged / (self.battery.max_charge_kw * n_periods * 0.25)) * 100 if n_periods > 0 else 0
        capacity_factor_discharge = (total_discharged / (self.battery.max_discharge_kw * n_periods * 0.25)) * 100 if n_periods > 0 else 0
        
        # Economic metrics
        if annual_projected_savings > 0:
            simple_payback_years = 8000 / annual_projected_savings  # Assuming 8000 EUR investment
        else:
            simple_payback_years = float('inf')
        
        # Check if results make sense
        results_quality = "good"
        if grid_consumption_reduction_kwh < 0:
            results_quality = "poor - battery increasing grid consumption"
        elif total_savings < 0:
            results_quality = "poor - negative savings"
        elif simple_payback_years > 20:
            results_quality = "poor - payback too long"
        
        return {
            # Period information
            'simulation_start': results_df.index[0],
            'simulation_end': results_df.index[-1],
            'period_days': n_days,
            'total_periods': n_periods,
            'time_step_hours': self.time_step_hours,
            'results_quality': results_quality,
            
            # CORRECTED energy metrics
            'total_house_consumption_kwh': total_house_consumption_kwh,
            'total_grid_import_kwh': total_grid_import_kwh,
            'grid_consumption_reduction_kwh': grid_consumption_reduction_kwh,
            'grid_consumption_reduction_pct': (grid_consumption_reduction_kwh / total_house_consumption_kwh * 100) if total_house_consumption_kwh > 0 else 0,
            
            # Cost metrics
            'total_cost_without_battery_eur': total_cost_without_battery,
            'total_cost_with_battery_eur': total_cost_with_battery,
            'total_savings_eur': total_savings,
            'savings_percentage': (total_savings / total_cost_without_battery * 100) if total_cost_without_battery > 0 else 0,
            'daily_avg_savings_eur': daily_avg_savings,
            'annual_projected_savings_eur': annual_projected_savings,
            'daily_fixed_cost_eur': self.daily_fixed_cost_eur,
            'total_fixed_cost_eur': fixed_cost_total,
            
            # Battery performance
            'battery_total_charged_kwh': total_charged,
            'battery_total_discharged_kwh': total_discharged,
            'battery_energy_throughput_kwh': energy_throughput,
            'battery_cycles': battery_cycles,
            'battery_theoretical_efficiency': battery_final_state['round_trip_efficiency'],
            'battery_actual_efficiency': actual_round_trip_efficiency,
            'battery_utilization_pct': battery_utilization_pct,
            'battery_capacity_factor_charge_pct': capacity_factor_charge,
            'battery_capacity_factor_discharge_pct': capacity_factor_discharge,
            
            # CORRECTED peak management
            'peak_house_consumption_kw': peak_house_consumption,
            'peak_grid_import_kw': peak_grid_import,
            'peak_reduction_kw': peak_reduction_kw,
            'peak_reduction_pct': (peak_reduction_kw / peak_house_consumption * 100) if peak_house_consumption > 0 else 0,
            
            # Price arbitrage
            'avg_price_eur_kwh': results_df['price_final_eur_kwh'].mean(),
            'avg_charge_price_eur_kwh': avg_charge_price,
            'avg_discharge_price_eur_kwh': avg_discharge_price,
            'arbitrage_spread_eur_kwh': price_spread,
            'arbitrage_spread_eur_mwh': price_spread * 1000,
            
            # Economic analysis
            'simple_payback_years': simple_payback_years,
            'degradation_cost_eur': battery_final_state.get('degradation_cost_eur', 0),
            'net_savings_after_degradation_eur': total_savings - battery_final_state.get('degradation_cost_eur', 0),
            
            # Performance statistics
            'successful_periods_pct': (self.performance_stats['successful_periods'] / self.performance_stats['total_periods'] * 100) if self.performance_stats['total_periods'] > 0 else 0,
            'strategy_error_rate_pct': (self.performance_stats['strategy_errors'] / self.performance_stats['total_periods'] * 100) if self.performance_stats['total_periods'] > 0 else 0,
            'data_completeness_pct': ((self.performance_stats['total_periods'] - self.performance_stats['data_missing_periods']) / self.performance_stats['total_periods'] * 100) if self.performance_stats['total_periods'] > 0 else 0,
        }
    
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