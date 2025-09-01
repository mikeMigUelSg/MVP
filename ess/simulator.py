"""
ess/simulator.py - Main simulation engine for energy arbitrage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from .battery import Battery
from .strategies import ArbitrageStrategy, OptimalArbitrageStrategy


class EnergyArbitrageSimulator:
    """
    Simulator for battery energy arbitrage operations.
    """
    
    def __init__(self,
                 battery: Battery,
                 strategy: Optional[object] = None,
                 time_step_minutes: int = 15):
        """
        Parameters
        ----------
        battery : Battery
            Battery model instance
        strategy : object
            Strategy instance (ArbitrageStrategy or OptimalArbitrageStrategy)
        time_step_minutes : int
            Simulation time step in minutes (default: 15)
        """
        self.battery = battery
        self.strategy = strategy or ArbitrageStrategy()
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60
        
        # Results storage
        self.results = []
        
    def run(self,
            consumption_df: pd.DataFrame,
            prices_df: pd.DataFrame,
            start_date: datetime,
            end_date: datetime,
            tariff_margin_eur_kwh: float = 0.01,
            grid_fees_eur_kwh: float = 0.0,
            vat_rate: float = 0.23) -> pd.DataFrame:
        """
        Run the simulation for the specified period.
        
        Parameters
        ----------
        consumption_df : pd.DataFrame
            Consumption data with 'kwh' and 'kw' columns
        prices_df : pd.DataFrame
            Price data with 'price_eur_per_kwh' column (needs D+1 for lookahead)
        start_date : datetime
            Simulation start
        end_date : datetime
            Simulation end
        tariff_margin_eur_kwh : float
            Retailer margin (EUR/kWh)
        grid_fees_eur_kwh : float
            Grid and system fees (EUR/kWh)
        vat_rate : float
            VAT rate (e.g., 0.23 for 23%)
        
        Returns
        -------
        pd.DataFrame
            Simulation results with all metrics
        """
        # Reset battery to initial state
        self.battery.reset()
        self.results = []
        
        # Create time range
        current_time = start_date
        end_time = end_date.replace(hour=23, minute=45, second=0, microsecond=0)
        
        print(f"Starting simulation from {start_date} to {end_date}")
        print(f"Battery: {self.battery.capacity_kwh} kWh, "
              f"Power: {self.battery.max_charge_kw}/{self.battery.max_discharge_kw} kW")
        
        step_count = 0
        total_steps = int((end_time - current_time).total_seconds() /
                          (self.time_step_minutes * 60)) + 1
        last_logged_date = None

        while current_time <= end_time:
            # Progress indicator - log once per day based on the date
            if current_time.date() != last_logged_date:
                print(f"Simulating {current_time.date()}...")
                last_logged_date = current_time.date()
            
            # Get current consumption and price
            try:
                consumption_kwh = consumption_df.loc[current_time, 'kwh']
                consumption_kw = consumption_df.loc[current_time, 'kw']
            except KeyError:
                # No consumption data for this timestamp
                consumption_kwh = 0
                consumption_kw = 0
            
            try:
                base_price = prices_df.loc[current_time, 'price_eur_per_kwh']
            except KeyError:
                # No price data - skip this timestep but count the step
                current_time += timedelta(minutes=self.time_step_minutes)
                step_count += 1
                continue
            
            # Calculate final price with margin and taxes
            price_with_margin = base_price + tariff_margin_eur_kwh + grid_fees_eur_kwh
            final_price = price_with_margin * (1 + vat_rate)
            
            # Get strategy decision
            action, power_kw = self.strategy.decide_action(
                current_time,
                base_price,  # Use base price for decisions
                consumption_kw,
                self.battery,
                prices_df
            )
            
            # Execute battery action
            battery_charge_kwh = 0
            battery_discharge_kwh = 0
            
            if action == 'charge':
                battery_charge_kwh = self.battery.charge(power_kw, self.time_step_hours)
            elif action == 'discharge':
                battery_discharge_kwh = self.battery.discharge(power_kw, self.time_step_hours)
            
            # Calculate energy flows
            # Energy from grid to house (net of battery discharge)
            grid_to_house_kwh = max(0, consumption_kwh - battery_discharge_kwh)
            
            # Total energy from grid (house + battery charging)
            total_grid_import_kwh = grid_to_house_kwh + battery_charge_kwh
            
            # Calculate costs
            cost_without_battery = consumption_kwh * final_price
            cost_with_battery = total_grid_import_kwh * final_price
            savings = cost_without_battery - cost_with_battery
            
            # Store results
            self.results.append({
                'datetime': current_time,
                'consumption_kwh': consumption_kwh,
                'consumption_kw': consumption_kw,
                'price_omie_eur_kwh': base_price,
                'price_final_eur_kwh': final_price,
                'battery_action': action,
                'battery_power_kw': power_kw,
                'battery_charge_kwh': battery_charge_kwh,
                'battery_discharge_kwh': battery_discharge_kwh,
                'battery_soc': self.battery.soc,
                'battery_soc_kwh': self.battery.soc_kwh,
                'grid_import_kwh': total_grid_import_kwh,
                'grid_to_house_kwh': grid_to_house_kwh,
                'cost_without_battery_eur': cost_without_battery,
                'cost_with_battery_eur': cost_with_battery,
                'savings_eur': savings
            })
            
            # Move to next time step
            current_time += timedelta(minutes=self.time_step_minutes)
            step_count += 1
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df.set_index('datetime', inplace=True)
        
        print(f"Simulation completed: {step_count} steps")
        
        return results_df
    
    def calculate_summary_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate summary metrics from simulation results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results from simulation run
        
        Returns
        -------
        Dict
            Summary metrics
        """
        total_consumption_kwh = results_df['consumption_kwh'].sum()
        total_cost_without_battery = results_df['cost_without_battery_eur'].sum()
        total_cost_with_battery = results_df['cost_with_battery_eur'].sum()
        total_savings = results_df['savings_eur'].sum()
        
        # Battery metrics
        total_charged = self.battery.total_charged_kwh
        total_discharged = self.battery.total_discharged_kwh
        battery_cycles = self.battery.cycles
        
        # Calculate efficiency
        if total_charged > 0:
            round_trip_efficiency = total_discharged / total_charged
        else:
            round_trip_efficiency = 0
        
        # Peak reduction
        peak_consumption = results_df['consumption_kw'].max()
        peak_with_battery = (results_df['consumption_kw'] - 
                            results_df['battery_discharge_kwh'] / (self.time_step_minutes / 60)).max()
        peak_reduction_kw = peak_consumption - peak_with_battery
        peak_reduction_pct = (peak_reduction_kw / peak_consumption * 100) if peak_consumption > 0 else 0
        
        # Time-based metrics
        n_days = (results_df.index[-1] - results_df.index[0]).days + 1
        daily_avg_savings = total_savings / n_days if n_days > 0 else 0
        annual_projected_savings = daily_avg_savings * 365
        
        # Battery utilization
        battery_active_periods = (results_df['battery_action'] != 'idle').sum()
        total_periods = len(results_df)
        battery_utilization = battery_active_periods / total_periods * 100
        
        return {
            'period_days': n_days,
            'total_consumption_kwh': total_consumption_kwh,
            'total_cost_without_battery_eur': total_cost_without_battery,
            'total_cost_with_battery_eur': total_cost_with_battery,
            'total_savings_eur': total_savings,
            'savings_percentage': total_savings / total_cost_without_battery * 100,
            'daily_avg_savings_eur': daily_avg_savings,
            'annual_projected_savings_eur': annual_projected_savings,
            'battery_total_charged_kwh': total_charged,
            'battery_total_discharged_kwh': total_discharged,
            'battery_cycles': battery_cycles,
            'battery_round_trip_efficiency': round_trip_efficiency,
            'battery_utilization_pct': battery_utilization,
            'peak_consumption_kw': peak_consumption,
            'peak_with_battery_kw': peak_with_battery,
            'peak_reduction_kw': peak_reduction_kw,
            'peak_reduction_pct': peak_reduction_pct,
            'avg_price_eur_kwh': results_df['price_final_eur_kwh'].mean(),
            'avg_charge_price_eur_kwh': results_df[results_df['battery_action'] == 'charge']['price_final_eur_kwh'].mean() if any(results_df['battery_action'] == 'charge') else 0,
            'avg_discharge_price_eur_kwh': results_df[results_df['battery_action'] == 'discharge']['price_final_eur_kwh'].mean() if any(results_df['battery_action'] == 'discharge') else 0,
        }
    
    def print_summary(self, metrics: Dict):
        """Print formatted summary of simulation results."""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY - ENERGY ARBITRAGE")
        print("="*60)
        
        print(f"\nPeriod: {metrics['period_days']} days")
        print(f"Total Consumption: {metrics['total_consumption_kwh']:.1f} kWh")
        
        print("\n--- COSTS & SAVINGS ---")
        print(f"Cost without battery: €{metrics['total_cost_without_battery_eur']:.2f}")
        print(f"Cost with battery:    €{metrics['total_cost_with_battery_eur']:.2f}")
        print(f"Total savings:        €{metrics['total_savings_eur']:.2f} ({metrics['savings_percentage']:.1f}%)")
        print(f"Daily avg savings:    €{metrics['daily_avg_savings_eur']:.2f}")
        print(f"Annual projection:    €{metrics['annual_projected_savings_eur']:.2f}")
        
        print("\n--- BATTERY PERFORMANCE ---")
        print(f"Total charged:        {metrics['battery_total_charged_kwh']:.1f} kWh")
        print(f"Total discharged:     {metrics['battery_total_discharged_kwh']:.1f} kWh")
        print(f"Equivalent cycles:    {metrics['battery_cycles']:.1f}")
        print(f"Round-trip efficiency: {metrics['battery_round_trip_efficiency']:.1%}")
        print(f"Utilization:          {metrics['battery_utilization_pct']:.1f}%")
        
        print("\n--- ARBITRAGE METRICS ---")
        print(f"Avg price:            €{metrics['avg_price_eur_kwh']:.4f}/kWh")
        print(f"Avg charge price:     €{metrics['avg_charge_price_eur_kwh']:.4f}/kWh")
        print(f"Avg discharge price:  €{metrics['avg_discharge_price_eur_kwh']:.4f}/kWh")
        if metrics['avg_charge_price_eur_kwh'] > 0:
            spread = metrics['avg_discharge_price_eur_kwh'] - metrics['avg_charge_price_eur_kwh']
            print(f"Price spread:         €{spread:.4f}/kWh")
        
        print("\n--- PEAK REDUCTION ---")
        print(f"Peak without battery: {metrics['peak_consumption_kw']:.2f} kW")
        print(f"Peak with battery:    {metrics['peak_with_battery_kw']:.2f} kW")
        print(f"Peak reduction:       {metrics['peak_reduction_kw']:.2f} kW ({metrics['peak_reduction_pct']:.1f}%)")
        
        print("="*60)