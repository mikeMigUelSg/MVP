#!/usr/bin/env python3
"""
quick_test.py - Script to quickly test the corrected simulation
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test different strategies with corrected simulator
from ess.battery import Battery
from ess.strategies import ArbitrageStrategy
from ess.simulator import EnergyArbitrageSimulator


def create_test_data(start_date: datetime, days: int = 3):
    """Create simple test data for verification."""
    
    # Create time index
    end_date = start_date + timedelta(days=days-1)
    time_index = pd.date_range(start_date, end_date, freq='15min')
    
    # Create simple consumption pattern (higher during day, lower at night)
    consumption_data = []
    for timestamp in time_index:
        hour = timestamp.hour
        if 6 <= hour <= 22:  # Day time
            base_consumption = 1.5  # kW
        else:  # Night time
            base_consumption = 0.5  # kW
        
        # Add some randomness
        import random
        consumption_kw = base_consumption * (0.8 + 0.4 * random.random())
        consumption_kwh = consumption_kw * 0.25  # 15 minutes
        
        consumption_data.append({
            'kwh': consumption_kwh,
            'kw': consumption_kw,
            'permil': consumption_kwh / 10.95 * 1000  # Fake permil value
        })
    
    consumption_df = pd.DataFrame(consumption_data, index=time_index)
    
    # Create simple price pattern (low at night, high during day)
    price_data = []
    for timestamp in time_index:
        hour = timestamp.hour
        if 2 <= hour <= 6:  # Very low prices at night
            base_price = 0.02  # 20 EUR/MWh
        elif 18 <= hour <= 21:  # High prices in evening
            base_price = 0.12  # 120 EUR/MWh  
        else:  # Normal prices
            base_price = 0.06  # 60 EUR/MWh
        
        # Add some randomness
        price_eur_kwh = base_price * (0.7 + 0.6 * random.random())
        
        price_data.append({
            'price_eur_per_mwh': price_eur_kwh * 1000,
            'price_eur_per_kwh': price_eur_kwh
        })
    
    prices_df = pd.DataFrame(price_data, index=time_index)
    
    return consumption_df, prices_df


def run_quick_test():
    """Run a quick test with synthetic data to verify corrections."""
    
    print("="*60)
    print("QUICK TEST - CORRECTED SIMULATOR")
    print("="*60)
    
    # Create test period
    start_date = datetime(2024, 1, 15)  # Avoid DST issues
    
    print(f"\n1. Creating synthetic test data...")
    consumption_df, prices_df = create_test_data(start_date, days=3)
    
    print(f"   Created {len(consumption_df)} consumption points")
    print(f"   Created {len(prices_df)} price points")
    print(f"   Price range: {prices_df['price_eur_per_kwh'].min():.4f} - {prices_df['price_eur_per_kwh'].max():.4f} EUR/kWh")
    print(f"   Consumption range: {consumption_df['kw'].min():.2f} - {consumption_df['kw'].max():.2f} kW")
    
    # Create battery
    print(f"\n2. Creating battery system...")
    battery = Battery(
        capacity_kwh=20.0,
        soc_init=0.5,
        max_charge_kw=5.0,
        max_discharge_kw=5.0,
        efficiency=0.92,
        soc_min=0.15,
        soc_max=0.95
    )
    
    print(f"   Battery: {battery.capacity_kwh} kWh, {battery.max_charge_kw}/{battery.max_discharge_kw} kW")
    print(f"   SOC range: {battery.soc_min*100:.0f}%-{battery.soc_max*100:.0f}%")
    print(f"   Round-trip efficiency: {battery.efficiency_charge * battery.efficiency_discharge:.1%}")
    
    # Create strategy
    print(f"\n3. Creating arbitrage strategy...")
    strategy = ArbitrageStrategy(
        charge_threshold_percentile=25,  # Charge in cheapest 25%
        discharge_threshold_percentile=75,  # Discharge in most expensive 25%
        min_price_spread=25,  # 25 EUR/MWh minimum spread
        lookahead_hours=24
    )
    
    print(f"   Charge threshold: {strategy.charge_threshold_percentile}%")
    print(f"   Discharge threshold: {strategy.discharge_threshold_percentile}%")
    print(f"   Minimum spread: {strategy.min_price_spread*1000:.0f} EUR/MWh")
    
    # Create simulator
    print(f"\n4. Running corrected simulation...")
    simulator = EnergyArbitrageSimulator(battery, strategy)
    
    end_date = start_date + timedelta(days=2)  # 3 days total
    
    results_df = simulator.run(
        consumption_df,
        prices_df,
        start_date,
        end_date,
        tariff_margin_eur_kwh=0.015,  # 15 EUR/MWh margin
        grid_fees_eur_kwh=0.05,
        vat_rate=0.23
    )
    
    # Calculate and display metrics
    print(f"\n5. Analyzing results...")
    metrics = simulator.calculate_summary_metrics(results_df)
    
    # Quick validation checks
    print(f"\n6. VALIDATION CHECKS:")
    
    house_consumption = metrics.get('total_house_consumption_kwh', 0)
    grid_import = metrics.get('total_grid_import_kwh', 0)
    reduction = metrics.get('grid_consumption_reduction_kwh', 0)
    
    print(f"   House consumption: {house_consumption:.1f} kWh")
    print(f"   Grid import: {grid_import:.1f} kWh")
    print(f"   Reduction: {reduction:.1f} kWh")
    
    if reduction > 0:
        print("   ✅ Battery REDUCED grid consumption (good!)")
    else:
        print("   ❌ Battery INCREASED grid consumption (bad!)")
    
    savings = metrics.get('total_savings_eur', 0)
    print(f"   Total savings: €{savings:.2f}")
    
    if savings > 0:
        print("   ✅ Positive savings (good!)")
    else:
        print("   ❌ Negative savings (bad!)")
    
    payback = metrics.get('simple_payback_years', float('inf'))
    print(f"   Payback: {payback:.1f} years")
    
    if payback < 15:
        print("   ✅ Reasonable payback period (good!)")
    else:
        print("   ❌ Payback too long (bad!)")
    
    peak_reduction = metrics.get('peak_reduction_kw', 0)
    print(f"   Peak reduction: {peak_reduction:.2f} kW")
    
    if peak_reduction >= 0:
        print("   ✅ Peak reduced or maintained (good!)")
    else:
        print("   ❌ Peak increased (bad!)")
    
    # Show detailed summary
    print(f"\n7. DETAILED RESULTS:")
    simulator.print_summary(metrics)
    
    # Final assessment
    quality = metrics.get('results_quality', 'unknown')
    print(f"\n8. FINAL ASSESSMENT: {quality.upper()}")
    
    if quality == 'good':
        print("   🎉 Corrections appear to be working correctly!")
        print("   You can now run the full simulation with confidence.")
    else:
        print("   ⚠️  Still some issues detected. Check the analysis above.")
        
    return results_df, metrics


def compare_strategies():
    """Compare different strategies to show the impact."""
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    start_date = datetime(2024, 1, 15)
    consumption_df, prices_df = create_test_data(start_date, days=2)
    
    strategies = [
        ("Conservative", ArbitrageStrategy(charge_threshold_percentile=15, 
                                         discharge_threshold_percentile=85, 
                                         min_price_spread=40)),
        ("Moderate", ArbitrageStrategy(charge_threshold_percentile=25, 
                                     discharge_threshold_percentile=75, 
                                     min_price_spread=25)),
        ("Aggressive", ArbitrageStrategy(charge_threshold_percentile=40, 
                                       discharge_threshold_percentile=60, 
                                       min_price_spread=10))
    ]
    
    results = {}
    
    for name, strategy in strategies:
        print(f"\nTesting {name} strategy...")
        
        battery = Battery(capacity_kwh=20.0, soc_init=0.5, max_charge_kw=5.0, 
                         max_discharge_kw=5.0, efficiency=0.92)
        
        simulator = EnergyArbitrageSimulator(battery, strategy)
        end_date = start_date + timedelta(days=1)
        
        try:
            results_df = simulator.run(consumption_df, prices_df, start_date, end_date)
            metrics = simulator.calculate_summary_metrics(results_df)
            results[name] = metrics
            
            savings = metrics.get('total_savings_eur', 0)
            reduction = metrics.get('grid_consumption_reduction_kwh', 0)
            utilization = metrics.get('battery_utilization_pct', 0)
            
            print(f"   Savings: €{savings:.2f}")
            print(f"   Grid reduction: {reduction:.1f} kWh")
            print(f"   Battery utilization: {utilization:.1f}%")
            
        except Exception as e:
            print(f"   Error: {e}")
            results[name] = None
    
    # Summary table
    print(f"\nSTRATEGY COMPARISON SUMMARY:")
    print(f"{'Strategy':<12} {'Savings':<10} {'Reduction':<12} {'Utilization':<12} {'Quality':<15}")
    print("-" * 65)
    
    for name, metrics in results.items():
        if metrics:
            savings = metrics.get('total_savings_eur', 0)
            reduction = metrics.get('grid_consumption_reduction_kwh', 0)
            utilization = metrics.get('battery_utilization_pct', 0)
            quality = metrics.get('results_quality', 'unknown')
            
            print(f"{name:<12} €{savings:<9.2f} {reduction:<11.1f} {utilization:<11.1f}% {quality:<15}")
        else:
            print(f"{name:<12} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<15}")


if __name__ == "__main__":
    # Run quick test
    results_df, metrics = run_quick_test()
    
    # Run strategy comparison
    compare_strategies()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)