"""
Inspect the computed SDP policy to understand what's happening.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the controller (if it was saved)
# Actually, let's just compute what the action should be for a specific state

# Parameters
C_bat = 10.0
dt_hours = 0.25
eta_c = 0.90
n_soc_points = 100
soc_min = 0.1
soc_max = 0.9

# Create SOC grid
X_grid = np.linspace(soc_min, soc_max, n_soc_points)
print(f"SOC grid spacing: {(soc_max - soc_min) / (n_soc_points - 1):.6f}")

# The constant battery power observed
battery_power_observed = 0.7182940516273847

# Calculate what SOC change this produces
p_eff = battery_power_observed * eta_c
soc_change = (p_eff * dt_hours) / C_bat

print(f"\nObserved battery power: {battery_power_observed:.10f} kW")
print(f"Effective power (with efficiency): {p_eff:.10f} kW")
print(f"SOC change per timestep: {soc_change:.10f}")
print(f"Number of grid points jumped: {soc_change / ((soc_max - soc_min) / (n_soc_points - 1)):.10f}")

# Check if this corresponds to a specific grid spacing
grid_spacing = (soc_max - soc_min) / (n_soc_points - 1)
n_points = soc_change / grid_spacing

print(f"\nThis corresponds to moving {n_points:.1f} grid points per timestep")

# So it's moving exactly 2 grid points!
# Let's verify: what battery power would give us a 2-grid-point move?
target_soc_change = 2 * grid_spacing
target_energy = target_soc_change * C_bat
target_power_eff = target_energy / dt_hours
target_power_ac = target_power_eff / eta_c

print(f"\nTo move 2 grid points:")
print(f"  Target SOC change: {target_soc_change:.10f}")
print(f"  Target energy: {target_energy:.10f} kWh")
print(f"  Target power (effective): {target_power_eff:.10f} kW")
print(f"  Target power (AC): {target_power_ac:.10f} kW")
print(f"  Matches observed? {abs(target_power_ac - battery_power_observed) < 1e-10}")

print("\n" + "="*60)
print("HYPOTHESIS:")
print("="*60)
print("The SDP is consistently choosing the action that moves the SOC")
print("forward by exactly 2 grid points. This might indicate:")
print("1. A bug in the optimization (always picking the same discretized action)")
print("2. A problem with the cost function that makes this particular")
print("   action appear optimal regardless of state")
print("3. An issue with action interpolation")

