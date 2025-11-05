"""
Test script to understand what feasible actions are being generated.
"""

import numpy as np

# Parameters from config
C_bat = 10.0  # kWh
dt_hours = 0.25  # 15 minutes
eta_c = 0.90
eta_d = 0.90
P_nom = 5.0  # max battery power
soc_min = 0.1
soc_max = 0.9
n_soc_points = 100

# Create SOC grid
X_grid = np.linspace(soc_min, soc_max, n_soc_points)

print("=" * 60)
print("FEASIBLE ACTIONS ANALYSIS")
print("=" * 60)
print(f"\nSOC grid: {n_soc_points} points from {soc_min} to {soc_max}")
print(f"Grid spacing: {(soc_max - soc_min) / (n_soc_points - 1):.6f}")

# Current SOC from data
current_soc = 0.178

# Find closest grid point
closest_idx = np.argmin(np.abs(X_grid - current_soc))
closest_soc = X_grid[closest_idx]

print(f"\nCurrent SOC: {current_soc:.6f}")
print(f"Closest grid point: {closest_soc:.6f} (index {closest_idx})")
print(f"Difference: {abs(current_soc - closest_soc):.6f}")

def p_eff(u, eta_c, eta_d):
    """Effective power considering efficiency."""
    if u < 0:  # Charging
        return u * eta_c
    elif u > 0:  # Discharging
        return u / eta_d
    return 0.0

def feasible_actions(x, x_grid, C_bat, dt_hours, eta_c, eta_d, P_nom):
    """Generate feasible actions (same logic as in controller)."""
    actions = []
    
    for x_next in x_grid:
        # Calculate needed action
        p_eff_needed = (x - x_next) * C_bat / dt_hours
        
        # Convert to action u
        if p_eff_needed < 0:  # Need to charge
            u = p_eff_needed / eta_c
        elif p_eff_needed > 0:  # Need to discharge
            u = p_eff_needed * eta_d
        else:
            u = 0.0
        
        # Check if within limits
        if abs(u) <= P_nom:
            actions.append((u, x_next))
    
    return actions

# Generate feasible actions for current SOC
actions = feasible_actions(closest_soc, X_grid, C_bat, dt_hours, eta_c, eta_d, P_nom)

print(f"\n{'='*60}")
print(f"Feasible actions for SOC={closest_soc:.6f}:")
print(f"Total actions: {len(actions)}")
print(f"\nFirst 10 actions (charging):")
print(f"{'Action (u)':<15} {'Next SOC':<12} {'P_eff':<12} {'Description'}")
print("-" * 60)

for i in range(min(10, len(actions))):
    u, x_next = actions[i]
    p_e = p_eff(u, eta_c, eta_d)
    desc = "charge" if u < 0 else ("discharge" if u > 0 else "idle")
    print(f"{u:<15.6f} {x_next:<12.6f} {p_e:<12.6f} {desc}")

# Available power from solar
net_available = 1.356  # kW

# What action would we WANT to take?
u_desired = -net_available  # negative = charging
p_eff_desired = p_eff(u_desired, eta_c, eta_d)
soc_next_desired = closest_soc - (p_eff_desired * dt_hours) / C_bat

print(f"\n{'='*60}")
print(f"DESIRED ACTION (use all available power):")
print(f"u_desired: {u_desired:.6f} kW")
print(f"p_eff: {p_eff_desired:.6f} kW")
print(f"Next SOC (desired): {soc_next_desired:.6f}")

# Find the closest action to what we want
action_values = np.array([u for u, _ in actions])
closest_action_idx = np.argmin(np.abs(action_values - u_desired))
u_closest, x_next_closest = actions[closest_action_idx]

print(f"\nCLOSEST FEASIBLE ACTION:")
print(f"u_closest: {u_closest:.6f} kW")
print(f"Next SOC: {x_next_closest:.6f}")
print(f"Difference from desired: {abs(u_closest - u_desired):.6f} kW")

# Check if the desired action is feasible
if abs(u_desired) <= P_nom:
    print(f"\n✓ Desired action is within power limits (|{u_desired:.3f}| <= {P_nom})")
else:
    print(f"\n✗ Desired action exceeds power limits (|{u_desired:.3f}| > {P_nom})")

# Find the action that corresponds to maximum charging
max_charge_actions = [(u, x_next) for u, x_next in actions if u < 0]
if max_charge_actions:
    u_max_charge, soc_max_charge = min(max_charge_actions, key=lambda a: a[0])
    print(f"\nMAXIMUM CHARGING ACTION in feasible set:")
    print(f"u_max_charge: {u_max_charge:.6f} kW")
    print(f"Next SOC: {soc_max_charge:.6f}")
    
    # Is this the action being selected?
    u_current = -0.718  # from the data
    print(f"\nACTUAL ACTION SELECTED BY SDP:")
    print(f"u_actual: {u_current:.6f} kW")
    
    # Find which grid point this corresponds to
    soc_next_actual = closest_soc - (p_eff(u_current, eta_c, eta_d) * dt_hours) / C_bat
    closest_grid_idx = np.argmin(np.abs(X_grid - soc_next_actual))
    closest_grid_soc = X_grid[closest_grid_idx]
    
    print(f"Next SOC (implied): {soc_next_actual:.6f}")
    print(f"Closest grid SOC: {closest_grid_soc:.6f}")
    
    # Check if this action is in the feasible set
    matching_actions = [(u, x_next) for u, x_next in actions 
                       if abs(u - u_current) < 0.001]
    if matching_actions:
        print(f"\n✓ This action IS in the feasible set")
    else:
        print(f"\n✗ This action is NOT in the feasible set (interpolation?)")

