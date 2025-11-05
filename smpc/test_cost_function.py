"""
Test script to understand why SDP is making poor decisions with negative export prices.
"""

import numpy as np

# Parameters from config
C_bat = 10.0  # kWh
dt_hours = 0.25  # 15 minutes
eta_c = 0.90
eta_d = 0.90
c_s = 0.15  # buy price
c_f = -100  # export price (heavily penalized)
c_deg = 0.01  # degradation cost
P_lim = 10.0  # injection limit

# Situation from the data
solar_power = 1.776
load_power = 0.42
net_available = solar_power - load_power  # 1.356 kW available

print("=" * 60)
print("SCENARIO ANALYSIS: Why is SDP exporting instead of charging?")
print("=" * 60)
print(f"\nAvailable power: {net_available:.3f} kW (solar - load)")
print(f"Export price: {c_f} €/kWh (should heavily penalize exports!)")
print(f"Import price: {c_s} €/kWh")
print(f"Degradation cost: {c_deg} €/kWh throughput")

# Current SOC from data
current_soc = 0.178

print(f"\nCurrent SOC: {current_soc:.3f}")
print(f"Battery capacity: {C_bat} kWh")
print(f"Timestep: {dt_hours} hours")

def p_eff(u, eta_c, eta_d):
    """Effective power considering efficiency."""
    if u < 0:  # Charging
        return u * eta_c
    elif u > 0:  # Discharging
        return u / eta_d
    return 0.0

def grid_power(u, R, P_lim):
    """Grid power (negative = export)."""
    phi = max(0.0, u + R - P_lim)  # curtailment
    return -u - R + phi

def stage_cost(u, R, P_lim, c_s, c_f, dt_hours, c_deg, eta_c, eta_d):
    """Calculate stage cost."""
    P_g = grid_power(u, R, P_lim)
    P_g_plus = max(0.0, P_g)      # import
    P_g_minus = max(0.0, -P_g)    # export

    # Grid cost
    grid_cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours

    # Degradation cost
    if u < 0:  # Charging
        throughput = abs(u) * eta_c
    elif u > 0:  # Discharging
        throughput = abs(u) / eta_d
    else:
        throughput = 0.0

    degradation_cost = throughput * dt_hours * c_deg

    return grid_cost, degradation_cost, P_g, P_g_plus, P_g_minus

# Residual (net load)
R = net_available

print("\n" + "=" * 60)
print("OPTION A: Charge battery with ALL available power")
print("=" * 60)

u_A = -net_available  # negative = charging (SDP convention)
grid_cost_A, deg_cost_A, P_g_A, import_A, export_A = stage_cost(u_A, R, P_lim, c_s, c_f, dt_hours, c_deg, eta_c, eta_d)

print(f"Battery power (u): {u_A:.3f} kW (negative = charging)")
print(f"Grid power: {P_g_A:.6f} kW")
print(f"Grid import: {import_A:.6f} kW")
print(f"Grid export: {export_A:.6f} kW")
print(f"Grid cost: {grid_cost_A:.6f} €")
print(f"Degradation cost: {deg_cost_A:.6f} €")
print(f"TOTAL COST: {grid_cost_A + deg_cost_A:.6f} €")

# Calculate next SOC
p_eff_A = p_eff(u_A, eta_c, eta_d)
soc_next_A = current_soc - (p_eff_A * dt_hours) / C_bat
print(f"Next SOC: {soc_next_A:.6f}")

print("\n" + "=" * 60)
print("OPTION B: Current situation (charge less, export more)")
print("=" * 60)

# From the data: battery charging at 0.718 kW, exporting 0.638 kW
u_B = -0.718  # negative = charging
grid_cost_B, deg_cost_B, P_g_B, import_B, export_B = stage_cost(u_B, R, P_lim, c_s, c_f, dt_hours, c_deg, eta_c, eta_d)

print(f"Battery power (u): {u_B:.3f} kW")
print(f"Grid power: {P_g_B:.6f} kW")
print(f"Grid import: {import_B:.6f} kW")
print(f"Grid export: {export_B:.6f} kW")
print(f"Grid cost: {grid_cost_B:.6f} €")
print(f"Degradation cost: {deg_cost_B:.6f} €")
print(f"TOTAL COST: {grid_cost_B + deg_cost_B:.6f} €")

# Calculate next SOC
p_eff_B = p_eff(u_B, eta_c, eta_d)
soc_next_B = current_soc - (p_eff_B * dt_hours) / C_bat
print(f"Next SOC: {soc_next_B:.6f}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Cost difference (B - A): {(grid_cost_B + deg_cost_B) - (grid_cost_A + deg_cost_A):.6f} €")
print(f"Option A is cheaper by: {(grid_cost_B + deg_cost_B) - (grid_cost_A + deg_cost_A):.6f} €")
print(f"\nSOC difference: {soc_next_B - soc_next_A:.6f}")

# Now let's calculate terminal cost to see if that's affecting the decision
print("\n" + "=" * 60)
print("TERMINAL COST ANALYSIS")
print("=" * 60)

def terminal_cost(x, soc_target, weight, C_bat, c_s, c_f):
    """Terminal cost with clamped c_f."""
    c_f_effective = max(0.0, c_f)
    return -x * C_bat * (c_s + c_f_effective) / 2.0

soc_target = 0.5
terminal_weight = 1.0

term_cost_A = terminal_cost(soc_next_A, soc_target, terminal_weight, C_bat, c_s, c_f)
term_cost_B = terminal_cost(soc_next_B, soc_target, terminal_weight, C_bat, c_s, c_f)

print(f"Terminal cost at SOC={soc_next_A:.3f} (Option A): {term_cost_A:.6f} €")
print(f"Terminal cost at SOC={soc_next_B:.3f} (Option B): {term_cost_B:.6f} €")
print(f"Terminal cost difference (A - B): {term_cost_A - term_cost_B:.6f} €")
print(f"  (negative = Option A has better terminal value)")

print("\n" + "=" * 60)
print("TOTAL COST INCLUDING TERMINAL")
print("=" * 60)
total_A = (grid_cost_A + deg_cost_A) + term_cost_A
total_B = (grid_cost_B + deg_cost_B) + term_cost_B

print(f"Option A total: {total_A:.6f} €")
print(f"Option B total: {total_B:.6f} €")
print(f"\nOption A should be preferred by: {total_B - total_A:.6f} €")

if total_A < total_B:
    print("\n✓ EXPECTED: Option A (charge more) is better!")
else:
    print("\n✗ PROBLEM: Option B (export more) appears better - BUG IN OPTIMIZATION!")
