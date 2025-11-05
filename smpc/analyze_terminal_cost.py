"""
Analyze terminal cost formula to see if it's undervaluing stored energy.
"""

import numpy as np

C_bat = 10.0  # kWh
c_s = 0.15  # import price
c_f = -100  # export price (penalized)

print("=" * 60)
print("TERMINAL COST ANALYSIS")
print("=" * 60)

print(f"\nImport price (c_s): €{c_s}/kWh")
print(f"Export price (c_f): €{c_f}/kWh (heavily penalized!)")
print(f"Battery capacity: {C_bat} kWh")

def terminal_cost_current(x, c_s, c_f, C_bat):
    """Current terminal cost formula with c_f clamped to 0."""
    c_f_effective = max(0.0, c_f)
    return -x * C_bat * (c_s + c_f_effective) / 2.0

def terminal_cost_proposed(x, c_s, c_f, C_bat):
    """Proposed fix: don't divide by 2 when c_f <= 0."""
    if c_f > 0:
        # Can profitably export, so value is average of buy/sell
        return -x * C_bat * (c_s + c_f) / 2.0
    else:
        # Cannot profitably export, so value is just avoiding purchase
        return -x * C_bat * c_s

print("\n" + "=" * 60)
print("SCENARIO: Storing 1 kWh of energy")
print("=" * 60)

# Starting at SOC = 0.5, ending at SOC = 0.6
# This represents storing 1 kWh (0.1 * 10 kWh = 1 kWh)
soc_before = 0.5
soc_after = 0.6

cost_before_current = terminal_cost_current(soc_before, c_s, c_f, C_bat)
cost_after_current = terminal_cost_current(soc_after, c_s, c_f, C_bat)
value_current = cost_before_current - cost_after_current

cost_before_proposed = terminal_cost_proposed(soc_before, c_s, c_f, C_bat)
cost_after_proposed = terminal_cost_proposed(soc_after, c_s, c_f, C_bat)
value_proposed = cost_before_proposed - cost_after_proposed

print(f"\nSOC before: {soc_before}")
print(f"SOC after:  {soc_after}")
print(f"Energy stored: 1 kWh")

print(f"\nCURRENT FORMULA (with /2):")
print(f"  Terminal cost before: €{cost_before_current:.4f}")
print(f"  Terminal cost after:  €{cost_after_current:.4f}")
print(f"  Value of 1 kWh stored: €{value_current:.4f}")
print(f"  (This is HALF of the import price!)")

print(f"\nPROPOSED FORMULA (no /2 when c_f <= 0):")
print(f"  Terminal cost before: €{cost_before_proposed:.4f}")
print(f"  Terminal cost after:  €{cost_after_proposed:.4f}")
print(f"  Value of 1 kWh stored: €{value_proposed:.4f}")
print(f"  (This equals the import price - correct!)")

print("\n" + "=" * 60)
print("IMPACT ON DECISION MAKING")
print("=" * 60)

# From earlier analysis: exporting 0.638 kW for 15 minutes
export_cost = 0.638 * 100 * 0.25  # €15.95

# Energy that could be stored instead
energy_that_could_be_stored = 0.638 * 0.25  # 0.1595 kWh

value_stored_current = energy_that_could_be_stored * value_current
value_stored_proposed = energy_that_could_be_stored * value_proposed

print(f"\nExport cost: €{export_cost:.4f}")
print(f"Energy that could be stored: {energy_that_could_be_stored:.4f} kWh")

print(f"\nWith CURRENT formula:")
print(f"  Future value of storing instead: €{value_stored_current:.4f}")
print(f"  Net benefit of charging more: €{value_stored_current - export_cost:.4f}")
print(f"  → Exporting appears better by €{export_cost - value_stored_current:.4f}")

print(f"\nWith PROPOSED formula:")
print(f"  Future value of storing instead: €{value_proposed * energy_that_could_be_stored:.4f}")
print(f"  Net benefit of charging more: €{value_stored_proposed - export_cost:.4f}")
print(f"  → Exporting still worse by €{export_cost - value_stored_proposed:.4f}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

if value_current < value_proposed:
    print(f"\n✗ CURRENT formula undervalues stored energy by {((value_proposed/value_current) - 1) * 100:.1f}%")
    print("  This is a BUG that could cause suboptimal decisions!")
    print("\n  The terminal cost formula should NOT divide by 2 when export")
    print("  price is negative, because you would never choose to export.")
else:
    print("\n✓ Formula is correct")

# But wait - even with the current formula, exporting should still be worse!
# Let me check if there's a different issue...

print("\n" + "=" * 60)
print("WAIT - EVEN WITH UNDERVALUED TERMINAL COST...")
print("=" * 60)
print(f"\nExport cost: €{export_cost:.4f}")
print(f"Terminal value gain (current): €{value_stored_current:.4f}")
print(f"Difference: €{export_cost - value_stored_current:.4f}")
print("\nExporting is STILL much worse, even with the bug!")
print("So the terminal cost bug doesn't fully explain the problem...")

