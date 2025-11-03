"""
Análise das oscilações na bateria durante períodos de baixa carga e produção constante.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('results/data/results_sdp.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Focus on morning period (9:00 - 17:00) where we see oscillations
morning_mask = (df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour <= 17)
df_morning = df[morning_mask].copy()

# Create figure with subplots
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# 1. Solar production and load
ax = axes[0]
ax.plot(df_morning['timestamp'], df_morning['solar_power'], label='Solar (kW)', color='orange', linewidth=2)
ax.plot(df_morning['timestamp'], df_morning['load_power'], label='Load (kW)', color='blue', linewidth=2)
ax.set_ylabel('Power (kW)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Solar Production and Load (9:00 - 17:00)')

# 2. Residual (solar - load)
ax = axes[1]
residual = df_morning['solar_power'] - df_morning['load_power']
ax.plot(df_morning['timestamp'], residual, label='Residual = PV - Load (kW)', color='green', linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_ylabel('Residual (kW)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Net Power (Residual)')

# 3. Battery power (showing oscillations)
ax = axes[2]
ax.plot(df_morning['timestamp'], df_morning['battery_power'], label='Battery Power (kW)',
        color='red', linewidth=2, marker='o', markersize=4)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.axhline(y=5, color='red', linestyle=':', alpha=0.5, label='Max charge')
ax.axhline(y=-5, color='blue', linestyle=':', alpha=0.5, label='Max discharge')
ax.set_ylabel('Battery Power (kW)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Battery Power (OSCILLATIONS VISIBLE HERE)')

# 4. Battery SOC
ax = axes[3]
ax.plot(df_morning['timestamp'], df_morning['battery_soc'] * 100, label='SOC (%)',
        color='purple', linewidth=2)
ax.set_ylabel('SOC (%)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Battery State of Charge')

# 5. Grid power
ax = axes[4]
ax.plot(df_morning['timestamp'], df_morning['grid_power'], label='Grid Power (kW)',
        color='brown', linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.fill_between(df_morning['timestamp'], 0, df_morning['grid_power'],
                 where=(df_morning['grid_power'] > 0), alpha=0.3, color='red', label='Import')
ax.fill_between(df_morning['timestamp'], 0, df_morning['grid_power'],
                 where=(df_morning['grid_power'] < 0), alpha=0.3, color='green', label='Export')
ax.set_ylabel('Grid Power (kW)')
ax.set_xlabel('Time')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Grid Power (+ = import, - = export)')

plt.tight_layout()
plt.savefig('results/plots/oscillation_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved to: results/plots/oscillation_analysis.png")

# Statistical analysis of oscillations
print("\n" + "="*60)
print("OSCILLATION ANALYSIS")
print("="*60)

# Find periods with high battery power variance
window_size = 4  # 1 hour window (4 x 15min)
df_morning['battery_power_std'] = df_morning['battery_power'].rolling(window_size).std()

high_osc_mask = df_morning['battery_power_std'] > 2.0  # High variance threshold
high_osc_periods = df_morning[high_osc_mask]

print(f"\nPeriods with high battery oscillation (std > 2.0 kW):")
print(f"Number of time steps: {len(high_osc_periods)}")

if len(high_osc_periods) > 0:
    print(f"\nExample oscillation period:")
    example_start = high_osc_periods.index[0] - 2
    example_end = high_osc_periods.index[0] + 6
    example = df.iloc[example_start:example_end]

    print("\nTime              | Solar | Load  | Residual | Battery | SOC   | Grid")
    print("-" * 80)
    for idx, row in example.iterrows():
        residual = row['solar_power'] - row['load_power']
        print(f"{row['timestamp']} | {row['solar_power']:5.2f} | {row['load_power']:5.2f} | "
              f"{residual:8.2f} | {row['battery_power']:7.2f} | {row['battery_soc']:5.2f} | "
              f"{row['grid_power']:7.2f}")

# Battery cycling analysis
battery_charge = df_morning[df_morning['battery_power'] > 0]['battery_power'].sum() * 0.25  # kWh
battery_discharge = df_morning[df_morning['battery_power'] < 0]['battery_power'].sum() * 0.25  # kWh

print(f"\n\nBattery cycling in morning period (9:00 - 17:00):")
print(f"Total charged:    {battery_charge:6.2f} kWh")
print(f"Total discharged: {abs(battery_discharge):6.2f} kWh")
print(f"Cycling ratio:    {min(battery_charge, abs(battery_discharge)) / max(battery_charge, abs(battery_discharge)):.2%}")

# Number of sign changes (charge/discharge switches)
sign_changes = (np.diff(np.sign(df_morning['battery_power'])) != 0).sum()
print(f"\nBattery action sign changes: {sign_changes}")
print(f"Average time between switches: {len(df_morning) * 15 / (sign_changes + 1):.1f} minutes")

print("\n" + "="*60)
