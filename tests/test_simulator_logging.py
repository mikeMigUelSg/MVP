import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Ensure project root is on sys.path for importing the package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ess.battery import Battery
from ess.simulator import EnergyArbitrageSimulator


def test_simulation_logs_once_per_day_and_counts_missing_steps(capsys):
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 1)

    # Consumption data for full day (96 steps)
    times = pd.date_range(start, periods=96, freq="15min")
    consumption_df = pd.DataFrame({
        "kwh": 0.25,  # arbitrary values
        "kw": 1.0,
    }, index=times)

    # Price data missing first hour (start at 01:00)
    price_times = pd.date_range(start + timedelta(hours=1), periods=92, freq="15min")
    prices_df = pd.DataFrame({
        "price_eur_per_kwh": 0.1
    }, index=price_times)

    battery = Battery(capacity_kwh=10, max_charge_kw=5, max_discharge_kw=5)
    sim = EnergyArbitrageSimulator(battery)
    sim.run(consumption_df, prices_df, start, end)

    captured = capsys.readouterr().out
    # Log "Simulating <date>..." should appear only once despite missing early prices
    assert captured.count("Simulating 2024-01-01") == 1
    # Step count should include missing price periods
    assert "Simulation completed: 96 steps" in captured
