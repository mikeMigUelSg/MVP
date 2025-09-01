import os
import sys
from datetime import datetime
import pandas as pd

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ess.battery import Battery
from ess.simulator import EnergyArbitrageSimulator
from ess.strategies import OptimalArbitrageStrategy


def test_optimal_strategy_generates_schedule():
    start = datetime(2024, 1, 1)
    # Two days of 15-minute data for lookahead
    times = pd.date_range(start, periods=96 * 2, freq="15min")

    # Constant consumption 1 kW (0.25 kWh per step)
    consumption_df = pd.DataFrame({"kwh": 0.25, "kw": 1.0}, index=times)

    # Price low in first half of each day, high in second half
    prices = [0.05 if t.hour < 12 else 0.2 for t in times]
    prices_df = pd.DataFrame({"price_eur_per_kwh": prices}, index=times)

    battery = Battery(capacity_kwh=10, max_charge_kw=5, max_discharge_kw=5)
    strategy = OptimalArbitrageStrategy()
    sim = EnergyArbitrageSimulator(battery, strategy)

    # Run for first day (48h lookahead requires two days of data)
    sim.run(consumption_df, prices_df, start, start)

    # Daily schedule should be computed with non-idle actions
    schedule = strategy.daily_schedule
    assert schedule is not None
    assert len(schedule) == 96 * 2
    assert schedule["action"].isin(["charge", "discharge"]).any()

