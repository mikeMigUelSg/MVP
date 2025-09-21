"""
ess/tariff.py - REFACTORED VERSION with modular invoice components
"""

import pandas as pd
from datetime import datetime


def _is_summer(date: datetime) -> bool:
    """Approximate check for Portuguese summer season (Apr-Oct)."""
    return 4 <= date.month <= 10


def _minutes(t) -> int:
    return t.hour * 60 + t.minute


def _period_daily(ts: datetime, option: str, season: str) -> str:
    m = _minutes(ts.time())
    if option == 'bi':
        if m >= 22 * 60 or m < 8 * 60:
            return 'vazio'
        return 'fora_vazio'
    if option == 'tri':
        if m >= 22 * 60 or m < 8 * 60:
            return 'vazio'
        if season == 'winter':
            if (8 * 60 + 30) <= m < (10 * 60 + 30) or (18 * 60) <= m < (20 * 60 + 30):
                return 'ponta'
            return 'cheias'
        else:  # summer
            if (10 * 60 + 30) <= m < (12 * 60) or (19 * 60 + 30) <= m < (21 * 60):
                return 'ponta'
            return 'cheias'
    return 'simples'


def _period_weekly(ts: datetime, option: str, season: str) -> str:
    m = _minutes(ts.time())
    wd = ts.weekday()  # Monday=0
    if option == 'bi':
        if wd <= 4:  # weekdays
            return 'vazio' if m < 7 * 60 else 'fora_vazio'
        if wd == 5:  # Saturday
            if season == 'winter':
                if m < 9 * 60 + 30 or 13 * 60 <= m < 18 * 60 + 30 or m >= 22 * 60:
                    return 'vazio'
                return 'fora_vazio'
            else:
                if m < 9 * 60 or 14 * 60 <= m < 20 * 60 or m >= 22 * 60:
                    return 'vazio'
                return 'fora_vazio'
        return 'vazio'  # Sunday
    if option == 'tri':
        if wd <= 4:  # weekdays
            if m < 7 * 60:
                return 'vazio'
            if season == 'winter':
                if 9 * 60 + 30 <= m < 12 * 60 or 18 * 60 + 30 <= m < 21 * 60:
                    return 'ponta'
                return 'cheias'
            else:
                if 9 * 60 + 15 <= m < 12 * 60 + 15:
                    return 'ponta'
                return 'cheias'
        if wd == 5:  # Saturday
            if season == 'winter':
                if 9 * 60 + 30 <= m < 13 * 60 or 18 * 60 + 30 <= m < 22 * 60:
                    return 'cheias'
                return 'vazio'
            else:
                if 9 * 60 <= m < 14 * 60 or 20 * 60 <= m < 22 * 60:
                    return 'cheias'
                return 'vazio'
        return 'vazio'  # Sunday
    return 'simples'


def get_tariff_period(ts: datetime, option: str, cycle: str) -> str:
    season = 'summer' if _is_summer(ts) else 'winter'
    if cycle == 'daily':
        return _period_daily(ts, option, season)
    return _period_weekly(ts, option, season)


def apply_indexed_tariff(prices_df: pd.DataFrame, tariff_cfg: dict) -> pd.DataFrame:
    """
    REFACTORED: Apply indexed tariff and return electricity term components WITHOUT VAT.
    
    Returns prices_df with additional columns:
    - tariff_period: Time-of-use period
    - price_omie_adjusted_eur_kwh: OMIE price * (1 + losses) * k1
    - k2_eur_kwh: K2 network access component  
    - tariff_energy_eur_kwh: Time-of-use tariff component
    - price_electricity_base_eur_kwh: Total electricity term (OMIE_adjusted + K2 + TAR)
    
    Note: VAT is NOT applied here - it's handled in the billing engine
    """
    idx_cfg = tariff_cfg['indexed']
    option = idx_cfg['option']
    cycle = idx_cfg['cycle']
    k1 = idx_cfg['k1']
    k2 = idx_cfg['k2_eur_kwh']
    losses = idx_cfg['losses_pct']
    rates = idx_cfg['tariff_energy_eur_kwh']

    periods = []
    omie_adjusted = []
    k2_list = []
    tariffs = []
    electricity_base = []

    for ts, row in prices_df.iterrows():
        # Get tariff period
        period = get_tariff_period(ts, option, cycle)
        periods.append(period)
        
        # OMIE adjusted for losses and k1
        omie_adj = row['price_omie_eur_kwh'] * (1 + losses) * k1
        omie_adjusted.append(omie_adj)
        
        # K2 network access
        k2_list.append(k2)
        
        # Time-of-use tariff
        if option == 'simples':
            tar = rates['simples']
        elif option == 'bi':
            tar = rates['bi'][period]
        elif option == 'tri':
            tar = rates['tri'][period]
        else:
            tar = 0.0
        tariffs.append(tar)
        
        # Total electricity base (before VAT)
        energy_base = omie_adj + k2 + tar
        electricity_base.append(energy_base)

    # Add columns to dataframe
    prices_df['tariff_period'] = periods
    prices_df['price_omie_adjusted_eur_kwh'] = omie_adjusted
    prices_df['k2_eur_kwh'] = k2_list
    prices_df['tariff_energy_eur_kwh'] = tariffs
    prices_df['price_electricity_base_eur_kwh'] = electricity_base
    
    # Keep final price for backward compatibility (includes everything with VAT)
    # This will be recalculated properly in billing engine
    prices_df['price_final_eur_kwh'] = prices_df['price_electricity_base_eur_kwh']
    
    return prices_df
