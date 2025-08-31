"""
ess/io.py - Data loading and processing functions
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional


def fetch_ren_prices(start_date: datetime, end_date: datetime, culture: str = "pt-PT") -> pd.DataFrame:
    """
    Fetch OMIE electricity prices from REN API.
    
    Parameters
    ----------
    start_date : datetime
        Start date for price data
    end_date : datetime
        End date for price data
    culture : str
        Culture/region identifier (default: "pt-PT")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, price_eur_per_mwh
    """
    url = "https://servicebus.ren.pt/datahubapi/electricity/ElectricityMarketPricesDaily"
    pt_data = []
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        params = {
            "culture": culture,
            "date": date_str
        }
        
        try:
            print(f"Fetching PT data for {date_str}...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            json_data = response.json()
            
            # Get hours and PT prices
            hours = json_data["xAxis"]["categories"]
            pt_series = next(s for s in json_data["series"] if s["name"] == "PT")
            pt_prices = pt_series["data"]
            
            for hour_str, price in zip(hours, pt_prices):
                hour = int(hour_str)
                if hour == 24:
                    # Hour 24 means midnight of the next day
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
                else:
                    timestamp = datetime.strptime(f"{date_str} {hour:02d}:00", "%Y-%m-%d %H:%M")
                pt_data.append({
                    "datetime": timestamp,
                    "price_eur_per_mwh": price
                })
        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(pt_data)
    if not df.empty:
        df.set_index('datetime', inplace=True)
    return df


def load_consumption_profile(filepath: str, profile_column: str = "BTN A") -> pd.DataFrame:
    """
    Load E-REDES consumption profile from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file
    profile_column : str
        Column name for the profile (default: "BTN A")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 15-minute intervals and permil values
    """
    # Read the Excel file
    df = pd.read_excel(filepath)
    
    # Set header from 3rd row and clean
    df.columns = df.iloc[2]
    df = df.drop([0, 1, 2]).reset_index(drop=True)
    
    # Choose columns: 1st NaN (date), 2nd NaN (hour) + valid ones
    is_nan = pd.isna(df.columns)
    first_nan_idx = np.flatnonzero(is_nan)[0]
    second_nan_idx = np.flatnonzero(is_nan)[1]
    non_nan_idx = list(np.flatnonzero(~is_nan))
    df = df.iloc[:, [first_nan_idx, second_nan_idx] + non_nan_idx]
    
    # Rename columns
    df.columns = ["date", "hour"] + df.columns.tolist()[2:]
    
    # Create proper datetime with 15-minute intervals
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    
    # Each day has 96 intervals (24 hours * 4 intervals per hour)
    intervals_per_day = 96
    df["interval_in_day"] = df.index % intervals_per_day
    
    # Convert interval number to time
    df["minutes_from_midnight"] = (df["interval_in_day"] + 1) * 15
    
    # Create proper datetime
    df["time"] = pd.to_datetime(df["date_only"].astype(str)) + pd.to_timedelta(df["minutes_from_midnight"], unit='minutes')
    
    # Select only time and profile column
    result = df[["time", profile_column]].copy()
    result.columns = ["datetime", "permil"]
    result.set_index("datetime", inplace=True)
    
    return result


def unnormalize_consumption(annual_consumption_kwh: float, permil_value: float) -> float:
    """
    Convert permil (per thousand) value to actual kWh.
    
    Parameters
    ----------
    annual_consumption_kwh : float
        Total annual consumption in kWh
    permil_value : float
        Normalized value (per thousand)
    
    Returns
    -------
    float
        Actual consumption in kWh
    """
    return permil_value / 1000 * annual_consumption_kwh


def resample_prices_to_15min(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly prices to 15-minute intervals.
    Each hour's price is repeated 4 times.
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with hourly prices
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 15-minute prices
    """
    # Resample to 15 minutes, forward-filling the hourly values
    prices_15min = prices_df.resample('15min').ffill()
    return prices_15min


def align_data_to_period(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Align dataframe to specific date range.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    start_date : datetime
        Start date
    end_date : datetime
        End date (inclusive)
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    # Ensure end_date includes the full day
    end_date = end_date.replace(hour=23, minute=45, second=0, microsecond=0)
    
    # Filter the dataframe
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask].copy()


def prepare_simulation_data(
    consumption_profile_path: str,
    annual_consumption_kwh: float,
    start_date: datetime,
    end_date: datetime,
    profile_column: str = "BTN A"
) -> tuple:
    """
    Prepare all data needed for simulation.
    
    Parameters
    ----------
    consumption_profile_path : str
        Path to E-REDES Excel file
    annual_consumption_kwh : float
        Annual consumption to scale the profile
    start_date : datetime
        Simulation start date
    end_date : datetime
        Simulation end date
    profile_column : str
        Profile column name
    
    Returns
    -------
    tuple
        (consumption_df, prices_df) both with 15-minute resolution
    """
    # Load consumption profile
    print("Loading consumption profile...")
    consumption_df = load_consumption_profile(consumption_profile_path, profile_column)
    
    # Unnormalize to actual kWh
    consumption_df["kwh"] = consumption_df["permil"].apply(
        lambda x: unnormalize_consumption(annual_consumption_kwh, x)
    )
    
    # Calculate power in kW (15 min = 0.25 hours)
    consumption_df["kw"] = consumption_df["kwh"] / 0.25
    
    # Fetch prices (add one extra day for lookahead)
    print("Fetching OMIE prices...")
    prices_df = fetch_ren_prices(start_date, end_date + timedelta(days=1))
    
    # Convert to EUR/kWh
    prices_df["price_eur_per_kwh"] = prices_df["price_eur_per_mwh"] / 1000
    
    # Resample to 15 minutes
    prices_15min = resample_prices_to_15min(prices_df)
    
    # Align both datasets to simulation period
    consumption_aligned = align_data_to_period(consumption_df, start_date, end_date)
    prices_aligned = prices_15min  # Keep extra day for lookahead
    
    return consumption_aligned, prices_aligned


def save_results(results_df: pd.DataFrame, filepath: str):
    """Save simulation results to CSV."""
    results_df.to_csv(filepath)
    print(f"Results saved to {filepath}")