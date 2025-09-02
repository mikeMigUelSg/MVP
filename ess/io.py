"""
ess/io.py - Enhanced data loading and processing functions
Fixed to handle time zone transitions and data quality issues
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from pathlib import Path
import warnings


PRICE_CACHE_FILE = Path("data/spot_prices.parquet")


def fetch_ren_prices(start_date: datetime, end_date: datetime, culture: str = "pt-PT") -> pd.DataFrame:
    """
    Fetch OMIE electricity prices from REN API with enhanced error handling.
    
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
    failed_dates = []
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        params = {
            "culture": culture,
            "date": date_str
        }
        
        try:
            print(f"Fetching PT data for {date_str}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            
            # Validate response structure
            if not json_data or "xAxis" not in json_data or "series" not in json_data:
                print(f"Invalid response structure for {date_str}")
                failed_dates.append(date_str)
                current_date += timedelta(days=1)
                continue
            
            # Get hours and PT prices
            hours = json_data["xAxis"]["categories"]
            pt_series = next((s for s in json_data["series"] if s["name"] == "PT"), None)
            
            if pt_series is None:
                print(f"PT data not found in response for {date_str}")
                failed_dates.append(date_str)
                current_date += timedelta(days=1)
                continue
            
            pt_prices = pt_series["data"]
            
            if len(hours) != len(pt_prices):
                print(f"Mismatched hours/prices length for {date_str}: {len(hours)} vs {len(pt_prices)}")
                failed_dates.append(date_str)
                current_date += timedelta(days=1)
                continue
            
            # Process hourly data
            daily_data = []
            for hour_str, price in zip(hours, pt_prices):
                try:
                    # REN provides hours numbered 1-24 where "1" corresponds to 00:00
                    hour = int(hour_str) - 1
                    
                    # Handle special cases for DST transitions
                    if hour < 0 or hour > 23:
                        print(f"Invalid hour {hour} for {date_str}")
                        continue
                    
                    # Create timestamp
                    timestamp = datetime.strptime(f"{date_str} {hour:02d}:00", "%Y-%m-%d %H:%M")
                    
                    # Validate price
                    if price is None or not isinstance(price, (int, float)):
                        print(f"Invalid price {price} for {date_str} {hour:02d}:00")
                        continue
                    
                    # Store valid data point
                    daily_data.append({
                        "datetime": timestamp,
                        "price_eur_per_mwh": float(price)
                    })
                    
                except (ValueError, TypeError) as e:
                    print(f"Error processing hour {hour_str} for {date_str}: {e}")
                    continue
            
            # Add daily data if we got reasonable amount
            if len(daily_data) >= 20:  # At least 20 hours of data
                pt_data.extend(daily_data)
            else:
                print(f"Insufficient valid data points for {date_str}: {len(daily_data)}")
                failed_dates.append(date_str)
                
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching data for {date_str}: {e}")
            failed_dates.append(date_str)
        except (KeyError, ValueError, TypeError) as e:
            print(f"Data parsing error for {date_str}: {e}")
            failed_dates.append(date_str)
        except Exception as e:
            print(f"Unexpected error fetching data for {date_str}: {e}")
            failed_dates.append(date_str)
        
        current_date += timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame(pt_data)
    
    if df.empty:
        print("WARNING: No price data was successfully fetched!")
        return df
    
    # Clean and validate DataFrame
    df = clean_price_dataframe(df)
    
    # Report summary
    total_days = (end_date - start_date).days + 1
    successful_days = total_days - len(failed_dates)
    print(f"Price data fetch summary: {successful_days}/{total_days} days successful")
    
    if failed_dates:
        print(f"Failed dates: {', '.join(failed_dates[:5])}{'...' if len(failed_dates) > 5 else ''}")
    
    return df


def clean_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate price DataFrame."""
    if df.empty:
        return df
    
    # Set datetime index
    df.set_index('datetime', inplace=True)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Remove duplicates (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Validate price ranges
    price_col = 'price_eur_per_mwh'
    if price_col in df.columns:
        # Flag extreme values
        extreme_low = df[price_col] < -500  # Below -500 EUR/MWh
        extreme_high = df[price_col] > 3000  # Above 3000 EUR/MWh
        
        if extreme_low.any():
            count = extreme_low.sum()
            print(f"WARNING: {count} extremely low prices detected (< -500 EUR/MWh)")
            
        if extreme_high.any():
            count = extreme_high.sum()
            print(f"WARNING: {count} extremely high prices detected (> 3000 EUR/MWh)")
        
        # Optional: Cap extreme values
        # df.loc[extreme_low, price_col] = -100
        # df.loc[extreme_high, price_col] = 1000
    
    return df


def load_cached_prices(start_year: int = 2015, refresh: bool = False) -> pd.DataFrame:
    """Load cached price data or fetch and cache if missing."""
    if PRICE_CACHE_FILE.exists() and not refresh:
        return pd.read_parquet(PRICE_CACHE_FILE)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime.utcnow()
    df = fetch_ren_prices(start_date, end_date)
    PRICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICE_CACHE_FILE)
    return df


def load_consumption_profile(filepath: str, profile_column: str = "BTN A") -> pd.DataFrame:
    """
    Load E-REDES consumption profile from Excel file with enhanced error handling.
    """
    try:
        # Read the Excel file with error handling
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
        except:
            # Fallback to xlrd engine
            df = pd.read_excel(filepath, engine='xlrd')
        
        if df.empty:
            raise ValueError("Excel file is empty")
        
        # Set header from 3rd row and clean
        if len(df) < 3:
            raise ValueError("Excel file has insufficient rows")
        
        df.columns = df.iloc[2]
        df = df.drop([0, 1, 2]).reset_index(drop=True)
        
        # Choose columns: 1st NaN (date), 2nd NaN (hour) + valid ones
        is_nan = pd.isna(df.columns)
        nan_indices = np.flatnonzero(is_nan)
        
        if len(nan_indices) < 2:
            raise ValueError("Expected format with at least 2 unnamed columns not found")
        
        first_nan_idx = nan_indices[0]
        second_nan_idx = nan_indices[1]
        non_nan_idx = list(np.flatnonzero(~is_nan))
        
        df = df.iloc[:, [first_nan_idx, second_nan_idx] + non_nan_idx]
        
        # Rename columns
        new_columns = ["date", "hour"] + df.columns.tolist()[2:]
        df.columns = new_columns
        
        # Check if profile column exists
        if profile_column not in df.columns:
            available_columns = [col for col in df.columns if col not in ["date", "hour"]]
            raise ValueError(f"Profile column '{profile_column}' not found. Available: {available_columns}")
        
        # Create proper datetime with 15-minute intervals
        df["date_only"] = pd.to_datetime(df["date"], errors='coerce').dt.date
        
        # Remove rows with invalid dates
        valid_dates = df["date_only"].notna()
        if not valid_dates.any():
            raise ValueError("No valid dates found in the data")
        
        df = df[valid_dates].copy()
        
        # Each day has 96 intervals (24 hours * 4 intervals per hour)
        intervals_per_day = 96
        df["interval_in_day"] = df.index % intervals_per_day
        
        # Convert interval number to time (intervals 1-96 become 0:15-24:00)
        df["minutes_from_midnight"] = (df["interval_in_day"] + 1) * 15
        
        # Handle the 24:00 case (interval 96) by moving to next day 00:00
        next_day_mask = df["minutes_from_midnight"] > 1440  # > 24 hours
        df.loc[next_day_mask, "date_only"] = df.loc[next_day_mask, "date_only"] + pd.Timedelta(days=1)
        df.loc[next_day_mask, "minutes_from_midnight"] = 0
        
        # Create proper datetime
        df["time"] = (pd.to_datetime(df["date_only"].astype(str)) + 
                      pd.to_timedelta(df["minutes_from_midnight"], unit='minutes'))
        
        # Select only time and profile column, handling potential data type issues
        result = df[["time", profile_column]].copy()
        result.columns = ["datetime", "permil"]
        
        # Convert permil to numeric, handling errors
        result["permil"] = pd.to_numeric(result["permil"], errors='coerce')
        
        # Remove rows with invalid permil values
        valid_permil = result["permil"].notna()
        result = result[valid_permil].copy()
        
        if result.empty:
            raise ValueError("No valid consumption data after cleaning")
        
        # Set index and sort
        result.set_index("datetime", inplace=True)
        result.sort_index(inplace=True)
        
        # Remove duplicates
        result = result[~result.index.duplicated(keep='first')]
        
        print(f"Loaded consumption profile: {len(result)} data points from {result.index[0]} to {result.index[-1]}")
        
        return result
        
    except Exception as e:
        print(f"Error loading consumption profile: {e}")
        raise


def unnormalize_consumption(annual_consumption_kwh: float, permil_value: float) -> float:
    """
    Convert permil (per thousand) value to actual kWh with validation.
    """
    if not isinstance(permil_value, (int, float)) or np.isnan(permil_value):
        return 0.0
    
    result = permil_value / 1000 * annual_consumption_kwh
    
    # Sanity check - residential consumption shouldn't be too extreme
    if result < 0:
        warnings.warn(f"Negative consumption calculated: {result:.4f} kWh")
        return 0.0
    elif result > 20:  # More than 20 kWh in 15 minutes is very high for residential
        warnings.warn(f"Very high consumption calculated: {result:.4f} kWh in 15 minutes")
    
    return result


def resample_prices_to_15min(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly prices to 15-minute intervals with enhanced handling.
    """
    if prices_df.empty:
        return prices_df
    
    try:
        # Ensure index is datetime and sorted
        if not isinstance(prices_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be datetime")
        
        if not prices_df.index.is_monotonic_increasing:
            prices_df = prices_df.sort_index()
        
        # Check for and handle DST transitions
        time_diffs = prices_df.index.to_series().diff()
        unusual_gaps = time_diffs[(time_diffs < pd.Timedelta('50min')) | (time_diffs > pd.Timedelta('70min'))]
        
        if len(unusual_gaps) > 0:
            print(f"WARNING: {len(unusual_gaps)} unusual time gaps detected in price data")
            print("This might be due to DST transitions or data quality issues")
        
        # Resample to 15 minutes using forward fill
        prices_15min = prices_df.resample('15min').ffill()
        
        # Fill any remaining NaN values
        prices_15min = prices_15min.fillna(method='bfill').fillna(method='ffill')
        
        print(f"Resampled prices: {len(prices_df)} hourly -> {len(prices_15min)} 15-minute intervals")
        
        return prices_15min
        
    except Exception as e:
        print(f"Error resampling prices: {e}")
        return prices_df


def align_data_to_period(df: pd.DataFrame, start_date: datetime, end_date: datetime, 
                        allow_partial: bool = True) -> pd.DataFrame:
    """
    Align dataframe to specific date range with enhanced handling.
    """
    if df.empty:
        return df
    
    try:
        # Ensure end_date includes the full day
        end_date_inclusive = end_date.replace(hour=23, minute=45, second=0, microsecond=0)
        
        # Filter the dataframe
        mask = (df.index >= start_date) & (df.index <= end_date_inclusive)
        result = df.loc[mask].copy()
        
        # Calculate coverage
        expected_periods = pd.date_range(start_date, end_date_inclusive, freq='15min')
        actual_periods = len(result)
        expected_count = len(expected_periods)
        coverage = actual_periods / expected_count * 100 if expected_count > 0 else 0
        
        print(f"Data alignment: {actual_periods}/{expected_count} periods ({coverage:.1f}% coverage)")
        
        if coverage < 80 and not allow_partial:
            warnings.warn(f"Low data coverage: {coverage:.1f}% for period {start_date} to {end_date}")
        
        return result
        
    except Exception as e:
        print(f"Error aligning data to period: {e}")
        return df


def fill_missing_data(df: pd.DataFrame, start_date: datetime, end_date: datetime, 
                     method: str = 'interpolate') -> pd.DataFrame:
    """
    Fill missing data points in time series.
    """
    if df.empty:
        return df
    
    try:
        # Create complete time index
        end_date_inclusive = end_date.replace(hour=23, minute=45, second=0, microsecond=0)
        complete_index = pd.date_range(start_date, end_date_inclusive, freq='15min')
        
        # Reindex to complete time series
        df_complete = df.reindex(complete_index)
        
        # Count missing values
        missing_count = df_complete.isnull().sum().sum()
        total_count = len(df_complete) * len(df_complete.columns)
        missing_pct = missing_count / total_count * 100 if total_count > 0 else 0
        
        if missing_count > 0:
            print(f"Filling {missing_count} missing values ({missing_pct:.1f}%) using method: {method}")
            
            if method == 'interpolate':
                df_complete = df_complete.interpolate(method='linear')
            elif method == 'ffill':
                df_complete = df_complete.fillna(method='ffill')
            elif method == 'bfill':
                df_complete = df_complete.fillna(method='bfill')
            else:
                # Default to forward fill then backward fill
                df_complete = df_complete.fillna(method='ffill').fillna(method='bfill')
            
            # Final fallback for any remaining NaN values
            if df_complete.isnull().any().any():
                print("WARNING: Some NaN values remain after filling. Using default values.")
                for col in df_complete.columns:
                    if 'price' in col.lower():
                        df_complete[col] = df_complete[col].fillna(0.1)  # Default price
                    else:
                        df_complete[col] = df_complete[col].fillna(0)  # Default for other columns
        
        return df_complete
        
    except Exception as e:
        print(f"Error filling missing data: {e}")
        return df


def prepare_simulation_data(
    consumption_profile_path: str,
    annual_consumption_kwh: float,
    start_date: datetime,
    end_date: datetime,
    profile_column: str = "BTN A",
    fill_missing: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare all data needed for simulation with enhanced error handling.
    """
    try:
        print("="*50)
        print("PREPARING SIMULATION DATA")
        print("="*50)
        
        # Load consumption profile
        print(f"\n1. Loading consumption profile from {consumption_profile_path}")
        consumption_df = load_consumption_profile(consumption_profile_path, profile_column)
        
        # Unnormalize to actual kWh
        print("2. Converting normalized consumption to actual kWh")
        consumption_df["kwh"] = consumption_df["permil"].apply(
            lambda x: unnormalize_consumption(annual_consumption_kwh, x)
        )
        
        # Calculate power in kW (15 min = 0.25 hours)
        consumption_df["kw"] = consumption_df["kwh"] / 0.25
        
        # Validate consumption data
        avg_daily_kwh = consumption_df["kwh"].sum() * 96 / len(consumption_df) if len(consumption_df) > 0 else 0
        print(f"   Estimated average daily consumption: {avg_daily_kwh:.1f} kWh")
        
        # Load prices from cache (add buffer days for lookahead)
        print("\n3. Loading OMIE prices from cache")
        all_prices = load_cached_prices()
        prices_df = all_prices.loc[start_date:end_date + timedelta(days=2)]

        if prices_df.empty:
            raise ValueError("No price data available in cache for requested period")
        
        # Convert to EUR/kWh
        prices_df["price_eur_per_kwh"] = prices_df["price_eur_per_mwh"] / 1000
        
        # Resample to 15 minutes
        print("4. Resampling prices to 15-minute intervals")
        prices_15min = resample_prices_to_15min(prices_df)
        
        # Align both datasets to simulation period
        print("5. Aligning data to simulation period")
        consumption_aligned = align_data_to_period(consumption_df, start_date, end_date)
        
        # Keep extra days for lookahead in prices
        prices_aligned = align_data_to_period(prices_15min, start_date, end_date + timedelta(days=2))
        
        # Fill missing data if requested
        if fill_missing:
            print("6. Filling missing data points")
            consumption_aligned = fill_missing_data(consumption_aligned, start_date, end_date, method='interpolate')
            prices_aligned = fill_missing_data(prices_aligned, start_date, end_date + timedelta(days=2), method='ffill')
        
        # Final validation
        print("\n7. Final data validation")
        cons_start, cons_end = consumption_aligned.index[0], consumption_aligned.index[-1]
        price_start, price_end = prices_aligned.index[0], prices_aligned.index[-1]
        
        print(f"   Consumption data: {len(consumption_aligned)} points from {cons_start} to {cons_end}")
        print(f"   Price data: {len(prices_aligned)} points from {price_start} to {price_end}")
        
        # Check data quality
        cons_coverage = len(consumption_aligned) / ((end_date - start_date).days + 1) / 96 * 100
        price_coverage = len(prices_aligned) / ((end_date + timedelta(days=2) - start_date).days + 1) / 96 * 100
        
        print(f"   Coverage: Consumption {cons_coverage:.1f}%, Prices {price_coverage:.1f}%")
        
        if cons_coverage < 90 or price_coverage < 90:
            warnings.warn("Low data coverage detected. Results may be affected.")
        
        print("\nData preparation completed successfully!")
        print("="*50)
        
        return consumption_aligned, prices_aligned
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise


def save_results(results_df: pd.DataFrame, filepath: str):
    """Save simulation results to CSV with enhanced error handling."""
    try:
        results_df.to_csv(filepath, float_format='%.6f')
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try alternative filename
        import time
        alt_filepath = f"results_backup_{int(time.time())}.csv"
        try:
            results_df.to_csv(alt_filepath, float_format='%.6f')
            print(f"Results saved to backup file: {alt_filepath}")
        except Exception as e2:
            print(f"Failed to save results to backup file: {e2}")


def validate_data_quality(consumption_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive data quality assessment.
    """
    quality_report = {
        'consumption': {},
        'prices': {},
        'overall_quality': 'unknown'
    }
    
    # Consumption data quality
    if not consumption_df.empty:
        cons_stats = {
            'total_points': len(consumption_df),
            'date_range': (consumption_df.index.min(), consumption_df.index.max()),
            'missing_values': consumption_df.isnull().sum().sum(),
            'negative_values': (consumption_df < 0).sum().sum(),
            'zero_values': (consumption_df == 0).sum().sum(),
            'extreme_values': (consumption_df > consumption_df.quantile(0.99) * 2).sum().sum(),
            'avg_daily_kwh': consumption_df['kwh'].sum() * 96 / len(consumption_df) if 'kwh' in consumption_df.columns else 0
        }
        quality_report['consumption'] = cons_stats
    
    # Price data quality
    if not prices_df.empty:
        price_col = 'price_eur_per_kwh' if 'price_eur_per_kwh' in prices_df.columns else prices_df.columns[0]
        price_stats = {
            'total_points': len(prices_df),
            'date_range': (prices_df.index.min(), prices_df.index.max()),
            'missing_values': prices_df.isnull().sum().sum(),
            'negative_values': (prices_df[price_col] < 0).sum(),
            'extreme_low': (prices_df[price_col] < -0.1).sum(),
            'extreme_high': (prices_df[price_col] > 1.0).sum(),
            'price_range': (prices_df[price_col].min(), prices_df[price_col].max()),
            'avg_price': prices_df[price_col].mean()
        }
        quality_report['prices'] = price_stats
    
    # Overall quality assessment
    issues = []
    if quality_report['consumption'].get('missing_values', 0) > 0:
        issues.append("consumption_missing")
    if quality_report['prices'].get('missing_values', 0) > 0:
        issues.append("prices_missing")
    if quality_report['consumption'].get('extreme_values', 0) > 0:
        issues.append("consumption_extreme")
    if quality_report['prices'].get('extreme_high', 0) > 0:
        issues.append("prices_extreme")
    
    if not issues:
        quality_report['overall_quality'] = 'good'
    elif len(issues) <= 2:
        quality_report['overall_quality'] = 'acceptable'
    else:
        quality_report['overall_quality'] = 'poor'
    
    quality_report['issues'] = issues
    
    return quality_report