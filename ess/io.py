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


def load_real_consumption(filepath: str) -> pd.DataFrame:
    """
    Load real consumption data using "START of period" timestamp convention.
    """
    try:
        print(f"Loading real consumption data from {filepath}")
        print("🔧 Using 'START OF PERIOD' timestamp convention")
        
        # Read Excel file
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
        except:
            df = pd.read_excel(filepath, engine='xlrd')
        
        if df.empty:
            raise ValueError("Real consumption file is empty")
        
        print(f"Loaded {len(df)} rows from real consumption file")
        
        # Use first 3 columns
        if len(df.columns) < 3:
            raise ValueError("File must have at least 3 columns")
        
        date_col = df.columns[0]
        time_col = df.columns[1]
        consumption_col = df.columns[2]
        
        print(f"Using columns: [{date_col}] [{time_col}] [{consumption_col}]")
        
        # Create datetime
        df['datetime'] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[time_col].astype(str),
            format='%Y/%m/%d %H:%M',
            errors='coerce'
        )
        
        # Clean data
        df[consumption_col] = pd.to_numeric(df[consumption_col], errors='coerce')
        valid_mask = df['datetime'].notna() & df[consumption_col].notna() & (df[consumption_col] >= 0)
        df_clean = df.loc[valid_mask].copy()
        
        if df_clean.empty:
            raise ValueError("No valid data after cleaning")
        
        print(f"After cleaning: {len(df_clean)} valid records")
        
        # *** LINHA CRÍTICA: Aplicar shift -15 minutos ***
        print("🔧 APPLYING -15min shift to convert to 'START OF PERIOD' convention")
        df_clean['datetime'] = df_clean['datetime'] - pd.Timedelta('15min')
        
        # Create result DataFrame
        result = pd.DataFrame({
            'kw': df_clean[consumption_col].values,
            'kwh': df_clean[consumption_col].values * 0.25
        }, index=df_clean['datetime'])
        
        result.sort_index(inplace=True)
        result = result[~result.index.duplicated(keep='first')]
        
        # Validation
        if result.index[0].time() == pd.Timestamp('00:00:00').time():
            print("✅ SUCCESS: Data now starts at 00:00 - START OF PERIOD convention applied!")
        
        print(f"✅ Loaded: {len(result)} points from {result.index[0]} to {result.index[-1]}")
        
        return result
        
    except Exception as e:
        print(f"Error loading real consumption data: {e}")
        raise






# Update the prepare_simulation_data function to use simplified version
def prepare_simulation_data(
    consumption_profile_path: str,
    annual_consumption_kwh: float,
    start_date: datetime,
    end_date: datetime,
    profile_column: str = "BTN A",
    fill_missing: bool = True,
    consumption_model: bool = False,
    real_consumption: bool = False,
    real_consumption_file: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare all data needed for simulation with simplified real consumption handling.
    """
    try:
        print("="*50)
        print("PREPARING SIMULATION DATA")
        print("="*50)
        
        # ====== CONSUMPTION DATA ======
        if real_consumption:
            if not real_consumption_file:
                raise ValueError("real_consumption_file must be provided when real_consumption=True")
            
            print(f"\n1. Loading REAL consumption data from {real_consumption_file}")
            consumption_df = load_real_consumption(real_consumption_file)  # Uses simplified version
            
            print("   ✅ Real consumption data loaded - using validated timestamp convention")
            
        else:
            # Original method - load E-REDES profile
            print(f"\n1. Loading consumption profile from {consumption_profile_path}")
            consumption_df = load_consumption_profile(consumption_profile_path, profile_column)
            
            print("2. Converting normalized consumption to actual kWh")
            consumption_df["kwh"] = consumption_df["permil"].apply(
                lambda x: unnormalize_consumption(annual_consumption_kwh, x)
            )
            
            # Calculate power in kW (15 min = 0.25 hours)
            consumption_df["kw"] = consumption_df["kwh"] / 0.25
        
        # Validate consumption data
        avg_daily_kwh = consumption_df["kwh"].sum() * 96 / len(consumption_df) if len(consumption_df) > 0 else 0
        print(f"   Estimated average daily consumption: {avg_daily_kwh:.1f} kWh")
        
        # ====== PRICE DATA ======
        step_num = 3 if real_consumption else 3
        print(f"\n{step_num}. Loading OMIE prices from cache")
        all_prices = load_cached_prices()
        prices_df = all_prices.loc[start_date:end_date + timedelta(days=2)]

        if prices_df.empty:
            raise ValueError("No price data available in cache for requested period")
        
        # Convert to EUR/kWh
        prices_df["price_eur_per_kwh"] = prices_df["price_eur_per_mwh"] / 1000
        
        # Resample to 15 minutes
        step_num += 1
        print(f"{step_num}. Resampling prices to 15-minute intervals")
        prices_15min = resample_prices_to_15min(prices_df)
        
        # ====== ALIGN DATA ======
        step_num += 1
        print(f"{step_num}. Aligning data to simulation period")

        if real_consumption:
            # For real consumption, just filter to the requested period
            consumption_aligned = align_data_to_period(consumption_df, start_date, end_date)
        else:
            # Original logic for E-REDES profile
            if consumption_model:
                consumption_aligned = align_consumption_by_calendar(consumption_df, start_date, end_date)
            else:
                consumption_aligned = align_data_to_period(consumption_df, start_date, end_date)
        
        # Keep extra days for lookahead in prices
        prices_aligned = align_data_to_period(prices_15min, start_date, end_date + timedelta(days=2))
        
        # ====== FILL MISSING DATA ======
        if fill_missing:
            step_num += 1
            print(f"{step_num}. Filling missing data points")
            consumption_aligned = fill_missing_data(consumption_aligned, start_date, end_date, method='interpolate')
            prices_aligned = fill_missing_data(prices_aligned, start_date, end_date + timedelta(days=2), method='ffill')
        
        # ====== FINAL VALIDATION ======
        step_num += 1
        print(f"\n{step_num}. Final data validation")
        cons_start, cons_end = consumption_aligned.index[0], consumption_aligned.index[-1]
        price_start, price_end = prices_aligned.index[0], prices_aligned.index[-1]
        
        consumption_type = "REAL" if real_consumption else "PROFILE"
        print(f"   {consumption_type} consumption data: {len(consumption_aligned)} points from {cons_start} to {cons_end}")
        print(f"   Price data: {len(prices_aligned)} points from {price_start} to {price_end}")
        
        # Check alignment for cost calculation
        common_periods = len(consumption_aligned.index.intersection(prices_aligned.index))
        alignment_pct = common_periods / len(consumption_aligned) * 100
        print(f"   ✅ Timestamp alignment: {alignment_pct:.1f}% - price calculations will be accurate")
        
        if alignment_pct < 95:
            warnings.warn(f"Only {alignment_pct:.1f}% of consumption periods have matching prices")
        
        print(f"\n✅ Data preparation completed successfully using {consumption_type} consumption data!")
        print(f"✅ Confirmed: Both datasets use 'end of period' timestamp convention")
        print("="*50)
        
        return consumption_aligned, prices_aligned
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
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
    
def align_consumption_by_calendar(cons_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Alinha o consumo para [start_date, end_date] casando por MM-DD HH:MM (ignora o ano)."""
    # índice alvo no ano pedido
    target_idx = pd.date_range(start_date.replace(second=0, microsecond=0),
                               end_date.replace(hour=23, minute=45, second=0, microsecond=0),
                               freq="15min")
    target = pd.DataFrame(index=target_idx)
    target["key"] = target.index.strftime("%m-%d %H:%M")

    # chave na origem (qualquer ano)
    src = cons_df.copy()
    src = src.sort_index()
    src["key"] = src.index.strftime("%m-%d %H:%M")

    # mantém só colunas de consumo que existirem
    cols = [c for c in ["permil", "kwh", "kw"] if c in src.columns]
    aligned = (
        target.reset_index()
              .merge(src.reset_index()[["key"] + cols], on="key", how="left")
              .set_index("index")[cols]
              .sort_index()
    )
    aligned.index.name = "datetime"
    return aligned



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
    fill_missing: bool = True,
    consumption_model: bool = False,
    real_consumption: bool = False,  # NOVA OPÇÃO
    real_consumption_file: str = None  # NOVA OPÇÃO
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare all data needed for simulation with support for real consumption data.
    """
    try:
        print("="*50)
        print("PREPARING SIMULATION DATA")
        print("="*50)
        
        # ====== CONSUMPTION DATA ======
        if real_consumption:
            if not real_consumption_file:
                raise ValueError("real_consumption_file must be provided when real_consumption=True")
            
            print(f"\n1. Loading REAL consumption data from {real_consumption_file}")
            consumption_df = load_real_consumption(real_consumption_file)
            
            # No need for unnormalization - data is already in kW and kWh
            print("   Real consumption data loaded - no normalization needed")
            
        else:
            # Original method - load E-REDES profile
            print(f"\n1. Loading consumption profile from {consumption_profile_path}")
            consumption_df = load_consumption_profile(consumption_profile_path, profile_column)
            
            print("2. Converting normalized consumption to actual kWh")
            consumption_df["kwh"] = consumption_df["permil"].apply(
                lambda x: unnormalize_consumption(annual_consumption_kwh, x)
            )
            
            # Calculate power in kW (15 min = 0.25 hours)
            consumption_df["kw"] = consumption_df["kwh"] / 0.25
        
        # Validate consumption data
        avg_daily_kwh = consumption_df["kwh"].sum() * 96 / len(consumption_df) if len(consumption_df) > 0 else 0
        print(f"   Estimated average daily consumption: {avg_daily_kwh:.1f} kWh")
        
        # ====== PRICE DATA ======
        step_num = 3 if real_consumption else 3
        print(f"\n{step_num}. Loading OMIE prices from cache")
        all_prices = load_cached_prices()
        prices_df = all_prices.loc[start_date:end_date + timedelta(days=2)]

        if prices_df.empty:
            raise ValueError("No price data available in cache for requested period")
        
        # Convert to EUR/kWh
        prices_df["price_eur_per_kwh"] = prices_df["price_eur_per_mwh"] / 1000
        
        # Resample to 15 minutes
        step_num += 1
        print(f"{step_num}. Resampling prices to 15-minute intervals")
        prices_15min = resample_prices_to_15min(prices_df)
        
        # ====== ALIGN DATA ======
        step_num += 1
        print(f"{step_num}. Aligning data to simulation period")

        if real_consumption:
            # For real consumption, just filter to the requested period
            consumption_aligned = align_data_to_period(consumption_df, start_date, end_date)
        else:
            # Original logic for E-REDES profile
            if consumption_model:
                consumption_aligned = align_consumption_by_calendar(consumption_df, start_date, end_date)
            else:
                consumption_aligned = align_data_to_period(consumption_df, start_date, end_date)
        
        # Keep extra days for lookahead in prices
        prices_aligned = align_data_to_period(prices_15min, start_date, end_date + timedelta(days=2))
        
        # ====== FILL MISSING DATA ======
        if fill_missing:
            step_num += 1
            print(f"{step_num}. Filling missing data points")
            consumption_aligned = fill_missing_data(consumption_aligned, start_date, end_date, method='interpolate')
            prices_aligned = fill_missing_data(prices_aligned, start_date, end_date + timedelta(days=2), method='ffill')
        
        # ====== FINAL VALIDATION ======
        step_num += 1
        print(f"\n{step_num}. Final data validation")
        cons_start, cons_end = consumption_aligned.index[0], consumption_aligned.index[-1]
        price_start, price_end = prices_aligned.index[0], prices_aligned.index[-1]
        
        consumption_type = "REAL" if real_consumption else "PROFILE"
        print(f"   {consumption_type} consumption data: {len(consumption_aligned)} points from {cons_start} to {cons_end}")
        print(f"   Price data: {len(prices_aligned)} points from {price_start} to {price_end}")
        
        # Check data quality
        cons_coverage = len(consumption_aligned) / ((end_date - start_date).days + 1) / 96 * 100
        price_coverage = len(prices_aligned) / ((end_date + timedelta(days=2) - start_date).days + 1) / 96 * 100
        
        print(f"   Coverage: Consumption {cons_coverage:.1f}%, Prices {price_coverage:.1f}%")
        
        if cons_coverage < 90 or price_coverage < 90:
            warnings.warn("Low data coverage detected. Results may be affected.")
        
        print(f"\nData preparation completed successfully using {consumption_type} consumption data!")
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