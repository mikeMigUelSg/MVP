#!/usr/bin/env python3
"""
scripts/fetch_spot_prices.py - Fetch and cache REN spot prices with smart incremental updates
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# CORREÇÃO DO PATH
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from ess.io import fetch_ren_prices

PRICE_CACHE_FILE = Path("data/spot_prices.parquet")

def analyze_cache_gaps(df: pd.DataFrame) -> dict:
    """Analyze existing cache to find missing dates."""
    if df.empty:
        return {
            'has_data': False,
            'earliest_date': None,
            'latest_date': None,
            'missing_start': None,
            'missing_end': None,
            'total_records': 0,
            'missing_days': None
        }
    
    earliest_date = df.index.min()
    latest_date = df.index.max()
    
    # Check if we need to fetch newer data (up to yesterday)
    yesterday = (datetime.now() - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    
    missing_start = None
    missing_end = None
    missing_days = 0
    
    if latest_date < yesterday:
        missing_start = latest_date + timedelta(hours=1)
        missing_end = yesterday
        missing_days = (missing_end.date() - missing_start.date()).days + 1
    
    return {
        'has_data': True,
        'earliest_date': earliest_date,
        'latest_date': latest_date,
        'missing_start': missing_start,
        'missing_end': missing_end,
        'total_records': len(df),
        'missing_days': missing_days
    }

def fetch_missing_data(existing_df: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    """Fetch only the missing data and merge with existing."""
    if not analysis['missing_start']:
        print("✅ Cache is up to date, no missing data")
        return existing_df
    
    print(f"🔄 Fetching missing data from {analysis['missing_start'].date()} to {analysis['missing_end'].date()}")
    print(f"   Missing {analysis['missing_days']} days of data")
    
    # Fetch missing data
    new_df = fetch_ren_prices(analysis['missing_start'], analysis['missing_end'])
    
    if new_df.empty:
        print("⚠️  No new data fetched")
        return existing_df
    
    print(f"📥 Fetched {len(new_df)} new records")
    
    # Merge with existing data
    if existing_df.empty:
        combined_df = new_df
    else:
        # Concatenate and remove duplicates
        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    
    print(f"📊 Combined dataset: {len(combined_df)} total records")
    return combined_df

def fetch_complete_history(start_year: int = 2015) -> pd.DataFrame:
    """Fetch complete history from start_year to yesterday."""
    start_date = datetime(start_year, 1, 1)
    end_date = (datetime.now() - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    
    print(f"🔄 Fetching complete history from {start_date.date()} to {end_date.date()}")
    return fetch_ren_prices(start_date, end_date)

def load_cached() -> pd.DataFrame:
    """Load cached price data from disk."""
    if PRICE_CACHE_FILE.exists():
        df = pd.read_parquet(PRICE_CACHE_FILE)
        print(f"📖 Loaded {len(df)} records from cache")
        return df
    else:
        print(f"📁 Cache file {PRICE_CACHE_FILE} not found")
        return pd.DataFrame()

def save_cache(df: pd.DataFrame) -> None:
    """Save DataFrame to cache."""
    PRICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICE_CACHE_FILE)
    print(f"💾 Saved {len(df)} records to {PRICE_CACHE_FILE}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache REN spot prices with smart updates")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force complete refresh from 2015",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for complete refresh (default: 2015)",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("REN SPOT PRICES - SMART INCREMENTAL FETCH")
    print("="*60)
    print(f"Cache file: {PRICE_CACHE_FILE}")
    
    if args.refresh:
        print("🔄 FULL REFRESH requested")
        df = fetch_complete_history(args.start_year)
        if not df.empty:
            save_cache(df)
    else:
        # Smart incremental update
        print("🧠 SMART UPDATE mode")
        existing_df = load_cached()
        analysis = analyze_cache_gaps(existing_df)
        
        if not analysis['has_data']:
            print("📂 No existing data, fetching complete history...")
            df = fetch_complete_history(args.start_year)
            if not df.empty:
                save_cache(df)
        else:
            print(f"📊 Cache analysis:")
            print(f"   Existing data: {analysis['earliest_date'].date()} to {analysis['latest_date'].date()}")
            print(f"   Total records: {analysis['total_records']:,}")
            
            if analysis['missing_days'] and analysis['missing_days'] > 0:
                print(f"   Missing: {analysis['missing_days']} days")
                
                # Fetch and merge missing data
                df = fetch_missing_data(existing_df, analysis)
                if len(df) > len(existing_df):
                    save_cache(df)
                    print(f"✅ Added {len(df) - len(existing_df)} new records")
                else:
                    df = existing_df
            else:
                df = existing_df
                print("✅ Cache is up to date!")
    
    if df.empty:
        print("❌ No data available")
        return
    
    # Final summary
    print("\n" + "="*60)
    print("📈 FINAL DATASET SUMMARY")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Days covered: {(df.index[-1] - df.index[0]).days + 1}")
    
    # Price statistics
    prices = df.iloc[:, 0]  # First column is price
    print(f"Price range: {prices.min():.1f} - {prices.max():.1f} EUR/MWh")
    print(f"Average price: {prices.mean():.1f} EUR/MWh")
    
    # Recent data check
    days_old = (datetime.now().date() - df.index[-1].date()).days
    if days_old == 0:
        print("✅ Data is current (today)")
    elif days_old == 1:
        print("✅ Data is recent (yesterday)")
    elif days_old <= 7:
        print(f"⚠️  Data is {days_old} days old")
    else:
        print(f"⚠️  Data is {days_old} days old - consider running --refresh")
    
    # Preview
    print(f"\n📊 Recent data preview:")
    print(df.tail())
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()