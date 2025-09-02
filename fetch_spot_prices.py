from __future__ import annotations

"""Utility script to fetch and cache REN spot prices since 2015.

This script fetches all available OMIE electricity prices from the REN API
starting in 2015 and stores them locally for fast subsequent access.  If the
cache file exists, it is loaded instead of querying the API again.  Use the
``--refresh`` flag to force a new download.
"""

from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd

from ess.io import fetch_ren_prices

CACHE_FILE = Path("data/spot_prices.parquet")


def fetch_and_cache(start_year: int = 2015) -> pd.DataFrame:
    """Fetch prices from the API and store them locally."""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime.utcnow()
    df = fetch_ren_prices(start_date, end_date)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    print(f"Saved {len(df)} records to {CACHE_FILE}")
    return df


def load_cached() -> pd.DataFrame:
    """Load cached price data from disk."""
    return pd.read_parquet(CACHE_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache REN spot prices")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fetch from API even if cache exists",
    )
    args = parser.parse_args()

    if CACHE_FILE.exists() and not args.refresh:
        df = load_cached()
        print(f"Loaded {len(df)} records from {CACHE_FILE}")
    else:
        df = fetch_and_cache()

    # Display a brief preview so users know it worked
    print(df.head())


if __name__ == "__main__":
    main()
