from __future__ import annotations

"""Utility script to manage cached REN spot prices.

Loads locally cached OMIE electricity prices and only fetches from the REN API
when the cache is missing or a refresh is requested.
"""

import argparse

from ess.io import load_cached_prices, PRICE_CACHE_FILE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache REN spot prices"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fetch from API even if cache exists",
    )
    args = parser.parse_args()

    existed = PRICE_CACHE_FILE.exists()
    df = load_cached_prices(refresh=args.refresh)
    if args.refresh or not existed:
        print(f"Saved {len(df)} records to {PRICE_CACHE_FILE}")
    else:
        print(f"Loaded {len(df)} records from {PRICE_CACHE_FILE}")

    # Display a brief preview so users know it worked
    print(df.head())


if __name__ == "__main__":
    main()
