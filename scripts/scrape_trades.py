import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HOUSE_URL  = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/scrape_trades_json.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_session():
    """Return a requests.Session with retry logic."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def fetch_json(url, session):
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def main():
    p = argparse.ArgumentParser(
        description="Download House+Senate trade disclosures and save to CSV"
    )
    p.add_argument("start_year", type=int, help="Earliest transaction year (e.g. 2020)")
    p.add_argument("end_year",   type=int, help="Latest   transaction year (e.g. 2024)")
    args = p.parse_args()

    now_year = datetime.now().year
    if not (2000 <= args.start_year <= args.end_year <= now_year):
        p.error(f"Years must be between 2000 and {now_year}, and start_year ≤ end_year.")

    OUTPUT_DIR = "./data/trades"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = get_session()

    logging.info(f"Fetching House transactions JSON")
    house_data = fetch_json(HOUSE_URL, session)
    logging.info(f"Fetched {len(house_data)} House records")

    logging.info(f"Fetching Senate transactions JSON")
    senate_data = fetch_json(SENATE_URL, session)
    logging.info(f"Fetched {len(senate_data)} Senate records")

    df_house  = pd.DataFrame(house_data)
    df_senate = pd.DataFrame(senate_data)

    logging.info("Concatenating House + Senate")
    df = pd.concat([df_house, df_senate], ignore_index=True)

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    mask = (
        (df["transaction_date"].dt.year >= args.start_year) &
        (df["transaction_date"].dt.year <= args.end_year)
    )
    df = df.loc[mask]

    timestamp = datetime.now().strftime("%Y%m%d")
    out_file = os.path.join(
        OUTPUT_DIR,
        f"trades_{args.start_year}-{args.end_year}_{timestamp}.csv"
    )
    df.to_csv(out_file, index=False)
    logging.info(f"Saved {len(df)} filtered records to {out_file}")
    print(f"Done. Saved {len(df)} trades to {out_file}")

if __name__ == "__main__":
    main()
