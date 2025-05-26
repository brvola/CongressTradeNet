import sys
import os
from datetime import datetime

import requests
import pandas as pd

HOUSE_URL  = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

def fetch_json(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def main():
    house = fetch_json(HOUSE_URL)
    senate = fetch_json(SENATE_URL)

    df_h = pd.DataFrame(house)
    df_s = pd.DataFrame(senate)

    df = pd.concat([df_h, df_s], ignore_index=True, sort=False)

    df["is_senator"] = df["senator"].notna()
    df["chamber"]    = df["is_senator"].map({True: "Senate", False: "House"})
    df["member"]     = df["senator"].fillna(df["representative"])

    df = df.drop(columns=["representative", "senator"], errors="ignore")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    out_dir = "./data/trades"
    os.makedirs(out_dir, exist_ok=True) 
    ts = datetime.now().strftime("%Y%m%d")
    fname = f"trades_2021-2023_{ts}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)

    print(f"Saved {len(df)} records ({df['is_senator'].sum()} senators, "
          f"{len(df)-df['is_senator'].sum()} representatives) → {path}")

if __name__ == "__main__":
    main()
