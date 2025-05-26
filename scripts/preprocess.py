import pandas as pd
import os
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/clean_votes_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SUFFIX = "2021-2023_20250526"
INPUT_FILE = f"./data/votes/votes_by_member_{SUFFIX}.csv" 
OUTPUT_DIR = "./data/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"votes_cleaned_{datetime.now().strftime('%Y%m%d')}.csv")

def clean_votes():
    try:
        logging.info("Loading votes data")
        votes = pd.read_csv(INPUT_FILE, names=[
            "chamber", "roll_number", "member_id", "legislator_name",
            "party", "state", "vote", "name_id"
        ])

        logging.info(f"Initial votes shape: {votes.shape}")

        logging.info("Cleaning legislator_name")
        def clean_name(name):
            if pd.isna(name):
                return "Unknown"
            if "(" in name and ")" in name:
                name_part = name.split("(")[0].strip()
                return name_part
            return name.strip()
        votes["legislator_name"] = votes["legislator_name"].apply(clean_name)

        logging.info("Converting votes to binary")
        votes["vote"] = votes["vote"].map({
            "Yes": 1, "No": 0, "Aye": 1, "Yea": 1, "Nay": 0,
            "Present": None, "Not Voting": None
        })

        votes["roll_number"] = votes["roll_number"].astype(str)

        logging.info("Cleaning other fields")
        votes["chamber"] = votes["chamber"].str.strip().fillna("Unknown")
        votes["member_id"] = votes["member_id"].str.strip().fillna("Unknown")
        votes["name_id"] = votes["name_id"].str.strip().fillna("Unknown")
        votes["party"] = votes["party"].str.strip().fillna("Unknown")
        votes["state"] = votes["state"].str.upper().str.strip().fillna("Unknown")

        valid_chambers = ["Senate", "House"]
        votes = votes[votes["chamber"].isin(valid_chambers)]

        logging.info("Handling missing values")
        votes = votes.dropna(subset=["chamber", "roll_number", "member_id"])

        logging.info("Removing duplicates")
        votes = votes.drop_duplicates(subset=["chamber", "roll_number", "member_id"])

        logging.info(f"Final votes shape: {votes.shape}")

        votes.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Cleaned votes saved to {OUTPUT_FILE}")
        print(f"Cleaned votes saved to {OUTPUT_FILE}")

    except Exception as e:
        logging.error(f"Error cleaning votes: {e}")
        raise

if __name__ == "__main__":
    clean_votes()