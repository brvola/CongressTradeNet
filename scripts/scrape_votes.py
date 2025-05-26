import requests
import os
import pandas as pd
import re
import tqdm
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import random
import sys
import logging
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scrape_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = "./data/votes"
CACHE_FILE = os.path.join(OUTPUT_DIR, "house_roll_cache.json")
RATE_LIMIT_DELAY = 0.5
MAX_WORKERS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
]

http_session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
http_session.mount("https://", HTTPAdapter(max_retries=retries))

all_votes = []
vote_summaries = []

def get_int(elem, tag, default=0):
    if elem is None:
        return default
    child = elem.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return int(child.text.strip())
    except ValueError:
        return default


def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%B %d, %Y", "%d-%b-%Y", "%Y-%m-%d", "%d-%b"):
        try:
            s = date_str.strip()
            if fmt == "%d-%b":
                s = f"{s}-{datetime.now().year}"
                fmt = "%d-%b-%Y"
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def normalize_vote_label(vote):
    return {
        'Aye':'Yes','Yea':'Yes','Yes':'Yes',
        'No':'No','Nay':'No',
        'Present':'Present','Not Voting':'Not Voting'
    }.get(vote.strip(), 'Other')


def is_blocked(response):
    if response.status_code in (403, 429):
        return True
    txt = response.text.lower()
    return 'blocked' in txt or 'forbidden' in txt


def get_congress_session(year):
    congress = (year - 1789) // 2 + 1
    session = 1 if year % 2 else 2
    return congress, session


def load_house_roll_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_house_roll_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def find_max_roll_house(year):
    cache = load_house_roll_cache()
    key = str(year)
    if key in cache:
        logging.info(f"Using cached max House roll for {year}: {cache[key]}")
        return cache[key]

    max_roll, roll, step_idx = 0, 1, 0
    steps = [500, 100, 1]
    while step_idx < len(steps):
        url = f"https://clerk.house.gov/evs/{year}/roll{roll:03d}.xml"
        r = http_session.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
        if is_blocked(r):
            logging.error("Blocked while checking max House roll")
            sys.exit(1)
        if r.status_code == 200 and '<vote-metadata>' in r.text:
            max_roll = max(max_roll, roll)
            roll += steps[step_idx]
        else:
            if steps[step_idx] == 1:
                break
            step_idx += 1
            roll = max_roll + 1
        time.sleep(RATE_LIMIT_DELAY + random.random() * 0.1)

    if max_roll:
        cache[key] = max_roll
        save_house_roll_cache(cache)
        logging.info(f"Determined max House roll for {year}: {max_roll}")
    else:
        logging.warning(f"No House rolls found for {year}")
    return max_roll


def get_senate_vote_menu(congress, session):
    url = f"https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_{congress}_{session}.xml"
    r = http_session.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
    if is_blocked(r): sys.exit(1)
    if r.status_code != 200 or '<vote_summary' not in r.text:
        return []
    root = ET.fromstring(r.text)
    votes = []
    for v in root.findall('.//vote'):
        try:
            num = int(v.findtext('vote_number').strip())
            votes.append({
                'vote_number': num,
                'vote_date': parse_date(v.findtext('vote_date')),
                'legis_num': v.findtext('issue'),
                'question': v.findtext('question'),
                'result': v.findtext('result'),
                'yeas': get_int(v.find('vote_tally'),'yeas'),
                'nays': get_int(v.find('vote_tally'),'nays'),
                'title': v.findtext('title')
            })
        except:
            continue
    return sorted(votes, key=lambda x: x['vote_number'])

def fetch_senate_roll(year, congress, session, info):
    roll = info['vote_number']
    url = f"https://www.senate.gov/legislative/LIS/roll_call_votes/vote{congress}{session}/vote_{congress}_{session}_{roll:05d}.xml"
    r = http_session.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
    if r.status_code == 429:
        time.sleep(5)
        r = http_session.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
    if is_blocked(r) or r.status_code != 200 or '<roll_call_vote' not in r.text:
        return None
    root = ET.fromstring(r.text)
    c = root.find('count')
    summary = {
        'chamber':'Senate','congress':congress,'session':session,
        'roll_number':roll,'chamber_roll_number':f"Senate_{roll}",
        'vote_date':parse_date(root.findtext('vote_date')),
        'title':root.findtext('vote_title') or info['title'],
        'question':root.findtext('vote_question_text') or info['question'],
        'result':root.findtext('vote_result') or info['result'],
        'yeas':get_int(c,'yeas',info['yeas']),'nays':get_int(c,'nays',info['nays']),
        'present':get_int(c,'present'),'not_voting':get_int(c,'absent')
    }
    votes = []
    for m in root.findall('./members/member'):
        votes.append({
            'chamber':'Senate','roll_number':roll,
            'member_id':m.findtext('lis_member_id') or '',
            'legislator_name':m.findtext('member_full') or '',
            'party':m.findtext('party') or '',
            'state':m.findtext('state') or '',
            'vote':normalize_vote_label(m.findtext('vote_cast') or '')
        })
    time.sleep(RATE_LIMIT_DELAY + random.random()*0.1)
    return summary, votes


def fetch_house_roll(year, roll):
    filename = f"roll{roll:03d}.xml"
    url = f"https://clerk.house.gov/evs/{year}/{filename}"
    r = http_session.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
    if r.status_code == 429:
        time.sleep(5)
        r = http_session.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
    if is_blocked(r) or r.status_code != 200 or '<vote-metadata>' not in r.text:
        return None
    xml = r.text
    block = re.search(r'<vote-metadata>(.*?)</vote-metadata>', xml, re.DOTALL)
    if not block:
        return None
    block = block.group(1)
    get_tag = lambda t: re.search(rf'<{t}>(.*?)</{t}>', block, re.DOTALL)
    summary = {
        'chamber':'House','congress':get_tag('congress').group(1),
        'session':get_tag('session').group(1),'roll_number':roll,
        'chamber_roll_number':f"House_{roll}",
        'vote_date':parse_date(get_tag('action-date').group(1)),
        'title':get_tag('vote-desc').group(1),
        'question':get_tag('vote-question').group(1),
        'result':get_tag('vote-result').group(1)
    }
    votes = []
    for rec in re.finditer(r'<recorded-vote>(.*?)</recorded-vote>', xml, re.DOTALL):
        txt = rec.group(1)
        lm = re.search(r'name-id="([^"]+)".*?party="([^"]+)".*?state="([^"]+)".*?>([^<]+)</legislator>', txt)
        vm = re.search(r'<vote>(.*?)</vote>', txt)
        if lm and vm:
            votes.append({
                'chamber':'House','roll_number':roll,
                'name_id':lm.group(1),'legislator_name':lm.group(4),
                'party':lm.group(2),'state':lm.group(3),
                'vote':normalize_vote_label(vm.group(1))
            })
    time.sleep(RATE_LIMIT_DELAY + random.random()*0.1)
    return summary, votes

def scrape_year(year):
    congress, session = get_congress_session(year)
    logging.info(f"Scraping Senate {congress}/{session}")
    menu = get_senate_vote_menu(congress, session)
    max_house = find_max_roll_house(year)
    logging.info(f"Max House roll for {year}: {max_house}")

    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for info in menu:
            futures[pool.submit(fetch_senate_roll, year, congress, session, info)] = None
        if max_house > 0:
            for r in range(1, max_house+1):
                futures[pool.submit(fetch_house_roll, year, r)] = None
        else:
            logging.warning(f"Skipping House scraping for {year} because max_roll=0")

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Year {year}"):
            res = future.result()
            if res:
                summary, votes = res
                vote_summaries.append(summary)
                all_votes.extend(votes)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scrape_congress_votes_parallel.py <start_year> <end_year>")
        sys.exit(1)
    start_year, end_year = map(int, sys.argv[1:])
    for yr in range(start_year, end_year+1):
        scrape_year(yr)

    ts = datetime.now().strftime("%Y%m%d")
    pd.DataFrame(vote_summaries).to_csv(os.path.join(OUTPUT_DIR, f"votes_summary_{start_year}-{end_year}_{ts}.csv"), index=False)
    pd.DataFrame(all_votes).to_csv(os.path.join(OUTPUT_DIR, f"votes_by_member_{start_year}-{end_year}_{ts}.csv"), index=False)
    logging.info("All data saved.")
    print(f"Done. Data saved to {OUTPUT_DIR}")
