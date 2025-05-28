import requests
import os
import pandas as pd
import tqdm
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import random
import sys
import logging
import json
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = "./data/votes"
SUMMARIES_DIR = os.path.join(OUTPUT_DIR, "summaries")
MEMBERS_DIR = os.path.join(OUTPUT_DIR, "members")
HOUSE_SUMM_DIR = os.path.join(SUMMARIES_DIR, "house")
SENATE_SUMM_DIR = os.path.join(SUMMARIES_DIR, "senate")
HOUSE_MEMB_DIR = os.path.join(MEMBERS_DIR, "house")
SENATE_MEMB_DIR = os.path.join(MEMBERS_DIR, "senate")
CACHE_FILE = os.path.join(OUTPUT_DIR, "house_roll_cache.json")
RATE_LIMIT_DELAY = 0.5
MAX_WORKERS = 3
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Chrome/91.0.4472.124 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/18.18363",
]
for d in (
    OUTPUT_DIR,
    SUMMARIES_DIR, MEMBERS_DIR,
    HOUSE_SUMM_DIR, SENATE_SUMM_DIR,
    HOUSE_MEMB_DIR, SENATE_MEMB_DIR,
    "logs"
):
    os.makedirs(d, exist_ok=True)
logging.basicConfig(
    filename="logs/scrape_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
http_session = requests.Session()
http_session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
def get_int(elem, tag, default=0):
    if elem is None:
        return default
    child = elem.find(tag)
    try:
        return int(child.text.strip())
    except:
        return default

def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%B %d, %Y, %I:%M %p","%B %d, %Y","%d-%b-%Y","%Y-%m-%d","%d-%b",):
        try:
            s = date_str.strip()
            if fmt == "%d-%b":
                s = f"{s}-{datetime.now().year}"
                fmt = "%d-%b-%Y"
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def normalize_vote_label(vote):
    return {
        'Aye':'Yes','Yea':'Yes','Yes':'Yes',
        'No':'No','Nay':'No',
        'Present':'Present','Not Voting':'Not Voting'
    }.get((vote or '').strip(), 'Other')

def is_blocked(response):
    if response.status_code in (403, 429):
        return True
    return 'blocked' in response.text.lower() or 'forbidden' in response.text.lower()

def get_congress_session(year):
    congress = (year - 1789) // 2 + 1
    session = 1 if year % 2 else 2
    return congress, session

def load_house_roll_cache():
    if os.path.exists(CACHE_FILE):
        return json.load(open(CACHE_FILE))
    return {}

def save_house_roll_cache(cache):
    json.dump(cache, open(CACHE_FILE,'w'))

def find_max_roll_house(year):
    cache = load_house_roll_cache()
    key = str(year)
    if key in cache:
        return cache[key]
    max_roll, roll, step_idx = 0,1,0
    steps=[500,100,1]
    while step_idx<len(steps):
        url=f"https://clerk.house.gov/evs/{year}/roll{roll:03d}.xml"
        r=http_session.get(url,headers={'User-Agent':random.choice(USER_AGENTS)})
        if is_blocked(r):sys.exit(1)
        if r.status_code==200 and '<vote-metadata>' in r.text:
            max_roll=max(max_roll,roll)
            roll+=steps[step_idx]
        else:
            if steps[step_idx]==1:break
            step_idx+=1;roll=max_roll+1
        time.sleep(RATE_LIMIT_DELAY+random.random()*0.1)
    if max_roll:
        cache[key]=max_roll;save_house_roll_cache(cache)
    return max_roll

def get_senate_vote_menu(congress,session):
    url=f"https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_{congress}_{session}.xml"
    r=http_session.get(url,headers={'User-Agent':random.choice(USER_AGENTS)})
    if is_blocked(r) or r.status_code!=200:return[]
    root=ET.fromstring(r.text);votes=[]
    for v in root.findall('.//vote'):
        try:
            num=int(v.findtext('vote_number').strip())
            votes.append({'vote_number':num,'vote_date':parse_date(v.findtext('vote_date')),'legis_num':v.findtext('issue'),'question':v.findtext('question'),'result':v.findtext('result'),'yeas':get_int(v.find('vote_tally'),'yeas'),'nays':get_int(v.find('vote_tally'),'nays'),'title':v.findtext('title')})
        except:continue
    return sorted(votes,key=lambda x:x['vote_number'])

def fetch_senate_roll(congress,session,info):
    roll=info['vote_number']
    url=f"https://www.senate.gov/legislative/LIS/roll_call_votes/vote{congress}{session}/vote_{congress}_{session}_{roll:05d}.xml"
    r=http_session.get(url,headers={'User-Agent':random.choice(USER_AGENTS)})
    if r.status_code==429:time.sleep(5);r=http_session.get(url,headers={'User-Agent':random.choice(USER_AGENTS)})
    if is_blocked(r) or r.status_code!=200 or '<roll_call_vote' not in r.text:return None
    root=ET.fromstring(r.text);c=root.find('count')
    summary={'chamber':'Senate','congress':congress,'session':session,'roll_number':roll,'chamber_roll_number':f"{roll}",'vote_date':parse_date(root.findtext('vote_date')),'title':root.findtext('vote_title') or info['title'],'question':root.findtext('vote_question_text') or info['question'],'result':root.findtext('vote_result') or info['result'],'yeas':get_int(c,'yeas',info['yeas']),'nays':get_int(c,'nays',info['nays']),'present':get_int(c,'present'),'not_voting':get_int(c,'absent')}
    votes=[]
    for m in root.findall('./members/member'):
        votes.append({'chamber':'Senate','roll_number':roll,'member_id':m.findtext('lis_member_id') or '','legislator_name':m.findtext('member_full') or '','party':m.findtext('party') or '','state':m.findtext('state') or '','vote':normalize_vote_label(m.findtext('vote_cast'))})
    time.sleep(RATE_LIMIT_DELAY+random.random()*0.1)
    return summary,votes

def fetch_house_roll(year,roll):
    url=f"https://clerk.house.gov/evs/{year}/roll{roll:03d}.xml"
    r=http_session.get(url,headers={'User-Agent':random.choice(USER_AGENTS)})
    if r.status_code==429:time.sleep(5);r=http_session.get(url,headers={'User-Agent':random.choice(USER_AGENTS)})
    if is_blocked(r) or r.status_code!=200 or '<vote-metadata>' not in r.text:return None
    root=ET.fromstring(r.text);meta=root.find('vote-metadata')
    summary={'chamber':'House','congress':meta.findtext('congress'),'session':meta.findtext('session')[0],'roll_number':roll,'chamber_roll_number':f"{roll}",'vote_date':parse_date(meta.findtext('action-date')),'title':meta.findtext('vote-desc'),'question':meta.findtext('vote-question'),'result':meta.findtext('vote-result'),'yeas':get_int(meta.find('vote-totals/totals-by-vote'),'yea-total'),'nays':get_int(meta.find('vote-totals/totals-by-vote'),'nay-total'),'present':get_int(meta.find('vote-totals/totals-by-vote'),'present-total'),'not_voting':get_int(meta.find('vote-totals/totals-by-vote'),'not-voting-total')}
    votes=[]
    for rec in root.findall('vote-data/recorded-vote'):
        member=rec.find('legislator')
        votes.append({'chamber':'House','roll_number':roll,'member_id':member.get('name-id'),'legislator_name':member.text.strip(),'party':member.get('party'),'state':member.get('state'),'vote':normalize_vote_label(rec.findtext('vote'))})
    time.sleep(RATE_LIMIT_DELAY+random.random()*0.1)
    return summary,votes

def scrape_year(year):
    congress,session=get_congress_session(year)
    menu=get_senate_vote_menu(congress,session)
    max_house=find_max_roll_house(year)
    local_sums,local_votes=[],[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures=[pool.submit(fetch_senate_roll,congress,session,info) for info in menu]
        futures+=[pool.submit(fetch_house_roll,year,roll) for roll in range(1,max_house+1)]
        for f in tqdm.tqdm(as_completed(futures),total=len(futures),desc=f"Year {year}"):
            res=f.result()
            if res: local_sums.append(res[0]);local_votes.extend(res[1])
    return local_sums,local_votes

if __name__=='__main__':
    if len(sys.argv)!=3: print("Usage: python scrape_congress_votes_parallel.py <start_year> <end_year>"); sys.exit(1)
    start_year,end_year=map(int,sys.argv[1:])
    for yr in range(start_year,end_year+1):
        paths={
            'House':{'summary':os.path.join(HOUSE_SUMM_DIR,f"{yr}.csv"),'members':os.path.join(HOUSE_MEMB_DIR,f"{yr}.csv")},
            'Senate':{'summary':os.path.join(SENATE_SUMM_DIR,f"{yr}.csv"),'members':os.path.join(SENATE_MEMB_DIR,f"{yr}.csv")}
        }
        if all(os.path.exists(p) for ch in paths.values() for p in ch.values()): print(f"[{yr}] All exist — skip"); continue
        print(f"[{yr}] Scraping…")
        sums,votes=scrape_year(yr)
        hs=[s for s in sums if s['chamber']=='House']
        ss=[s for s in sums if s['chamber']=='Senate']
        hv=[v for v in votes if v['chamber']=='House']
        sv=[v for v in votes if v['chamber']=='Senate']
        pd.DataFrame(hs).to_csv(paths['House']['summary'],index=False)
        pd.DataFrame(ss).to_csv(paths['Senate']['summary'],index=False)
        pd.DataFrame(hv).to_csv(paths['House']['members'],index=False)
        pd.DataFrame(sv).to_csv(paths['Senate']['members'],index=False)
        print(f"[{yr}] Written House summary & members and Senate summary & members")
    print("Done.")
