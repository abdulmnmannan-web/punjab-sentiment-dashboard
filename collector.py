import pandas as pd
import time
import requests
import feedparser
import json
import os
import re
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from groq import Groq

# — Load API key —
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        with open(os.path.expanduser("~/.groq_key")) as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise Exception("GROQ_API_KEY not found in environment or ~/.groq_key file")

client = Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"
CUTOFF_DATE = datetime.now(timezone.utc) - timedelta(days=7)
PARTIES = ["AAP", "INC", "BJP", "SAD"]

SYSTEM_PROMPT = """You are a political news analyst for Punjab, India. Given a news headline, determine:
1. Which party (AAP, INC, BJP, SAD) the headline is actually ABOUT or DIRECTED AT -- i.e. whose reputation this headline affects. If a leader from one party criticizes another party, the article is about the party being criticized, not the speaker's party. If no specific Punjab political party is the clear subject, respond with "None".
2. The sentiment toward that party: "favoring" (positive/complimentary), "against" (critical/negative), or "neutral" (factual, no clear sentiment, or sarcasm/mockery should be scored as "against" the party being mocked).
3. A sentiment score from -1.0 (very negative) to 1.0 (very positive), 0.0 for neutral.
4. A one-sentence reasoning.
Respond ONLY with valid JSON in this exact format, no other text:
{"party": "INC", "sentiment_label": "against", "sentiment_score": -0.6, "reasoning": "explanation here"}
"""

def analyze_with_ai(headline, searched_party, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Headline: {headline}\n\n(This was found while searching for news about: {searched_party})"}
                ],
                temperature=0.1,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r'^```json\s*|\s*```$', '', text.strip())
            parsed = json.loads(text)
            party = parsed.get("party", "None")
            if party not in PARTIES:
                party = None
            return (
                party,
                float(parsed.get("sentiment_score", 0.0)),
                parsed.get("sentiment_label", "neutral"),
                0.9,
                parsed.get("reasoning", "No reasoning provided")
            )
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return None, 0.0, "error", 0.0, f"AI error after retries: {e}"
    return None, 0.0, "error", 0.0, "Failed after retries"

KEYWORDS_FILE = "keywords.json"

def load_keywords():
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def is_recent(dt):
    if dt is None:
        return True
    return dt >= CUTOFF_DATE

def parse_gdelt_date(seendate_str):
    try:
        return datetime.strptime(seendate_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def parse_rss_date(published_str):
    try:
        dt = parsedate_to_datetime(published_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def fetch_gdelt(query, max_records=100, retries=2):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    start = CUTOFF_DATE.strftime("%Y%m%d%H%M%S")
    end = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    params = {
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": max_records, "sort": "datedesc",
        "startdatetime": start, "enddatetime": end
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            return resp.json().get("articles", [])
        except Exception:
            if attempt < retries:
                time.sleep(15)
                continue
            return []
    return []

def process_gdelt(articles, searched_party):
    rows = []
    max_articles = 4  # reduced for reliability
    for a in articles:
        if len(rows) >= max_articles:
            break
        title = a.get("title", "")
        dt = parse_gdelt_date(a.get("seendate", ""))
        if not is_recent(dt):
            continue
        party, score, label, confidence, reasoning = analyze_with_ai(title, searched_party)
        if party is None:
            continue
        rows.append({
            "searched_party": searched_party, "subject_party": party,
            "confidence": confidence, "analysis_reasoning": reasoning, "party": party,
            "source_type": "News Archive (GDELT)", "source": a.get("domain", "unknown"),
            "title_original": title, "title_english": title,
            "publishedAt": a.get("seendate"), "url": a.get("url"),
            "compound": score, "sentiment_label": label,
            "fetched_at": datetime.now().isoformat()
        })
        print(f"  {searched_party} GDELT: {len(rows)}/{max_articles} analyzed")
        time.sleep(1)
    return rows

def fetch_rss(query, lang="en-IN", country="IN"):
    lang_code = lang.split("-")[0]
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl={lang}&gl={country}&ceid={country}:{lang_code}"
    return feedparser.parse(url).entries

def process_rss(entries, searched_party, needs_translation=False):
    rows = []
    max_articles = 4  # reduced for reliability
    for e in entries:
        if len(rows) >= max_articles:
            break
        dt = parse_rss_date(e.get("published", ""))
        if not is_recent(dt):
            continue
        orig = e.get("title", "")
        party, score, label, confidence, reasoning = analyze_with_ai(orig, searched_party)
        if party is None:
            continue
        rows.append({
            "searched_party": searched_party, "subject_party": party,
            "confidence": confidence, "analysis_reasoning": reasoning, "party": party,
            "source_type": "Google News (Punjabi)" if needs_translation else "Google News (English)",
            "source": e.get("source", {}).get("title", "Google News"),
            "title_original": orig, "title_english": orig,
            "publishedAt": e.get("published"), "url": e.get("link"),
            "compound": score, "sentiment_label": label,
            "fetched_at": datetime.now().isoformat()
        })
        print(f"  {searched_party} RSS: {len(rows)}/{max_articles} analyzed")
        time.sleep(1)
    return rows

def run_collection():
    keywords = load_keywords()
    all_rows = []
    for party, terms in keywords.items():
        print(f"[{datetime.now()}] GDELT: {party}...")
        for q in terms.get("english", []):
            all_rows += process_gdelt(fetch_gdelt(q), party)
            time.sleep(6)
        print(f"[{datetime.now()}] RSS English: {party}...")
        for q in terms.get("english", []):
            all_rows += process_rss(fetch_rss(q), party)
        print(f"[{datetime.now()}] RSS Punjabi: {party}...")
        for q in terms.get("punjabi", []):
            all_rows += process_rss(fetch_rss(q, lang="pa-IN", country="IN"), party, needs_translation=True)
    df = pd.DataFrame(all_rows)
    output_file = "punjab_sentiment_live.csv"
    if os.path.exists(output_file):
        try:
            old_df = pd.read_csv(output_file)
            df = pd.concat([old_df, df], ignore_index=True)
            df.drop_duplicates(subset="url", keep="last", inplace=True)
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        except Exception:
            pass
    df.to_csv(output_file, index=False)
    print(f"[{datetime.now()}] Saved {len(df)} total articles (AI-analyzed, last 7 days only).")

if __name__ == "__main__":
    print("=" * 60)
    print("Punjab News Collector (Groq AI analysis, last 7 days)")
    print("=" * 60)

    if os.getenv("GITHUB_ACTIONS"):
        print(f"\n[{datetime.now()}] Running in GitHub Actions – single collection cycle")
        run_collection()
        print(f"[{datetime.now()}] Done. Exiting.")
    else:
        while True:
            print(f"\n[{datetime.now()}] Starting collection...")
            run_collection()
            print(f"[{datetime.now()}] Sleeping 4 hours...")
            time.sleep(14400)
