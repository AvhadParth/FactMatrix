import os
import requests
import argparse
from datetime import datetime
import joblib

# ========================
# 1. Load API Keys
# ========================
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

def assert_keys():
    missing = []
    if not NEWSAPI_KEY:
        missing.append("NEWSAPI_KEY")
    if not GOOGLE_FACTCHECK_API_KEY:
        missing.append("GOOGLE_FACTCHECK_API_KEY")
    if missing:
        raise RuntimeError(f"Missing environment variable(s): {', '.join(missing)}")

# ========================
# 2. Load Model
# ========================
MODEL_PATH = "models/fake_news_clf.joblib"
pipe = None

def load_model():
    global pipe
    if pipe is None:
        pipe = joblib.load(MODEL_PATH)

# ========================
# 3. Fetch News (Everything Endpoint)
# ========================
def fetch_news(query="climate", language="en", page_size=5):
    """
    Fetch news articles from NewsAPI using the 'everything' endpoint.
    Allows keyword-based search instead of just top headlines.
    """
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"language={language}&"
        f"sortBy=publishedAt&"
        f"pageSize={page_size}&"
        f"apiKey={NEWSAPI_KEY}"
    )

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Error fetching news: {resp.status_code}, {resp.text}")
        return []

    data = resp.json()
    articles = data.get("articles", [])
    print(f"[{datetime.now().isoformat()}] Fetched {len(articles)} articles for query '{query}'")
    return articles

# ========================
# 4. Google Fact Check
# ========================
def fact_check_claim(claim):
    url = (
        f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
        f"?query={claim}&key={GOOGLE_FACTCHECK_API_KEY}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return f"Fact-check API error: {resp.status_code}"

    data = resp.json()
    claims = data.get("claims", [])
    if not claims:
        return "No claims found"

    # Just return the first claim review
    review = claims[0].get("claimReview", [{}])[0]
    publisher = review.get("publisher", {}).get("name", "Unknown")
    text = review.get("textualRating", "No rating")
    return f"{publisher}: {text}"

# ========================
# 5. Verify News Articles
# ========================
def verify_articles(articles):
    load_model()
    for art in articles:
        title = art.get("title", "")
        url = art.get("url", "")
        if not title:
            continue
        pred = pipe.predict([title])[0]
        fact = fact_check_claim(title)

        print("-" * 80)
        print(f"Title: {title}")
        print(f"Source: {art.get('source', {}).get('name', 'Unknown')}")
        print(f"URL: {url}")
        print(f"Prediction: {pred}")
        print(f"Fact-check: {fact}")

# ========================
# 6. Run Once
# ========================
def run_once(query):
    assert_keys()
    articles = fetch_news(query=query)
    verify_articles(articles)

# ========================
# 7. Main CLI
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection with NewsAPI + Fact-check")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--query", type=str, default="climate", help="Keyword to search in news")
    args = parser.parse_args()

    if args.once:
        run_once(query=args.query)