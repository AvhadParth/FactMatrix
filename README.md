# Fake News Detection — Full local system (macOS)

Overview:
- TF-IDF + Linear SVM baseline model for fake-news classification.
- Streamlit UI for manual checks + database browsing.
- Automated fetcher: NewsAPI -> model -> Google Fact Check Tools -> SQLite.

Prereqs:
- macOS (zsh)
- Python 3.9+ via Homebrew
- NewsAPI key and Google Fact Check API key

Quickstart (exact commands):
1) Clone / cd into project:
   cd ~/fake-news-detection

2) Setup venv:
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt

3) Add API keys to ~/.zshrc:
   export NEWSAPI_KEY="..."
   export GOOGLE_FACTCHECK_API_KEY="..."
   source ~/.zshrc

4) Train the model:
   python train_model.py --data data/train.csv --out models/fake_news_clf.joblib

5) Run Streamlit UI:
   streamlit run app.py

6) Run fetch once:
   python news_fetch_and_verify.py --once

7) Schedule:
   - Use cron or launchd (see notes in project documentation).

Files:
- train_model.py : trains and saves model
- news_fetch_and_verify.py : fetch + verify + store
- app.py : Streamlit UI
- data/train.csv : training data (sample)
- data/news_verifications.db : created at runtime
- models/fake_news_clf.joblib : saved model

Notes:
- Replace sample train.csv with a larger curated dataset for real evaluation.
- Do not commit your API keys to source control.