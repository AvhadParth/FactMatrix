import os
import requests
import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------
# 1. Streamlit Config (must be the FIRST command)
# ------------------------------------------------
st.set_page_config(
    page_title="FactMatrix",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# 2. Load Environment Variables
# ------------------------------------------------
load_dotenv()
FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY")

if not FACTCHECK_KEY:
    st.error("❌ Google Fact Check API key not found.\n\n👉 Add it to your .env as GOOGLE_FACTCHECK_KEY=your_key_here.")
    st.stop()

# ------------------------------------------------
# 3. Google Fact Check API Function
# ------------------------------------------------
def fact_check_google(query):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": FACTCHECK_KEY,
        "languageCode": "en",
        "pageSize": 5   # fetch up to 5 results
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        claims = data.get("claims", [])
        results = []
        for c in claims:
            text = c.get("text", "N/A")
            claimant = c.get("claimant", "Unknown source")
            review = c.get("claimReview", [])
            if review:
                site = review[0].get("publisher", {}).get("name", "Unknown")
                title = review[0].get("title", "")
                url = review[0].get("url", "")
                rating = review[0].get("textualRating", "No verdict")
                results.append({
                    "claim": text,
                    "claimant": claimant,
                    "site": site,
                    "title": title,
                    "url": url,
                    "rating": rating
                })
        return results
    except Exception as e:
        return [{"error": str(e)}]

# ------------------------------------------------
# 4. Streamlit UI Layout
# ------------------------------------------------
st.title("📊 FactMatrix: Structured Claim Analysis and Evidence Retrieval")

st.markdown("""
Welcome to **FactMatrix** — an intelligent tool for analyzing circulating claims.  

Enter a *news headline or statement* below, and FactMatrix will:  

1. Query the **Google Fact Check database** to find related claims.  
2. Display their **verdicts** (True, False, Misleading, etc).  
3. Provide **evidence articles** reviewed by trusted publishers.  

✨ This allows you to explore how a topic is being discussed, what false claims exist, and what verified evidence is available.
""")

# ------------------------------------------------
# 5. User Input
# ------------------------------------------------
user_input = st.text_area("✍ Enter a news headline or statement to analyze:", height=100)

if st.button("🔎 Analyze Claims"):
    if not user_input.strip():
        st.warning("⚠ Please enter a statement before analyzing.")
    else:
        # --------------------
        # Google Fact Check
        # --------------------
        st.subheader("🌍 FactMatrix Results")
        results = fact_check_google(user_input)

        if not results:
            st.info("ℹ No verified claims were found for this query.")
        else:
            for r in results:
                if "error" in r:
                    st.error(f"❌ Error fetching results: {r['error']}")
                else:
                    st.markdown(f"**🗣 Claim:** {r['claim']}")
                    st.write(f"- Source: {r['claimant']}")
                    st.write(f"- Reviewed by: {r['site']}")
                    st.write(f"- Verdict: *{r['rating']}*")
                    st.write(f"- Evidence: [{r['title']}]({r['url']})")
                    st.markdown("---")