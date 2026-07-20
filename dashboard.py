import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="Punjab Sentiment Dashboard", layout="wide")

DATA_FILE = "punjab_sentiment_live.csv"
KEYWORDS_FILE = "keywords.json"

SAFFRON, WHITE, GREEN, BLUE = "#FF9933", "#FFFFFF", "#138808", "#000080"
st.markdown(f"""
    <style>
    .stApp {{ background-color: #f8f9fa; }}
    section[data-testid="stSidebar"] {{ background-color: {WHITE}; border-right: 3px solid {SAFFRON}; }}
    h1, h2, h3 {{ color: {BLUE} !important; }}
    .stButton>button {{ background-color: {SAFFRON}; color: white; border-radius: 5px; }}
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def classify_language(source_type):
    if "Punjabi" in str(source_type):
        return "Punjabi"
    elif "English" in str(source_type):
        return "English"
    else:
        return "Archive (GDELT)"

st.title("Punjab Political Sentiment Dashboard")

with st.sidebar:
    st.header("Keywords")
    try:
        with open(KEYWORDS_FILE, encoding="utf-8") as f:
            kw = json.load(f)
    except FileNotFoundError:
        kw = {}
    new_kw_text = st.text_area("Edit keywords JSON", json.dumps(kw, indent=2, ensure_ascii=False), height=250)
    if st.button("Save Keywords"):
        try:
            parsed = json.loads(new_kw_text)
            with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            st.success("Saved. Collector will use these on its next cycle.")
        except json.JSONDecodeError:
            st.error("Invalid JSON -- fix syntax and try again.")

df = load_data()

if df.empty:
    st.warning("No data yet. Run `python collector.py` in another terminal tab first, and wait for it to finish one cycle.")
else:
    df["language"] = df["source_type"].apply(classify_language)

    col_a, col_b = st.columns(2)
    with col_a:
        party_filter = st.multiselect("Party", df["party"].dropna().unique(), default=list(df["party"].dropna().unique()))
    with col_b:
        lang_filter = st.multiselect("Language / Source", df["language"].unique(), default=list(df["language"].unique()))

    filtered = df[df["party"].isin(party_filter) & df["language"].isin(lang_filter)]

    st.subheader("Party Scorecards")
    all_parties = sorted(df["party"].dropna().unique())
    cols = st.columns(len(all_parties)) if all_parties else []
    for i, party in enumerate(all_parties):
        party_df = filtered[filtered["party"] == party]
        with cols[i]:
            if len(party_df) == 0:
                st.metric(label=party, value="0 articles")
            else:
                avg_sent = party_df["compound"].mean()
                favoring = (party_df["sentiment_label"] == "favoring").sum()
                against = (party_df["sentiment_label"] == "against").sum()
                neutral = (party_df["sentiment_label"] == "neutral").sum()
                st.metric(label=party, value=f"{avg_sent:+.3f}", delta=f"{len(party_df)} articles")
                st.caption(f"😊 {favoring} favoring | 😞 {against} against | 😐 {neutral} neutral")

    st.subheader("Average Sentiment by Party")
    summary = filtered.groupby("party")["compound"].agg(["mean", "count"]).reset_index()
    fig = px.bar(summary, x="party", y="mean", text="count", labels={"mean": "Avg Sentiment"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Latest Headlines")
    recent = filtered.sort_values("fetched_at", ascending=False).head(30)

    if recent.empty:
        st.info("No headlines match the current filters.")
    else:
        for _, row in recent.iterrows():
            label = row.get("sentiment_label", "neutral")
            if label == "favoring":
                mood_emoji, color = "😊", GREEN
            elif label == "against":
                mood_emoji, color = "😞", "#CC0000"
            else:
                mood_emoji, color = "😐", "#888888"

            col1, col2 = st.columns([0.08, 0.92])
            with col1:
                st.markdown(f"<h2 style='color:{color};'>{mood_emoji}</h2>", unsafe_allow_html=True)
            with col2:
                url = row.get("url", "")
                title = row.get("title_english", row.get("title_original", "(no title)"))
                if url and isinstance(url, str):
                    st.markdown(f"<a href='{url}' target='_blank' style='text-decoration:none;color:black;font-weight:bold;font-size:16px;'>{title}</a>", unsafe_allow_html=True)
                else:
                    st.write(f"**{title}**")

                st.markdown(f"<span style='background-color:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px;'>{label} {row.get('party','')}</span>", unsafe_allow_html=True)
                st.caption(f"Searched: {row.get('searched_party','N/A')} | {row.get('language','')} | {row.get('source','')} | {row.get('fetched_at')}")
                st.caption(f"Score: {row.get('compound', 0):.3f}")

                with st.expander("Why this label?"):
                    st.write(f"**Analysis:** {row.get('analysis_reasoning', 'No reasoning available')}")
                    st.write(f"**Confidence:** {row.get('confidence', 'N/A')}")
            st.markdown("---")
                # ────────────────────────────────────────────────
    # News Portal Sentiment Breakdown
    # ────────────────────────────────────────────────
    st.subheader("News Portal Sentiment Breakdown")

    if filtered.empty:
        st.info("No data available for the selected filters.")
    else:
        # Create a summary table
        portal_summary = (
            filtered
            .groupby(["source", "party", "sentiment_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Ensure the three sentiment columns exist
        for col in ["favoring", "against", "neutral"]:
            if col not in portal_summary.columns:
                portal_summary[col] = 0

        portal_summary["total"] = portal_summary["favoring"] + portal_summary["against"] + portal_summary["neutral"]

        # Average sentiment score per source + party
        avg_scores = (
            filtered
            .groupby(["source", "party"])["compound"]
            .mean()
            .reset_index()
            .rename(columns={"compound": "avg_score"})
        )

        portal_summary = portal_summary.merge(avg_scores, on=["source", "party"], how="left")

        # Sort by total articles
        portal_summary = portal_summary.sort_values("total", ascending=False)

        # Show only top portals to keep it clean
        top_portals = portal_summary["source"].value_counts().head(15).index
        portal_summary = portal_summary[portal_summary["source"].isin(top_portals)]

        st.dataframe(
            portal_summary[["source", "party", "favoring", "against", "neutral", "total", "avg_score"]],
            use_container_width=True,
            hide_index=True
        )

        st.caption("Showing top 15 news sources by article volume. Positive = favoring, Negative = against.")

if st.button("Refresh view"):
    st.cache_data.clear()
    st.rerun()
