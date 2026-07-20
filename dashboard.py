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
        df = pd.read_csv(DATA_FILE, encoding="utf-8")
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
            st.success("Saved.")
        except json.JSONDecodeError:
            st.error("Invalid JSON")

df = load_data()

if df.empty:
    st.warning("No data yet.")
else:
    df["language"] = df["source_type"].apply(classify_language)

    # Filters
    col_a, col_b = st.columns(2)
    with col_a:
        party_filter = st.multiselect("Party", df["party"].dropna().unique(), default=list(df["party"].dropna().unique()))
    with col_b:
        lang_filter = st.multiselect("Language / Source", df["language"].unique(), default=list(df["language"].unique()))

    filtered = df[df["party"].isin(party_filter) & df["language"].isin(lang_filter)]

    # ────────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ────────────────────────────────────────────────
    st.markdown("### Executive Summary")

    if not filtered.empty:
        total_articles = len(filtered)
        min_date = filtered["fetched_at"].min().strftime("%d %b %Y")
        max_date = filtered["fetched_at"].max().strftime("%d %b %Y")

        party_stats = filtered.groupby("party").agg(
            avg_score=("compound", "mean"),
            count=("compound", "count"),
            against=("sentiment_label", lambda x: (x == "against").sum())
        ).reset_index()

        most_negative = party_stats.loc[party_stats["avg_score"].idxmin()]
        most_covered = party_stats.loc[party_stats["count"].idxmax()]

        against_pct = (filtered["sentiment_label"] == "against").mean() * 100

        st.info(f"""
**Data Period:** {min_date} → {max_date}  
**Total Articles Analysed:** {total_articles}

**Key Findings:**
- Overall media tone is largely **critical** ({against_pct:.0f}% of coverage is negative).
- **{most_negative['party']}** is facing the most negative coverage (Avg score: {most_negative['avg_score']:.2f}).
- **{most_covered['party']}** has received the highest volume of coverage ({int(most_covered['count'])} articles).
- Coverage remains heavily skewed towards critical reporting across major parties.
        """)
    else:
        st.info("No data available for the selected filters.")

    # ────────────────────────────────────────────────
    # Party Scorecards
    # ────────────────────────────────────────────────
    st.subheader("Party Scorecards")
    all_parties = sorted(df["party"].dropna().unique())
    cols = st.columns(len(all_parties)) if all_parties else []
    for i, party in enumerate(all_parties):
        party_df = filtered[filtered["party"] == party]
        with cols[i]:
            if len(party_df) == 0:
                st.metric(label=party, value="—")
            else:
                avg_sent = party_df["compound"].mean()
                favoring = (party_df["sentiment_label"] == "favoring").sum()
                against = (party_df["sentiment_label"] == "against").sum()
                neutral = (party_df["sentiment_label"] == "neutral").sum()
                st.metric(label=party, value=f"{avg_sent:+.3f}", delta=f"{len(party_df)} articles")
                st.caption(f"😊 {favoring} | 😞 {against} | 😐 {neutral}")

    # Average Sentiment Chart
    st.subheader("Average Sentiment by Party")
    summary = filtered.groupby("party")["compound"].agg(["mean", "count"]).reset_index()
    fig = px.bar(summary, x="party", y="mean", text="count", labels={"mean": "Avg Sentiment"},
                 color="mean", color_continuous_scale=["#CC0000", "#FF9933", "#138808"])
    st.plotly_chart(fig, use_container_width=True)

    # News Portal Breakdown
    st.subheader("News Portal Sentiment Breakdown")
    if not filtered.empty:
        portal_summary = (
            filtered.groupby(["source", "party", "sentiment_label"])
            .size().unstack(fill_value=0).reset_index()
        )
        for col in ["favoring", "against", "neutral"]:
            if col not in portal_summary.columns:
                portal_summary[col] = 0
        portal_summary["total"] = portal_summary["favoring"] + portal_summary["against"] + portal_summary["neutral"]
        avg_scores = filtered.groupby(["source", "party"])["compound"].mean().reset_index().rename(columns={"compound": "avg_score"})
        portal_summary = portal_summary.merge(avg_scores, on=["source", "party"], how="left")
        portal_summary = portal_summary.sort_values("total", ascending=False)
        top_portals = portal_summary["source"].value_counts().head(12).index
        portal_summary = portal_summary[portal_summary["source"].isin(top_portals)]

        st.dataframe(
            portal_summary[["source", "party", "favoring", "against", "neutral", "total", "avg_score"]],
            use_container_width=True,
            hide_index=True
        )
        st.caption("Top news sources by volume")

    # Collection Period + Download
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        if not filtered.empty:
            min_date = filtered["fetched_at"].min().strftime("%d %b %Y")
            max_date = filtered["fetched_at"].max().strftime("%d %b %Y")
            st.info(f"**Article Collection Period:** {min_date} → {max_date}")
    with col2:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Data (CSV)",
            data=csv,
            file_name="punjab_sentiment_articles.csv",
            mime="text/csv",
            key="download_full_data"
        )

    # All Articles Table
    st.subheader("All Articles")
    if not filtered.empty:
        display_df = filtered[["fetched_at", "source", "party", "sentiment_label", "compound", "title_english", "url"]].copy()
        display_df = display_df.rename(columns={
            "fetched_at": "Date",
            "source": "News Portal",
            "party": "Party",
            "sentiment_label": "Sentiment",
            "compound": "Score",
            "title_english": "Headline",
            "url": "Link"
        }).sort_values("Date", ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link"),
                "Score": st.column_config.NumberColumn(format="%.3f"),
                "Date": st.column_config.DatetimeColumn(format="DD MMM YYYY")
            }
        )
        st.caption(f"Showing {len(display_df)} articles")

if st.button("Refresh view"):
    st.cache_data.clear()
    st.rerun()
