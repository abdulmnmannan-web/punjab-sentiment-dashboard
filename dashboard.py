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

       # ────────────────────────────────────────────────
    # Collection Period + Download
    # ────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        if not filtered.empty and "fetched_at" in filtered.columns:
            min_date = filtered["fetched_at"].min()
            max_date = filtered["fetched_at"].max()
            st.info(f"**Article Collection Period:** {min_date.strftime('%d %b %Y')} → {max_date.strftime('%d %b %Y')}")
        else:
            st.info("Article Collection Period: Not available")

    with col2:
        # Download button
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Data (CSV)",
            data=csv,
            file_name="punjab_sentiment_articles.csv",
            mime="text/csv"
        )

    # ────────────────────────────────────────────────
    # Latest Headlines (News Portal Style)
    # ────────────────────────────────────────────────
    st.subheader("Latest Headlines")

        # ────────────────────────────────────────────────
    # News Portal Sentiment Breakdown
    # ────────────────────────────────────────────────
    st.subheader("News Portal Sentiment Breakdown")

    if filtered.empty:
        st.info("No data available for the selected filters.")
    else:
        portal_summary = (
            filtered
            .groupby(["source", "party", "sentiment_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for col in ["favoring", "against", "neutral"]:
            if col not in portal_summary.columns:
                portal_summary[col] = 0

        portal_summary["total"] = portal_summary["favoring"] + portal_summary["against"] + portal_summary["neutral"]

        avg_scores = (
            filtered
            .groupby(["source", "party"])["compound"]
            .mean()
            .reset_index()
            .rename(columns={"compound": "avg_score"})
        )

        portal_summary = portal_summary.merge(avg_scores, on=["source", "party"], how="left")
        portal_summary = portal_summary.sort_values("total", ascending=False)

        top_portals = portal_summary["source"].value_counts().head(15).index
        portal_summary = portal_summary[portal_summary["source"].isin(top_portals)]

        st.dataframe(
            portal_summary[["source", "party", "favoring", "against", "neutral", "total", "avg_score"]],
            use_container_width=True,
            hide_index=True
        )
        st.caption("Showing top 15 news sources by article volume.")

    # ────────────────────────────────────────────────
    # Collection Period + Download
    # ────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        if not filtered.empty and "fetched_at" in filtered.columns:
            min_date = filtered["fetched_at"].min()
            max_date = filtered["fetched_at"].max()
            st.info(f"**Article Collection Period:** {min_date.strftime('%d %b %Y')} → {max_date.strftime('%d %b %Y')}")
        else:
            st.info("Article Collection Period: Not available")

    with col2:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Data (CSV)",
            data=csv,
            file_name="punjab_sentiment_articles.csv",
            mime="text/csv"
        )

    # ────────────────────────────────────────────────
    # Latest Headlines (News Portal Style)
    # ────────────────────────────────────────────────
    st.subheader("Latest Headlines")

    recent = filtered.sort_values("fetched_at", ascending=False).head(40)

    if recent.empty:
        st.info("No headlines match the current filters.")
    else:
        for _, row in recent.iterrows():
            label = row.get("sentiment_label", "neutral")
            if label == "favoring":
                mood_emoji, color, badge = "🟢", "#138808", "Favoring"
            elif label == "against":
                mood_emoji, color, badge = "🔴", "#CC0000", "Against"
            else:
                mood_emoji, color, badge = "⚪", "#666666", "Neutral"

            title = row.get("title_english", row.get("title_original", "(no title)"))
            url = row.get("url", "")
            source = row.get("source", "Unknown")
            party = row.get("party", "")
            score = row.get("compound", 0)
            fetched = row.get("fetched_at")

            with st.container():
                st.markdown(
                    f"""
                    <div style="border-left: 5px solid {color}; padding: 12px 16px; margin-bottom: 12px; background-color: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:13px; color:#555;">{source} • {party}</span>
                            <span style="background-color:{color}; color:white; padding:2px 10px; border-radius:12px; font-size:12px;">{badge}</span>
                        </div>
                        <div style="margin-top:6px;">
                            <a href="{url}" target="_blank" style="text-decoration:none; color:#111; font-size:17px; font-weight:600; line-height:1.4;">
                                {title}
                            </a>
                        </div>
                        <div style="margin-top:8px; font-size:13px; color:#666;">
                            Score: {score:.3f} &nbsp;|&nbsp; {fetched}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("Why this classification?"):
                    st.write(row.get("analysis_reasoning", "No reasoning available"))

    if st.button("Refresh view"):
        st.cache_data.clear()
        st.rerun()
