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

# Sidebar (kept but cleaner)
with st.sidebar:
    st.header("Filters & Settings")
    st.caption("Advanced options")
    try:
        with open(KEYWORDS_FILE, encoding="utf-8") as f:
            kw = json.load(f)
    except FileNotFoundError:
        kw = {}
    with st.expander("Edit Keywords (Advanced)"):
        new_kw_text = st.text_area("Keywords JSON", json.dumps(kw, indent=2, ensure_ascii=False), height=200)
        if st.button("Save Keywords"):
            try:
                parsed = json.loads(new_kw_text)
                with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
                st.success("Saved")
            except:
                st.error("Invalid JSON")

df = load_data()

if df.empty:
    st.warning("No data available.")
else:
    df["language"] = df["source_type"].apply(classify_language)

    # Filters
    col_a, col_b = st.columns(2)
    with col_a:
        party_filter = st.multiselect("Party", sorted(df["party"].dropna().unique()), default=list(df["party"].dropna().unique()))
    with col_b:
        lang_filter = st.multiselect("Language / Source", sorted(df["language"].unique()), default=list(df["language"].unique()))

    filtered = df[df["party"].isin(party_filter) & df["language"].isin(lang_filter)]

    # ─── Executive Summary ───
    st.markdown("### Executive Summary")
    if not filtered.empty:
        total = len(filtered)
        min_date = filtered["fetched_at"].min().strftime("%d %b %Y")
        max_date = filtered["fetched_at"].max().strftime("%d %b %Y")
        against_pct = (filtered["sentiment_label"] == "against").mean() * 100

        party_stats = filtered.groupby("party").agg(
            avg_score=("compound", "mean"),
            count=("compound", "count")
        ).reset_index()

        most_neg = party_stats.loc[party_stats["avg_score"].idxmin()]
        most_vol = party_stats.loc[party_stats["count"].idxmax()]

        st.info(f"""
**Data Period:** {min_date} → {max_date}  
**Total Articles Analysed:** {total}

**Key Findings:**
- Overall media tone is largely critical (**{against_pct:.0f}%** of coverage is negative).
- **{most_vol['party']}** has received the highest volume of coverage ({int(most_vol['count'])} articles).
- **{most_neg['party']}** currently has the most negative average score ({most_neg['avg_score']:.2f}).
- Coverage across major parties remains heavily critical in this period.
        """)

    # ─── Party Scorecards ───
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
                total = len(party_df)
                against = (party_df["sentiment_label"] == "against").sum()
                against_pct = (against / total) * 100
                favoring = (party_df["sentiment_label"] == "favoring").sum()
                neutral = (party_df["sentiment_label"] == "neutral").sum()

                st.metric(label=party, value=f"{avg_sent:+.3f}", delta=f"{total} articles")
                st.caption(f"Negative: **{against_pct:.0f}%**")
                st.caption(f"😊 {favoring}  😞 {against}  😐 {neutral}")
                if total < 10:
                    st.caption("⚠️ Low sample")

    # ─── Average Sentiment Chart ───
    st.subheader("Average Sentiment by Party")
    if not filtered.empty:
        summary = filtered.groupby("party")["compound"].agg(["mean", "count"]).reset_index()
        fig = px.bar(summary, x="party", y="mean", text="count",
                     labels={"mean": "Avg Sentiment", "party": "Party"},
                     color="mean", color_continuous_scale=["#CC0000", "#FF9933", "#138808"])
        st.plotly_chart(fig, use_container_width=True)

    # ─── Top Headlines ───
    st.subheader("Top Headlines")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Negative**")
        neg = filtered[filtered["sentiment_label"] == "against"].nsmallest(5, "compound")
        if neg.empty:
            st.write("No data")
        else:
            for _, row in neg.iterrows():
                st.markdown(f"- [{row['title_english'][:90]}...]({row['url']})  \n  <small>{row['source']} • {row['party']} • {row['compound']:.2f}</small>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Most Positive**")
        pos = filtered[filtered["sentiment_label"] == "favoring"].nlargest(5, "compound")
        if pos.empty:
            st.write("No strongly positive headlines")
        else:
            for _, row in pos.iterrows():
                st.markdown(f"- [{row['title_english'][:90]}...]({row['url']})  \n  <small>{row['source']} • {row['party']} • {row['compound']:.2f}</small>", unsafe_allow_html=True)

    # ─── News Portal Breakdown ───
    st.subheader("News Portal Sentiment Breakdown")
    if not filtered.empty:
        portal = (filtered.groupby(["source", "party", "sentiment_label"])
                  .size().unstack(fill_value=0).reset_index())
        for col in ["favoring", "against", "neutral"]:
            if col not in portal.columns:
                portal[col] = 0
        portal["total"] = portal["favoring"] + portal["against"] + portal["neutral"]
        avg = filtered.groupby(["source", "party"])["compound"].mean().reset_index().rename(columns={"compound": "avg_score"})
        portal = portal.merge(avg, on=["source", "party"], how="left")
        portal = portal.sort_values("total", ascending=False)
        top = portal["source"].value_counts().head(12).index
        portal = portal[portal["source"].isin(top)]

        st.dataframe(portal[["source", "party", "favoring", "against", "neutral", "total", "avg_score"]],
                     use_container_width=True, hide_index=True)

    # ─── Download + Period ───
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        if not filtered.empty:
            st.info(f"**Collection Period:** {filtered['fetched_at'].min().strftime('%d %b %Y')} → {filtered['fetched_at'].max().strftime('%d %b %Y')}")
    with c2:
        st.download_button("Download Full Data (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                           "punjab_sentiment_articles.csv", "text/csv", key="dl")

    # ─── All Articles with Search ───
    st.subheader("All Articles")
    search = st.text_input("Search headlines", placeholder="Type to filter headlines...")

    display = filtered[["fetched_at", "source", "party", "sentiment_label", "compound", "title_english", "url"]].copy()
    display = display.rename(columns={
        "fetched_at": "Date", "source": "News Portal", "party": "Party",
        "sentiment_label": "Sentiment", "compound": "Score",
        "title_english": "Headline", "url": "Link"
    }).sort_values("Date", ascending=False)

    if search:
        display = display[display["Headline"].str.contains(search, case=False, na=False)]

    st.dataframe(display, use_container_width=True, hide_index=True,
                 column_config={
                     "Link": st.column_config.LinkColumn("Link"),
                     "Score": st.column_config.NumberColumn(format="%.2f"),
                     "Date": st.column_config.DatetimeColumn(format="DD MMM YYYY")
                 })
    st.caption(f"Showing {len(display)} articles")

if st.button("Refresh view"):
    st.cache_data.clear()
    st.rerun()
