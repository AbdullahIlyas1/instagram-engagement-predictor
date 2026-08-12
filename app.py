import streamlit as st
import joblib
import numpy as np
import re
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Page Config ──
st.set_page_config(page_title="Instagram Engagement Predictor", page_icon="📱", layout="centered")

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%); }
    .title-box {
        background: linear-gradient(135deg, #833ab4, #fd1d1d, #fcb045);
        padding: 2px; border-radius: 16px; margin-bottom: 30px;
    }
    .title-inner {
        background: #0f0f0f; border-radius: 14px; padding: 30px;
        text-align: center;
    }
    .title-inner h1 { color: white; font-size: 2.2em; margin: 0; }
    .title-inner p { color: #aaa; margin: 8px 0 0 0; font-size: 1em; }
    .result-high { background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 20px; border-radius: 12px; text-align: center; color: white; }
    .result-mid { background: linear-gradient(135deg, #f7971e, #ffd200);
        padding: 20px; border-radius: 12px; text-align: center; color: white; }
    .result-low { background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 20px; border-radius: 12px; text-align: center; color: white; }
    .result-score { font-size: 3em; font-weight: 800; margin: 0; }
    .result-label { font-size: 1em; opacity: 0.9; }
    .metric-card {
        background: #1e1e2e; border: 1px solid #333; border-radius: 10px;
        padding: 15px; text-align: center; margin: 5px 0;
    }
    .metric-card .val { font-size: 1.6em; font-weight: 700; color: white; }
    .metric-card .lbl { font-size: 0.75em; color: #aaa; margin-top: 4px; }
    .tip-box {
        background: #1e1e2e; border-left: 4px solid #833ab4;
        padding: 12px 16px; border-radius: 8px; margin: 6px 0; color: #ddd;
        font-size: 0.9em;
    }
    .footer { text-align: center; color: #555; font-size: 0.8em; margin-top: 40px; }
    div[data-testid="stTextArea"] textarea {
        background: #1e1e2e; border: 1px solid #444; color: white; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ──
model = joblib.load('best_model.pkl')
feature_cols = joblib.load('feature_cols.pkl')
analyzer = SentimentIntensityAnalyzer()

# ── Header ──
st.markdown("""
<div class="title-box">
  <div class="title-inner">
    <h1>📱 Instagram Engagement Predictor</h1>
    <p>Paste your caption and get an AI-powered engagement prediction before you post</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ──
caption = st.text_area("✍️ Your Instagram Caption", height=140,
    placeholder="e.g. Just launched our new collection! Drop a comment below and tag a friend 🔥 #fashion #style #new")

col1, col2, col3 = st.columns(3)
with col1:
    hashtag_count = st.number_input("# Hashtags", min_value=0, max_value=30, value=3)
with col2:
    post_hour = st.slider("Posting Hour", 0, 23, 12)
with col3:
    impressions = st.number_input("Expected Impressions", min_value=100, max_value=100000, value=10000)

post_day = st.selectbox("📅 Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

st.markdown("---")

# ── Predict ──
if st.button("🔮 Predict My Engagement", use_container_width=True):
    if not caption.strip():
        st.warning("Please enter a caption first!")
    else:
        clean = str(caption).lower()
        clean = re.sub(r'http\S+|www\S+', '', clean)
        clean = re.sub(r'#\w+', '', clean)
        clean = re.sub(r'@\w+', '', clean)
        clean = re.sub(r'[^\w\s]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        word_count = len(clean.split())
        char_count = len(clean)
        hashtag_density = hashtag_count / word_count if word_count > 0 else 0
        emoji_count = sum(1 for c in caption if ord(c) > 127462)
        has_question = int('?' in caption)
        cta_keywords = ['comment','share','tag','click','link in bio','follow','save','swipe','check','visit','subscribe','dm']
        has_cta = int(any(w in caption.lower() for w in cta_keywords))
        readability = textstat.flesch_kincaid_grade(clean)
        sentiment = analyzer.polarity_scores(caption)
        is_weekend = 1 if day_map[post_day] >= 5 else 0
        impressions_log = np.log1p(impressions)

        features = np.array([[
            word_count, char_count, hashtag_count, hashtag_density,
            emoji_count, has_question, has_cta, readability,
            sentiment['pos'], sentiment['neg'], sentiment['compound'],
            post_hour, day_map[post_day], is_weekend, impressions_log
        ]])

        prediction = model.predict(features)[0]

        # Result card
        if prediction >= 10:
            css_class = "result-high"
            label = "🔥 High Engagement Expected"
        elif prediction >= 5:
            css_class = "result-mid"
            label = "👍 Moderate Engagement Expected"
        else:
            css_class = "result-low"
            label = "📉 Low Engagement Expected"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-score">{prediction:.2f}%</div>
            <div class="result-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Post Analysis")
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, str(word_count), "Words"),
            (c2, f"{sentiment['compound']:.2f}", "Sentiment"),
            (c3, str(hashtag_count), "Hashtags"),
            (c4, f"{readability:.1f}", "Readability"),
        ]
        for col, val, lbl in metrics:
            with col:
                st.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="val">{"✅" if has_cta else "❌"}</div><div class="lbl">Call to Action</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="metric-card"><div class="val">{"✅" if has_question else "❌"}</div><div class="lbl">Has Question</div></div>', unsafe_allow_html=True)

        st.markdown("### 💡 Tips to Improve")
        tips = []
        if not has_cta:
            tips.append("Add a call to action — e.g. 'Comment below' or 'Tag a friend'")
        if hashtag_count < 3:
            tips.append("Try using 3-7 hashtags for better reach")
        if sentiment['compound'] < 0:
            tips.append("Consider a more positive tone in your caption")
        if word_count < 10:
            tips.append("Longer captions (15-20 words) tend to perform better")
        if post_hour < 7 or post_hour > 22:
            tips.append("Try posting between 9am and 9pm for better engagement")
        if readability > 12:
            tips.append("Simplify your language — shorter sentences perform better on Instagram")

        if tips:
            for tip in tips:
                st.markdown(f'<div class="tip-box">💬 {tip}</div>', unsafe_allow_html=True)
        else:
            st.success("Your post looks great! No major improvements needed.")

st.markdown('<div class="footer">Abdullah Ilyas | A00081833 | MSc Data Science Dissertation</div>', unsafe_allow_html=True)