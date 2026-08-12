import streamlit as st
import joblib
import numpy as np
import re
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Load model and features ──
model = joblib.load('best_model.pkl')
feature_cols = joblib.load('feature_cols.pkl')
analyzer = SentimentIntensityAnalyzer()

# ── Page config ──
st.set_page_config(page_title="Instagram Engagement Predictor", page_icon="📱", layout="centered")

st.title("📱 Instagram Engagement Predictor")
st.markdown("Enter your post details below to predict how well it will perform.")
st.markdown("---")

# ── User inputs ──
caption = st.text_area("✍️ Write your Instagram caption here", height=150,
                        placeholder="e.g. Just launched our new product! Check the link in bio 🔥 #fashion #style")

col1, col2, col3 = st.columns(3)
with col1:
    hashtag_count = st.number_input("# Hashtags", min_value=0, max_value=30, value=3)
with col2:
    post_hour = st.slider("Posting Hour", 0, 23, 12)
with col3:
    impressions = st.number_input("Expected Impressions", min_value=100, max_value=100000, value=10000)

post_day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

st.markdown("---")

# ── Predict button ──
if st.button("🔮 Predict Engagement", use_container_width=True):

    if not caption.strip():
        st.warning("Please enter a caption first!")
    else:
        # Clean caption
        clean = str(caption).lower()
        clean = re.sub(r'http\S+|www\S+', '', clean)
        clean = re.sub(r'#\w+', '', clean)
        clean = re.sub(r'@\w+', '', clean)
        clean = re.sub(r'[^\w\s]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Extract features
        word_count = len(clean.split())
        char_count = len(clean)
        hashtags_in_caption = len(re.findall(r'#\w+', caption))
        hashtag_density = hashtag_count / word_count if word_count > 0 else 0
        emoji_count = sum(1 for c in caption if ord(c) > 127462)
        has_question = int('?' in caption)
        cta_keywords = ['comment','share','tag','click','link in bio','follow','save','swipe','check','visit','subscribe','dm']
        has_cta = int(any(w in caption.lower() for w in cta_keywords))
        readability = textstat.flesch_kincaid_grade(clean)
        sentiment = analyzer.polarity_scores(caption)
        is_weekend = 1 if day_map[post_day] >= 5 else 0
        impressions_log = np.log1p(impressions)

        # Build feature array
        features = np.array([[
            word_count, char_count, hashtag_count, hashtag_density,
            emoji_count, has_question, has_cta, readability,
            sentiment['pos'], sentiment['neg'], sentiment['compound'],
            post_hour, day_map[post_day], is_weekend, impressions_log
        ]])

        prediction = model.predict(features)[0]

        # ── Show result ──
        st.markdown("### 🎯 Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Engagement Rate", f"{prediction:.2f}%")
        with col2:
            if prediction >= 10:
                st.success("🔥 High Engagement Expected!")
            elif prediction >= 5:
                st.warning("👍 Moderate Engagement Expected")
            else:
                st.error("📉 Low Engagement Expected")

        # ── Feature breakdown ──
        st.markdown("### 📊 Your Post Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", word_count)
            st.metric("Hashtags", hashtag_count)
        with col2:
            st.metric("Sentiment", f"{sentiment['compound']:.2f}")
            st.metric("Readability Grade", f"{readability:.1f}")
        with col3:
            st.metric("Has CTA", "Yes ✅" if has_cta else "No ❌")
            st.metric("Has Question", "Yes ✅" if has_question else "No ❌")

        # ── Tips ──
        st.markdown("### 💡 Tips to Improve Your Post")
        if has_cta == 0:
            st.info("Add a call to action — e.g. 'Comment below' or 'Tag a friend'")
        if hashtag_count < 3:
            st.info("Try using 3-7 hashtags for better reach")
        if sentiment['compound'] < 0:
            st.info("Consider a more positive tone in your caption")
        if word_count < 10:
            st.info("Longer captions (15-20 words) tend to perform better")
        if post_hour < 7 or post_hour > 22:
            st.info("Try posting between 9am-9pm for better engagement")

st.markdown("---")
st.caption("Abdullah Ilyas | A00081833 | MSc Data Science Dissertation")