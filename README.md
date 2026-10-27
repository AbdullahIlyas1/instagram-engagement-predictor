# 📱 Instagram Engagement Predictor

> Predict how well your Instagram post will perform — before you publish it.

---

## What This Project Does

You paste a caption. You get a predicted engagement rate. You get told exactly what to improve.

No fluff. Just data.

This is a machine learning system trained on 2,331 Instagram posts. It reads your caption, extracts 15 features from the text, and returns a score telling you how likely your post is to perform well.

It runs as a live web app — no coding needed.

**Try it here:** [Live App](https://instagram-engagement-predictor-fl2z.onrender.com)

---

## Why I Built This

I come from a digital and social media background. I watched brands spend serious money on content and get disappointing results — not because the idea was bad, but because the copy was wrong.

Nobody had a data-driven way to evaluate a caption before hitting post.

So for my MSc Data Science dissertation I decided to build one.

---

## How It Works

### 1. Data
- Dataset: 2,331 Instagram posts (sourced from Kaggle)
- Each post has a caption, like count, comment count, impression count, and timestamp

### 2. Engagement Rate Formula
```
Engagement Rate = (Likes + Comments) / Impressions × 100
```
Calculated from scratch rather than using a pre-existing column — to demonstrate the methodology and ensure consistency.

### 3. Features Extracted (15 total)

| Category | Features |
|---|---|
| Text | word_count, char_count, has_question, has_cta, readability_score, emoji_count |
| Sentiment | sentiment_pos, sentiment_neg, sentiment_neu, sentiment_compound (via VADER) |
| Hashtag | hashtag_count, hashtag_density |
| Temporal & Reach | post_hour, post_day, impressions_log |

### 4. Models Trained

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.63 | 15.78 | 8.89 |
| XGBoost | 0.67 | 14.84 | 6.59 |
| **Random Forest** ✅ | **0.76** | **12.70** | **6.18** |

Random Forest won. It explains 76% of the variance in engagement rate using only text and metadata.

### 5. SHAP Analysis
Used SHAP (SHapley Additive exPlanations) to explain why the model makes each prediction.

Top predictors:
- `impressions_log` — reach is the dominant factor
- `readability_score` — simpler captions perform better
- `post_hour` — timing matters
- `char_count` — caption length has a real effect

### 6. Web App
Built with Streamlit. User inputs a caption and gets:
- Predicted engagement rate
- Feature breakdown
- Tips to improve the score

---

## Tech Stack

```
Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP
VADER · textstat · Streamlit · joblib · Google Colab · GitHub
```

---

## Project Structure

```
instagram_app/
├── app.py               # Streamlit web application
├── best_model.pkl       # Trained Random Forest model
├── feature_cols.pkl     # Feature column names
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Run It Locally

```bash
git clone https://github.com/AbdullahIlyas1/instagram-engagement-predictor
cd instagram-engagement-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations (being honest)

- Dataset is synthetic — generated to represent realistic engagement, not scraped from real Instagram accounts
- Text-only features — no image or video analysis
- Impression count dominates predictions — which is partly outside a creator's control

---

## Version 2 — Coming Soon

Version 1 is a working proof of concept.

Version 2 will be trained on real Instagram data scraped via Apify — real posts, real engagement, real results. The pipeline is already built. It is just a matter of feeding it better data.

Follow along on LinkedIn for updates.

---

## About

Built by **Abdullah Ilyas** as part of an MSc Data Science dissertation.

- 🔗 [LinkedIn](https://linkedin.com/in/abdullahilyas)
- 🌐 [Live App](https://abdullahilyas1-instagram-engagement-predictor.streamlit.app)
- 📧 Available for data analyst roles in the UK

---

*Version 1 — August 2026*
