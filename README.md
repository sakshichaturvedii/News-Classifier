# News Classifier (Topic Detection + Fake News Detection)

This is a **Flask-based web application** that performs:

- **News Topic Classification** — Identifies the topic of a news article (e.g., Politics, Business, Sports)
- **Fake News Detection** — Predicts whether the news is **Real** or **Fake**

---

## Features

Users can input a news **headline** or **paragraph** and get instant predictions on:

-  The most likely **topic category** (World, Business, Sports, Sci/Tech)
-  Whether the news is **Fake or Real**
-  Confidence levels of both predictions

---

## Technologies Used

- **Python 3**
- **Flask** — Backend web framework
- **Scikit-learn** — Logistic Regression, Multinomial Naive Bayes
- **TF-IDF Vectorizer** — Text preprocessing
- **HTML + CSS** — Frontend UI
- **Pickle** — Model serialization

---

## Model Details

### ✅ Fake News Classifier (Logistic Regression)
- **Accuracy:** 95.3%
- Balanced **precision** and **recall** across Fake and Real classes

### ✅ Topic Classifier (Multinomial Naive Bayes)
- **Accuracy:** 88.1%
- Strong performance across **World, Business, Sports, and Sci/Tech** topics

---

## Demo Video

You can watch a short demo of the project in the repository:  
**`News Classifier.mp4`**

It shows how a user can enter news text and view predictions along with confidence scores.

---




