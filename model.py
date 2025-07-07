import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Training Fake News Classifier...")

df_fake = pd.read_csv('data/fake_news_clean.csv')

df_fake = df_fake.dropna()

tfidf_fake = TfidfVectorizer()
X_fake = tfidf_fake.fit_transform(df_fake['clean_title'].str.lower())  
y_fake = df_fake['label']  

X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(
    X_fake, y_fake, test_size=0.2, random_state=42
)

model_fake = LogisticRegression()
model_fake.fit(X_train_fake, y_train_fake)

y_pred_fake = model_fake.predict(X_test_fake)
print("\n Fake News Accuracy:", accuracy_score(y_test_fake, y_pred_fake))
print("\n Classification Report:\n", classification_report(y_test_fake, y_pred_fake))

with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model_fake, f)

with open('model/fake_news_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_fake, f)

print("Fake News Classifier trained and saved.\n")


print("Training Topic Classifier...")

df_topic = pd.read_csv('data/topic_news_clean.csv')

df_topic = df_topic.dropna()

tfidf_topic = TfidfVectorizer()
X_topic = tfidf_topic.fit_transform(df_topic['clean_text'].str.lower())  
y_topic = df_topic['label'] 

X_train_topic, X_test_topic, y_train_topic, y_test_topic = train_test_split(
    X_topic, y_topic, test_size=0.2, random_state=42
)

model_topic = MultinomialNB()
model_topic.fit(X_train_topic, y_train_topic)

y_pred_topic = model_topic.predict(X_test_topic)
print("\n Topic Classification Accuracy:", accuracy_score(y_test_topic, y_pred_topic))
print("\n Classification Report:\n", classification_report(y_test_topic, y_pred_topic))

with open('model/topic_model.pkl', 'wb') as f:
    pickle.dump(model_topic, f)

with open('model/topic_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_topic, f)

print(" Topic Classifier trained and saved.")
