# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def build_model(max_features=5000, ngram_range=(1,2), max_iter=200):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("clf", LogisticRegression(max_iter=max_iter))
    ])

def save_model(model, path="toxic_model.pkl"):
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path="toxic_model.pkl"):
    return joblib.load(path)
