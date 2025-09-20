# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from preprocess import clean_text, combine_context

# Change this path if your CSV is elsewhere
DATA_PATH = "data/train.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Example: combine previous comment as parent for context
    df["parent"] = df["comment_text"].shift(1).fillna("")
    df["comment_with_context"] = df.apply(lambda row: combine_context(row["parent"], row["comment_text"]), axis=1)

    # Simplify to toxic vs non-toxic (0 = clean, 1 = toxic)
    # Here we consider "toxic" label as 1 if any of the toxic categories > 0
    toxic_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    df["label"] = df[toxic_columns].sum(axis=1).apply(lambda x: 1 if x>0 else 0)

    return df[["comment_with_context", "label"]]

def main():
    df = load_data()
    X = df["comment_with_context"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print("Classification Report:\n", classification_report(y_val, y_pred))

    joblib.dump(pipeline, "toxic_model.pkl")
    print("✅ Model saved as toxic_model.pkl")

if __name__ == "__main__":
    main()
