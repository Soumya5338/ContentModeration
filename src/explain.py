# src/explain.py
from .preprocess import clean_text
import joblib
import shap

model = joblib.load("toxic_model.pkl")
vectorizer = model.named_steps["tfidf"]

def explain_text(text: str):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    explainer = shap.LinearExplainer(model.named_steps["clf"], vectorizer.transform(["dummy text"]))
    shap_values = explainer.shap_values(X)
    feature_names = vectorizer.get_feature_names_out()
    important_words = sorted(zip(feature_names, shap_values[0]), key=lambda x: -abs(x[1]))[:10]
    return important_words

