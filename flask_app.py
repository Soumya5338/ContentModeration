from flask import Flask, request, jsonify
from src.explain import explain_text
import joblib

app = Flask(__name__)

# Load model once at startup
model = joblib.load("toxic_model.pkl")

# ---------------- Helper Functions ----------------

def predict_comment_toxicity(text: str):
    """
    Predict toxicity, return label, probability score (0-100), and top words.
    """
    prediction = model.predict([text])[0]
    score = float(model.predict_proba([text])[0][1]) * 100  # probability of being toxic
    top_words = explain_text(text)
    label = "toxic" if prediction == 1 else "not toxic"
    return label, round(score, 2), top_words

def compute_flag_score(parent_score: float, reply_score: float):
    """
    Compute overall conversation flag score.
    Reply toxicity has higher weight.
    """
    return round(reply_score * 0.7 + parent_score * 0.3, 2)

def determine_reason_suggestion(parent_pred, reply_pred, flag_score):
    """
    Generate context-aware reason and suggestion.
    """
    if reply_pred == "toxic":
        reason = "Reply is toxic and should be reviewed immediately."
        suggestion = "Flag reply for moderation."
    elif parent_pred == "toxic":
        reason = "Parent comment is toxic, but reply is not toxic."
        suggestion = "Warn about toxic parent; reply is safe."
    elif flag_score > 50:
        reason = "Conversation shows potential toxicity."
        suggestion = "Review both comments."
    else:
        reason = "Neither comment is toxic."
        suggestion = "Safe to post."

    # Optional severity
    severity = "High" if flag_score > 70 else "Medium" if flag_score > 40 else "Low"
    return reason, suggestion, severity

# ---------------- Flask Route ----------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    parent = data.get("parent", "").strip()
    reply = data.get("reply", "").strip()

    if not parent and not reply:
        return jsonify({"error": "Both parent and reply cannot be empty."}), 400

    # Predict each comment
    parent_pred, parent_score, parent_top_words = predict_comment_toxicity(parent)
    reply_pred, reply_score, reply_top_words = predict_comment_toxicity(reply)

    # Compute combined flag score
    flag_score = compute_flag_score(parent_score, reply_score)

    # Context-aware reasoning
    reason, suggestion, severity = determine_reason_suggestion(parent_pred, reply_pred, flag_score)

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"

    return jsonify({
        "parent": parent,
        "reply": reply,
        "parent_prediction": parent_pred,
        "reply_prediction": reply_pred,
        "parent_score": parent_score,
        "reply_score": reply_score,
        "flag_score": flag_score,
        "severity": severity,
        "reason": reason,
        "suggestion": suggestion,
        "parent_top_words": parent_top_words,
        "reply_top_words": reply_top_words
    })

# ---------------- Run Server ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
