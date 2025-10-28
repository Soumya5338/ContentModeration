from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from transformers import pipeline
import time

app = Flask(__name__)
CORS(app)

# ---------------- BERT Model Loading ----------------
try:
    MODEL_NAME = joblib.load("toxic_model.pkl")  # should contain Hugging Face model name
    print(f"Loading BERT pipeline for model: {MODEL_NAME}...")
    
    model = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        return_all_scores=True,
        device=-1  # CPU
    )
    print("✅ BERT Model pipeline loaded successfully.")
except Exception as e:
    print(f"⚠️ Failed to load BERT model. Error: {e}")
    model = None
    MODEL_NAME = "Loading Failed"

# ---------------- Helper Functions ----------------
def predict_comment_toxicity(text: str):
    if not model or not text.strip():
        return "not toxic", 0.0, []

    results = model(text)[0]

    toxic_result = next(
        (res for res in results if res['label'].lower() == 'toxic'),
        next((res for res in results if res['label'] == 'LABEL_1'), None)
    )

    if toxic_result:
        score = toxic_result['score'] * 100
        label = "toxic" if score > 50 else "not toxic"
    else:
        label = "not toxic"
        score = 0.0

    # Simple top words placeholder
    words = [w for w in text.lower().split() if w not in ["the","a","an","is","it","to","and","of","in","for"]]
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    top_words = [w for w, _ in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)][:3]

    return label, round(score, 2), top_words

def compute_flag_score(parent_score: float, reply_score: float):
    return round(reply_score * 0.7 + parent_score * 0.3, 2)

def determine_reason_suggestion(parent_pred, reply_pred, flag_score):
    if reply_pred == "toxic":
        reason = "Reply is toxic and should be reviewed immediately."
        suggestion = "Change the comment."
    elif parent_pred == "toxic":
        reason = "Parent comment is toxic, but reply is not toxic."
        suggestion = "Warn about toxic parent; reply is safe."
    elif flag_score > 50:
        reason = "Conversation shows potential toxicity."
        suggestion = "Review both comments."
    else:
        reason = "Neither comment is toxic."
        suggestion = "Safe to post."

    severity = "High" if flag_score > 70 else "Medium" if flag_score > 40 else "Low"
    return reason, suggestion, severity

# ---------------- Flask Routes ----------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.get_json(force=True)

    parent = data.get("parent", "").strip()
    reply = data.get("reply", "").strip()

    if not parent and not reply:
        return jsonify({"error": "Both parent and reply cannot be empty."}), 400

    parent_pred, parent_score, parent_top_words = predict_comment_toxicity(parent)
    reply_pred, reply_score, reply_top_words = predict_comment_toxicity(reply)

    flag_score = compute_flag_score(parent_score, reply_score)
    reason, suggestion, severity = determine_reason_suggestion(parent_pred, reply_pred, flag_score)

    end_time = time.time()

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
        "reply_top_words": reply_top_words,
        "latency_ms": round((end_time - start_time) * 1000, 2)
    })

@app.route("/", methods=["GET"])
def home():
    return f"Flask server is running! Model loaded: {MODEL_NAME}"

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
