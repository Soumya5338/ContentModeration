from flask import Flask, request, jsonify
from flask_cors import CORS
# from src.explain import explain_text # <-- REMOVED: This function needs to be rewritten for BERT
import joblib
from transformers import pipeline, AutoTokenizer
import time

app = Flask(__name__)
CORS(app)

# ---------------- BERT Model Loading ----------------
# NOTE: Instead of a scikit-learn pipeline, we now load the Hugging Face text-classification pipeline.
# We expect 'toxic_model.pkl' to contain the *name* of the Hugging Face model (e.g., 'unitary/toxic-bert').
try:
    # Load the model name placeholder saved by bert_train.py
    # If the file is missing or corrupted, use a default name.
    MODEL_NAME = joblib.load("toxic_model.pkl") 
    print(f"Loading BERT pipeline for model: {MODEL_NAME}...")
    
    # Initialize the BERT pipeline
    model = pipeline(
        "text-classification", 
        model=MODEL_NAME, # Use the model name
        tokenizer=MODEL_NAME, 
        return_all_scores=True, # Essential for getting probability scores
        device=-1 # -1 for CPU (default), 0 for first GPU
    )
    print("✅ BERT Model pipeline loaded successfully.")
except Exception as e:
    # Fallback if the necessary files or libraries aren't present
    print(f"⚠️ Failed to load BERT model. Error: {e}")
    print("Please ensure 'toxic_model.pkl' (containing the model name) and the 'transformers' library are installed.")
    model = None # Set model to None if loading fails
    # Set a placeholder name for the home route
    MODEL_NAME = "Loading Failed"

# ---------------- Helper Functions ----------------

def predict_comment_toxicity(text: str):
    """
    Predicts toxicity using the BERT pipeline.
    Output format is typically: [{'label': 'LABEL', 'score': 0.99}]
    """
    if not model or not text.strip():
        # Handle empty text or failed model load
        return "not toxic", 0.00, []

    # The pipeline outputs predictions for all classes 
    results = model(text)[0] 
    
    # We assume 'toxic' is the positive class. The specific label name can vary.
    # We check for 'toxic' (used by some models) and the generic 'LABEL_1' (default HF classification).
    toxic_result = next((res for res in results if res['label'].lower() == 'toxic'), 
                        next((res for res in results if res['label'] == 'LABEL_1'), None))
    
    if toxic_result:
        score = toxic_result['score'] * 100
        # Determine label based on a 50% threshold
        prediction = 1 if score > 50 else 0
        label = "toxic" if prediction == 1 else "not toxic"
    else:
        # Default fallback if the expected label isn't found
        label = "not toxic"
        score = 0.00

    # NOTE: BERT explainability is complex. Replacing `explain_text` with a simple placeholder logic.
    # To get real BERT explainability (top words based on attention), you'd need libraries like Captum.
    words = text.lower().split()
    word_counts = {}
    for word in words:
        # Simple removal of common English stop words
        if word not in ["the", "a", "an", "is", "it", "to", "and", "of", "in", "for"]: 
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get the top 3 most frequent words as a proxy for "top words"
    top_words = [w for w, count in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)][:3]

    return label, round(score, 2), top_words


def compute_flag_score(parent_score: float, reply_score: float):
    """Computes a combined moderation score."""
    return round(reply_score * 0.7 + parent_score * 0.3, 2)

def determine_reason_suggestion(parent_pred, reply_pred, flag_score):
    """Determines moderation reason, suggestion, and severity."""
    reason = ""
    suggestion = ""
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

    # Predictions
    parent_pred, parent_score, parent_top_words = predict_comment_toxicity(parent)
    reply_pred, reply_score, reply_top_words = predict_comment_toxicity(reply)
    
    flag_score = compute_flag_score(parent_score, reply_score)
    reason, suggestion, severity = determine_reason_suggestion(parent_pred, reply_pred, flag_score)
    
    suggested_reply = ""
    
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
        "suggested_reply": suggested_reply,
        "parent_top_words": parent_top_words,
        "reply_top_words": reply_top_words,
        "latency_ms": round((end_time - start_time) * 1000, 2)
    })

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running! Model loaded: " + (MODEL_NAME if 'MODEL_NAME' in locals() else 'None (Check logs)')

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
