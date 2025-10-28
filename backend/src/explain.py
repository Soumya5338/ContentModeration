import json
import time

# You must leave the apiKey blank for the Canvas environment.
API_KEY = "" 
# Using the recommended model for structured output and grounding
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + API_KEY
MAX_RETRIES = 3

def explain_text(parent: str, reply: str, flag_score: int) -> dict:
    """
    Uses the Gemini API with structured output to generate human-readable 
    moderation feedback (severity, reason, suggestion) based on the BERT score.
    """
    # 1. Define the System Instruction for the LLM
    system_prompt = (
        "You are an expert AI content moderator. Your task is to analyze a conversation "
        "and provide a structured, actionable moderation report. The context is a parent comment, "
        "and the text to be moderated is the reply comment. The 'flag_score' is the computed "
        "toxicity likelihood (0-100) from a deep learning model (BERT). "
        "Based on the flag_score and the text, assign severity, describe the reason, "
        "and suggest an action."
        "Severity thresholds: Low (<30), Medium (30-70), High (>70)."
    )

    # 2. Define the User Query
    user_query = (
        f"Analyze the following conversation based on the reply comment:\n\n"
        f"Parent Comment: '{parent}'\n"
        f"Reply Comment (to be moderated): '{reply}'\n\n"
        f"BERT Toxicity Flag Score: {flag_score}."
    )

    # 3. Define the Structured Output Schema (Required by the Flask app)
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "severity": {
                "type": "STRING",
                "description": "The determined severity (Low, Medium, or High) based on the flag_score."
            },
            "reason": {
                "type": "STRING",
                "description": "A concise explanation of why the reply was flagged or cleared, referencing specific language or intent in the reply, especially if the BERT score is high."
            },
            "suggestion": {
                "type": "STRING",
                "description": "The recommended moderation action. E.g., 'Approve comment.', 'Monitor closely.', or 'Flag reply for moderation.' based on severity."
            }
        },
        "required": ["severity", "reason", "suggestion"],
        "propertyOrdering": ["severity", "reason", "suggestion"]
    }
    
    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        },
        # Use Google Search for grounding to provide better-informed suggestions
        "tools": [{ "google_search": {} }] 
    }

    # 4. Make the API Call with Exponential Backoff
    for attempt in range(MAX_RETRIES):
        try:
            # We use the raw fetch here as this code is executed outside the browser
            response = fetch(API_URL, {
                'method': 'POST',
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps(payload)
            })

            result = response.json()
            
            # Extract and parse the JSON string from the LLM response
            if result.get('candidates') and result['candidates'][0].get('content'):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)
                return parsed_json
            
            print(f"Attempt {attempt + 1} failed: Unexpected LLM response structure.")

        except Exception as e:
            # print(f"LLM API Call Error (Attempt {attempt + 1}): {e}") # Suppressing due to retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s

    # 5. Return a safe fallback if all attempts fail
    return {
        "severity": "Unknown",
        "reason": "LLM analysis failed after multiple retries. BERT score was {}%.".format(flag_score),
        "suggestion": "Manual Review Required."
    }

if __name__ == '__main__':
    # This block is conceptual, as the API call won't run outside the environment.
    pass
