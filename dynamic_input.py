import requests

url = "http://10.122.38.55:5000/predict"

while True:
    parent = input("Enter parent comment (or 'exit' to quit): ")
    if parent.lower() == 'exit':
        break
    reply = input("Enter reply comment: ")

    data = {"parent": parent, "reply": reply}
    response = requests.post(url, json=data)
    result = response.json()

    print(f"Parent: {result['parent']}")
    print(f"Reply: {result['reply']}")
    print(f"Parent Prediction: {result['parent_prediction']} ({result['parent_score']}%)")
    print(f"Reply Prediction: {result['reply_prediction']} ({result['reply_score']}%)")
    print(f"Flag Score: {result['flag_score']}% | Severity: {result['severity']}")
    print(f"Reason: {result['reason']}")
    print(f"Suggestion: {result['suggestion']}")
    print(f"Parent Top Words: {result.get('parent_top_words', [])}")
    print(f"Reply Top Words: {result.get('reply_top_words', [])}")
    print("-" * 60)
