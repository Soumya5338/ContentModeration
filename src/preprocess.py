# src/preprocess.py
import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def combine_context(parent: str, reply: str) -> str:
    """Combine parent + reply for context-aware moderation"""
    parent_clean = clean_text(parent) if parent else ""
    reply_clean = clean_text(reply)
    return f"Parent: {parent_clean}. Reply: {reply_clean}"
