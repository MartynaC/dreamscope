import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import pipeline

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from dreamscope_backend.preprocess import *

model = SentenceTransformer("all-mpnet-base-v2")
lemmatizer = WordNetLemmatizer()

def match_dream(dream_text, top_k=5):
    dream_text = lemmatize(dream_text)
    df = pd.read_csv("dreamscope_backend/data/dream_symbols_clean_v5.csv")
    embeddings = np.load("dreamscope_backend/data/symbol_embeddings.npy")

    query_embedding = model.encode([dream_text])
    similarities = np.dot(embeddings, query_embedding.T).flatten() # type: ignore
    top_indices = np.argsort(similarities)[-(top_k * 10):][::-1]

    best_per_symbol = {}
    for idx in top_indices:
        symbol = df.iloc[idx]["symbol_clean"]
        score = similarities[idx]
        if symbol not in best_per_symbol or score > best_per_symbol[symbol]["score"]:
            best_per_symbol[symbol] = {
                "Dream Symbol": symbol,
                "Context": df.iloc[idx]["context"],
                "Interpretation": df.iloc[idx]["meaning_clean"],
                "score": round(float(score), 6)
            }

    dream_words = set(dream_text.lower().split())
    for symbol, data in best_per_symbol.items():
        symbol_words = set(symbol.split())
        if symbol_words & dream_words:
            data["score"] = round(data["score"] + 0.05, 6)

    results = sorted(best_per_symbol.values(), key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] > 0.3]

    return results[:top_k]


def match_emotions(dream_text):
    def initialize_model():
        return pipeline(
            'text-classification',
            model='SamLowe/roberta-base-go_emotions',
            top_k=None  # ← get ALL emotions, not just top 4
        )

    classifier = initialize_model()
    emotions = classifier(dream_text)[0]
    emotions = [e for e in emotions if e['label'] != 'neutral']
    
    # recalculate scores to sum to 100%
    total = sum(e['score'] for e in emotions)
    emotions = [{'label': e['label'], 'score': e['score'] / total} for e in emotions]
    
    # return top 4 after rescaling
    emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:4]
    return emotions


if __name__ == "__main__":
    dream = "i dreamed about birds and horses"
    results = match_dream(dream)
    emotions = match_emotions(dream)

    print(results)
    print(emotions)