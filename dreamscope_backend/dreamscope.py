import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from dreamscope_backend.preprocess import *

model = SentenceTransformer("all-mpnet-base-v2")
lemmatizer = WordNetLemmatizer()

def match_dream(dream_text, top_k=5): # 5 best matches
    dream_text = lemmatize(dream_text)
    df = pd.read_csv("dreamscope_backend/data/dream_symbols.csv")
    embeddings = np.load("dreamscope_backend/data/symbol_embeddings.npy")

    query_embedding = model.encode([dream_text])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # Only include results with a similarity score above a certain threshold
            results.append({
                "Dream Symbol": df.iloc[idx]["symbol"],
                "Interpretation": df.iloc[idx]["meaning"],
                "score": round(float(similarities[idx]), 4)
                })
    return results


if __name__ == "__main__":
    dream =  "i dreamed about a birds and horses"
    results = match_dream(dream)

    for r in results:
        print(f"\n{r['Dream Symbol']} (score: {r['score']})")
        print(r['Interpretation'])
