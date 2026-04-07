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

def match_emotions(dream_text):

    # Initializing model
    def initialize_model():
        emotion_classifier = pipeline(
            'text-classification',
            model='SamLowe/roberta-base-go_emotions',
            top_k=4
        )
        return emotion_classifier

    # Classifying emotions
    def classify(emotion_classifier, dream_text):
        response = classifier(dream_text)
        return response

    classifier = initialize_model()
    emotions = classify(classifier, dream_text)

    return emotions[0]


if __name__ == "__main__":

    dream =  "i dreamed about birds and horses"
    results = match_dream(dream)

    # for r in results:
    #     print(f"\n{r['Dream Symbol']} (score: {r['score']})")
    #     print(r['Interpretation'])

    emotions = match_emotions(dream)[0]

    # for emotion in emotions:
    #     print(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

    print(results)
    print(emotions)
