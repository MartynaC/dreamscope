import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

model = SentenceTransformer("all-mpnet-base-v2")
lemmatizer = WordNetLemmatizer()

 #Run this code once when dataset is updated or when model has changed
 #to create the embeddings for the dream symbols and interpretations,
 #then comment it out to avoid re-running it every time.

#df = pd.read_csv("raw_data/dreams_interpretations.csv")
#print(df.head())
#print(df.columns)
#
#texts = (df["Dream Symbol"] + " " + df["Interpretation"]).tolist()
#print(texts[0])
#
#embeddings = model.encode(texts, show_progress_bar=True)
#print(embeddings.shape)
#
#np.save("raw_data/symbol_embeddings.npy", embeddings)
#print("saved")

def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.lower().split()])


def match_dream(dream_text, top_k=5): # 5 best matches
    dream_text = lemmatize(dream_text)
    df = pd.read_csv("Notebooks/raw_data/dreams_interpretations.csv")
    embeddings = np.load("Notebooks/raw_data/symbol_embeddings.npy")

    query_embedding = model.encode([dream_text])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Only include results with a similarity score above a certain threshold
            results.append({
                "Dream Symbol": df.iloc[idx]["Dream Symbol"],
                "Interpretation": df.iloc[idx]["Interpretation"],
                "score": round(float(similarities[idx]), 4)
                })
    return results


if __name__ == "__main__":
    dream =  "i dreamed about a birds and horses"
    results = match_dream(dream)

    for r in results:
        print(f"\n{r['Dream Symbol']} (score: {r['score']})")
        print(r['Interpretation'])
