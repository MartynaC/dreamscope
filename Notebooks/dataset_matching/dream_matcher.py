import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

model = SentenceTransformer("all-mpnet-base-v2")
lemmatizer = WordNetLemmatizer()

#This code is to be run once when dataset is updated or when model has changed
#to create the embeddings for the dream symbols and interpretations,

# data from raw_data folder:
#df = pd.read_csv("raw_data/dream_symbols.csv")
#texts = (df["symbol"] + " " + df["meaning"]).tolist()
# data from Thomas's Notebooks folder:
#df = pd.read_csv("Notebooks/dream_symbols_clean.csv")
#texts = (df["original_label"] + " " + df["interpretation_text_clean"]).tolist()

#embeddings = model.encode(texts, show_progress_bar=True)
#np.save("raw_data/symbol_embeddings.npy", embeddings)
#print("Embeddings created and saved successfully.")

def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.lower().split()])


def match_dream(dream_text, top_k=5): # 5 best matches
    dream_text = lemmatize(dream_text)
    df = pd.read_csv("Notebooks/raw_data/dream_symbols.csv")
    #df = pd.read_csv("Notebooks/dream_notebook/dream_symbols_clean.csv")
    embeddings = np.load("Notebooks/raw_data/symbol_embeddings.npy")

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
    dream =  "i dreamed about a serpent"
    results = match_dream(dream)


    for r in results:
        print(f"\n{r['Dream Symbol']} (score: {r['score']})")
        print(r['Interpretation'])
