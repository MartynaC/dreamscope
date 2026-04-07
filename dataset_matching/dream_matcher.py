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
#df = pd.read_csv("Notebooks/dream_symbols_clean_v2.csv")
#texts = (df["symbol_clean"] + " " + df["context_clean"]).tolist()

#embeddings = model.encode(texts, show_progress_bar=True)
#np.save("raw_data/symbol_embeddings.npy", embeddings)
#print("Embeddings created and saved successfully.")

def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.lower().split()])


def match_dream(dream_text, top_k=5): # 5 best matches
    dream_text = lemmatize(dream_text)
    df = pd.read_csv("Notebooks/dream_symbols_clean_v2.csv")
    #df = pd.read_csv("Notebooks/dream_symbols_clean.csv")
    embeddings = np.load("raw_data/symbol_embeddings.npy")
    
    query_embedding = model.encode([dream_text])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
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
        if symbol_words & dream_words:  # if any symbol word appears in dream text add score boost
            data["score"] = round(data["score"] + 0.05, 6)        
    
    # sort by score and return top_k unique symbols
    results = sorted(best_per_symbol.values(), key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] > 0.3]
    
    return results[:top_k]


if __name__ == "__main__":
    dream =  "i dreamed about a serpent"
    results = match_dream(dream)
    
    
    for r in results:
        print(f"\n{r['Dream Symbol']} (score: {r['score']})")
        print(f"Context: {r['Context']}")
        print(r['Interpretation'])
