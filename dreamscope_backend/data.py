import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def save_embeddings():

    # Run this code once when dataset is updated or when model has changed
    # to create the embeddings for the dream symbols and interpretations,
    # then comment it out to avoid re-running it every time.

    model = SentenceTransformer("all-mpnet-base-v2")

    df = pd.read_csv("data/dream_symbols.csv")

    texts = (df["Dream Symbol"] + " " + df["Interpretation"]).tolist()

    embeddings = model.encode(texts, show_progress_bar=True)
    print(embeddings.shape)

    np.save("raw_data/symbol_embeddings.npy", embeddings)
    print("saved")
