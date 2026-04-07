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

    df = pd.read_csv("dreamscope_backend/data/dream_symbols.csv")
    texts = (df["symbol"] + " " + df["meaning"]).tolist()

    # data from Thomas's Notebooks folder:
    #df = pd.read_csv("dreamscope_backend/data/dream_symbols_clean.csv")
    #texts = (df["original_label"] + " " + df["interpretation_text_clean"]).tolist()

    embeddings = model.encode(texts, show_progress_bar=True)
    print(embeddings.shape)

    np.save("dreamscope_backend/data/symbol_embeddings.npy", embeddings)
    print("saved")


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
