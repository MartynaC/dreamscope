import numpy as np
import pandas as pd
import chromadb
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# def save_embeddings():

#     # Run this code once when dataset is updated or when model has changed
#     # to create the embeddings for the dream symbols and interpretations,
#     # then comment it out to avoid re-running it every time.

#     model = SentenceTransformer("all-mpnet-base-v2")

#     df = pd.read_csv("dreamscope_backend/data/dream_symbols_clean_v5.csv")
#     texts = (df["symbol_clean"] + " " + df["context_clean"]).tolist()

#     embeddings = model.encode(texts, show_progress_bar=True)
#     print(embeddings.shape)

#     np.save("dreamscope_backend/data/symbol_embeddings.npy", embeddings)
#     print("saved")

def create_vector_store():
    # Run this code once when dataset is updated or when model has changed
    # to prepare the data for the dream symbols and interpretations
    # and embed it in a vector store, then comment it out to avoid re-running
    # it every time.

    # Load data
    data = pd.read_csv("data/dream_symbols_clean_v5.csv")

    # Replace NaNs in context_clean with the value of the slug column
    data.loc[data['context_clean'].isnull(), 'context_clean'] = \
        data.loc[data['context_clean'].isnull(), 'slug']

    data = data[data['context_clean'] != data['meaning_clean']].reset_index(drop=True)

    # Initialize sentence transformer and define context embeddings
    sentence_transformer_model = SentenceTransformer(
        "all-mpnet-base-v2",
        device = 'cpu'
        )
    embeddings_context_clean = sentence_transformer_model.encode(
        data['context_clean'].to_list(),
        show_progress_bar=True
        )

    # Set up the vector store with medadata
    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection_context_metadata = client.get_or_create_collection("dream_symbols_metadata")
    metadatas = data[['slug', 'meaning_clean']].to_dict(orient='records')

    collection_context_metadata.add(
        documents=data['context_clean'].tolist()[:5000],
        embeddings=embeddings_context_clean[:5000],
        metadatas=metadatas[:5000],
        ids=[str(i) for i in range(5000)]
    )

    collection_context_metadata.add(
        documents=data['context_clean'].tolist()[5000:],
        embeddings=embeddings_context_clean[5000:],
        metadatas=metadatas[5000:],
        ids=[str(i) for i in range(5000, len(data))]
    )

if __name__ == "__main__":
    create_vector_store()
