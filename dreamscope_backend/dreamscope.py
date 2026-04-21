import numpy as np
import pandas as pd

import chromadb

from sentence_transformers import SentenceTransformer
from transformers import pipeline

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
nltk.download('wordnet')

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

chroma_client = chromadb.PersistentClient(path=f'{BASE_DIR}/data/chroma_db')

#from dreamscope_backend.preprocess import *

lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.lower().split()])
model = SentenceTransformer("all-mpnet-base-v2")

def match_dream_symbols(dream_text, top_k=5):
    dream_text = lemmatize(dream_text)
    df = pd.read_csv(f"{BASE_DIR}/data/dream_symbols_clean_v5.csv")
    embeddings = np.load(f"{BASE_DIR}/data/symbol_embeddings.npy")
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

# Initialize sentence transformer and define context embeddings
sentence_transformer_model = SentenceTransformer(
    "all-mpnet-base-v2",
    )



def match_dream(dream_text, top_k=5):

    # Divide into sentences
    sentences = sent_tokenize(dream_text)

    # Embed dream sentences
    searched_vectors = sentence_transformer_model.encode(sentences)

    # Query the vector dB
    # client = chromadb.PersistentClient(path="./data/chroma_db")


    collection_context_metadata = chroma_client.get_collection("dream_symbols_metadata")

    result_metadata = collection_context_metadata.query(
        query_embeddings=searched_vectors,
        n_results=6
        )
    print ("✅ retrieved raw symbols...")

    # Generate context - interpretation tuples and removing duplicates
    flat_results = [
        (context, interpretation['meaning_clean'])
        for contexts, interpretations in zip(result_metadata['documents'], result_metadata['metadatas'])
        for context, interpretation in zip(contexts, interpretations)
        ]
    flat_results_unique = set(flat_results)

    # Rerank results
    # Initialize reranker
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # Define symbol context-dream text pairs
    pairs = [(dream_text, symbol[0]) for symbol in flat_results]

    # Rerank according to results relevance
    scores = reranker.predict(pairs)

    # Trier en gardant les métadonnées
    ranked = sorted(zip(flat_results, scores), key=lambda x: x[1], reverse=True)
    print ("✅ reranked symbols...")

    # Extract list of ranked interpretations
    interpretations = [symbol[0][0] + ' ' + symbol[0][1] for symbol in ranked]

    # Initialize response Model
    pipe = pipeline(
        "text-generation",
        model='microsoft/Phi-3-mini-4k-instruct',
        device='cpu'
        )
    print ("✅ initialized LLM...")

    # Define prompt template
    prompt = f'You are interpreting a dream submitted by the user. \
        The original dream text is :{dream_text}. \
        The main interpretations for the dream are {interpretations[:top_k]}. \
        You give a summary of these interpretations, \
        strictly using these interpretations only and not adding any new idea, \
        in less than 100 words.'

    messages = [
            {'role': 'system', 'content': prompt},
        ]

    # Query model
    output = pipe(messages)
    return output[0]['generated_text'][-1]['content']


def match_emotions(dream_text):
    def initialize_model():
        return pipeline(
            'text-classification',
            model='SamLowe/roberta-base-go_emotions',
            top_k=None  # ← get ALL emotions, not just top 4
        )

    classifier = initialize_model()
    output = classifier(dream_text)[0]
    output_df = pd.DataFrame(output)
    output_df = output_df[output_df['label'] != 'neutral'].reset_index(drop=True)
    total = sum(output_df['score'])
    output_df['score'] = output_df['score'] / total
    output_df

    # retrieve file with emotion colors
    df = pd.read_csv("dreamscope_backend/data/goemotions_unique_colors.csv")

    emotions_df = output_df.merge(
        df, how='left', on='label'
        )[['label','score','HEX','RGB']].head(4)

    emotions = [
        {"label": row['label'], "score": row['score'], "RGB": eval(row['RGB'])}
        for _, row in emotions_df.iterrows()
    ]

    return emotions


if __name__ == "__main__":
    dream = "i dreamed about birds and horses"
    results = match_dream(dream)
    emotions = match_emotions(dream)

    print(results)
    print(emotions)
