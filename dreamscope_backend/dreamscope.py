import pandas as pd

import chromadb

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from nltk.tokenize import sent_tokenize

from google import genai

from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

# Initialize Gemini Client
gem_client = genai.Client(api_key=os.getenv("API_KEY"))


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
    client = chromadb.PersistentClient(path=f'{BASE_DIR}/data/chroma_db')


    collection_context_metadata = client.get_collection("dream_symbols_metadata")

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

    # Rerank results
    # Initialize reranker
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # Define symbol context-dream text pairs
    pairs = [(dream_text, symbol[0]) for symbol in flat_results]

    # Rerank according to results relevance
    scores = reranker.predict(pairs)

    # Sort and keep metadata
    ranked = sorted(zip(flat_results, scores), key=lambda x: x[1], reverse=True)
    print ("✅ reranked symbols...")

    # Extract list of ranked interpretations
    interpretations = [symbol[0][0] + ' ' + symbol[0][1] for symbol in ranked]

    print(interpretations[:5])

    # Define prompt template
    prompt = f'You are interpreting a dream submitted by the user. \
        The original dream text is :{dream_text}. \
        The main interpretations for the dream are {interpretations[:top_k]}. \
        You give a summary of these interpretations, \
        strictly using these interpretations only and not adding any new idea, \
        in less than 100 words.'

    # Query Google API
    response = gem_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
        )
    return response.text


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
