import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import tempfile
# from google.cloud import storage
import urllib.request

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Local paths (for building index only)
LOCAL_IMAGE_DIR = "dreamscope_backend/data/abstract_art_512"
LOCAL_EMBEDDINGS_PATH = "dreamscope_backend/data/clip_embeddings.npy"
LOCAL_FILENAMES_PATH = "dreamscope_backend/data/clip_filenames.npy"

# GCS paths (for production)
BUCKET_NAME = "dreamscope-images"
EMBEDDINGS_BLOB = "clip_embeddings.npy"
FILENAMES_BLOB = "clip_filenames.npy"
GCS_IMAGE_DIR = "abstract_art_512"


def build_clip_index():
    filenames = [f for f in os.listdir(LOCAL_IMAGE_DIR) if f.endswith('.jpg')]
    embeddings = []

    for i, fname in enumerate(filenames):
        path = os.path.join(LOCAL_IMAGE_DIR, fname)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output
            embedding = model.visual_projection(pooled)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        embeddings.append(embedding[0].numpy())

        if i % 100 == 0:
            print(f"{i}/{len(filenames)} done")

    np.save(LOCAL_EMBEDDINGS_PATH, np.array(embeddings))
    np.save(LOCAL_FILENAMES_PATH, np.array(filenames))
    print(f"saved {len(filenames)} image embeddings")


# def load_from_gcs(blob_name):
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(blob_name)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
#         blob.download_to_filename(tmp.name)
#         return np.load(tmp.name, allow_pickle=True)

def load_from_gcs(blob_name):
    url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
        urllib.request.urlretrieve(url, tmp.name)
    return np.load(tmp.name, allow_pickle=True)


GCS_BASE_URL = "https://storage.googleapis.com/dreamscope-images/abstract_art_512"

def match_images_clip(dream_text, n=3, use_gcs=False):
    if use_gcs:
        embeddings = load_from_gcs(EMBEDDINGS_BLOB)
        filenames = load_from_gcs(FILENAMES_BLOB)
    else:
        embeddings = np.load(LOCAL_EMBEDDINGS_PATH)
        filenames = np.load(LOCAL_FILENAMES_PATH)

    inputs = processor(text=[dream_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        pooled = text_outputs.pooler_output
        text_embedding = model.text_projection(pooled)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)
    text_embedding = text_embedding[0].numpy()

    similarities = np.dot(embeddings, text_embedding)
    top_indices = np.argsort(similarities)[-n:][::-1]

    # return URLs instead of filenames
    return [f"{GCS_BASE_URL}/{filenames[i]}" for i in top_indices]


if __name__ == "__main__":
    # build_clip_index()  # run once, then comment out

    dreams = [
        "i was flying around the city",
        "there was a baby speaking to me in polish",
        "i was drowning in dark water",
    ]
    for dream in dreams:
        print(f"\n{dream}")
        images = match_images_clip(dream, n=3)
        for img in images:
            print(f"  {img}")
