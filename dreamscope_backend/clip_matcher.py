import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

IMAGE_DIR = "dreamscope_backend/data/abstract_art_512"
EMBEDDINGS_PATH = "dreamscope_backend/data/clip_embeddings.npy"
FILENAMES_PATH = "dreamscope_backend/data/clip_filenames.npy"

def build_clip_index():
    filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:500]
    embeddings = []

    for i, fname in enumerate(filenames):
        path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output  # (1, 768)
            embedding = model.visual_projection(pooled)  # (1, 512) ← projected
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        embeddings.append(embedding[0].numpy())

        if i % 100 == 0:
            print(f"{i}/{len(filenames)} done")

    np.save(EMBEDDINGS_PATH, np.array(embeddings))
    np.save(FILENAMES_PATH, np.array(filenames))
    print(f"saved {len(filenames)} image embeddings")


def match_images_clip(dream_text, n=3):
    embeddings = np.load(EMBEDDINGS_PATH)
    filenames = np.load(FILENAMES_PATH)

    inputs = processor(text=[dream_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        pooled = text_outputs.pooler_output  # (1, 512)
        text_embedding = model.text_projection(pooled)  # (1, 512) ← projected
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)
    text_embedding = text_embedding[0].numpy()

    similarities = np.dot(embeddings, text_embedding)
    top_indices = np.argsort(similarities)[-n:][::-1]
    return [filenames[i] for i in top_indices]


if __name__ == "__main__":
    build_clip_index()  # run once, then comment out
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