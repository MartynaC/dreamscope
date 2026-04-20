from fastapi import FastAPI
from dreamscope_backend.dreamscope import match_dream, match_emotions
from dreamscope_backend.clip_matcher import match_images_clip

app = FastAPI()

@app.get("/")
def root():
    return {"welcome": "Welcome to Dreamscope, the app of your dreams... litterally!"}

@app.get("/interpretations")
def interpretations(dream_text):
    emotions = match_emotions(dream_text)
    descriptions = match_dream(dream_text)
    return {"emotions": emotions, "descriptions": descriptions}

@app.get("/images")
def images(dream_text):
    image_urls = match_images_clip(dream_text, n=3, use_gcs=True)
    return {"images": image_urls}