from fastapi import FastAPI

from dreamscope_backend.dreamscope import match_dream

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"welcome": "Welcome to Dreamscope, the app of your dreams... litterally!"}

# Prediction endpoint
@app.get("/guess")
def guess(dream_text):
    best_guess = match_dream(dream_text)
    return {"guess": best_guess}
