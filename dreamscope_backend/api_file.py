from fastapi import FastAPI

from dreamscope_backend.dreamscope import match_dream, match_emotions

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"welcome": "Welcome to Dreamscope, the app of your dreams... litterally!"}

# 'interpretations' endpoint
@app.get("/interpretations")
def interpretations(dream_text):
    emotions = match_emotions(dream_text)
    descriptions = match_dream(dream_text)
    return {"emotions": emotions, "descriptions": descriptions}

# # 'emotions' endpoint
# @app.get("/emotions")
# def emotions(dream_text):
#     emotions = match_emotions(dream_text)
#     return {"emotions": emotions}
