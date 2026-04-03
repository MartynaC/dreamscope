# Imports
from transformers import pipeline

# Initializing model
def initialize_model():
    emotion_classifier = pipeline(
        'text-classification',
        model='SamLowe/roberta-base-go_emotions',
        top_k=4
    )
    return emotion_classifier

# Classifying emotions
def classify(classifier, dream_text):
    response = classifier(dream_text)
    return response
