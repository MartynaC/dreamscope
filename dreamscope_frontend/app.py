import streamlit as st
import requests

API_URL = "https://dreamscope-api-356964226060.europe-west1.run.app/"

st.set_page_config(page_title="DreamScope", page_icon="🌙")
st.title("🌙 DreamScope")
st.subheader("Dream interpretation")

dream_input = st.text_area("Describe your dream", placeholder="Describe your dream in as much detail as you can remember...")

# # Cache the model loading to improve performance
# @st.cache_resource
# def load_model():
#     return initialize_model()
# classifier = load_model()

url = f"{API_URL}/interpretations"
params= {
    'dream_text': dream_input
}

if st.button("Interpret my dream"):
    if dream_input:
        with st.spinner("Analysing your dream..."):
            # emotions = classify(classifier, dream_input)[0] # emotion_classification
            # results = match_dream(dream_input) # dream_matcher
            results = requests.get(url, params=params).json()

        st.subheader("Emotions detected")
        for emotion in results['emotions']:
            st.write(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

        st.subheader("Symbol interpretations")
        for r in results['descriptions']:
            st.write(f"**{r['Dream Symbol']}** (score: {r['score']})")
            st.write(str(r['Interpretation']))
            st.divider()
    else:
        st.warning("Please describe your dream first.")
#
