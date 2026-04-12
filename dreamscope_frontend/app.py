import streamlit as st
import requests
from dreamscope_backend.dreamscope import match_dream, match_emotions
from dreamscope_backend.color_matcher import match_images_by_emotion
from dreamscope_backend.clip_matcher import match_images_clip

API_URL = "https://dreamscope-api-356964226060.europe-west1.run.app/"

st.set_page_config(page_title="DreamScope", page_icon="🌙")

tab = st.sidebar.radio("Navigation", ["🌙 MVP", "✨ Extended"], label_visibility="collapsed")

st.title("🌙 DreamScope")
st.subheader("Dream interpretation")

dream_input = st.text_area("Describe your dream", placeholder="Describe your dream in as much detail as you can remember...")

url = f"{API_URL}/interpretations"
params = {'dream_text': dream_input}

if tab == "🌙 MVP":
    if st.button("Interpret my dream"):
        if dream_input:
            with st.spinner("Analysing your dream..."):
                # LOCAL - comment out when using API
                emotions = match_emotions(dream_input)
                results = match_dream(dream_input)

                # API - comment out when working locally
                # response = requests.get(url, params=params).json()
                # emotions = response['emotions']
                # results = response['descriptions']

            st.subheader("Emotions detected")
            for emotion in emotions:
                st.write(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

            st.subheader("Symbol interpretations")
            for r in results:
                st.write(f"**{r['Dream Symbol']}** (score: {r['score']})")
                st.caption(r.get('Context', ''))
                st.write(str(r['Interpretation']))
                st.divider()
        else:
            st.warning("Please describe your dream first.")

elif tab == "✨ Extended":
    if st.button("Interpret my dream"):
        if dream_input:
            with st.spinner("Analysing your dream..."):
                emotions = match_emotions(dream_input)
                results = match_dream(dream_input)

            st.subheader("Emotions detected")
            for emotion in emotions:
                st.write(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

            st.subheader("Dream images — color matched to emotion")
            top_emotion = emotions[0]['label']
            matched_images = match_images_by_emotion(top_emotion, n=3)
            cols = st.columns(3)
            for col, img_name in zip(cols, matched_images):
                img_path = f"dreamscope_backend/data/abstract_art_512/{img_name}"
                col.image(img_path, use_container_width=True, caption=img_name)

            st.subheader("Dream images — CLIP matched to dream description")
            matched_images = match_images_clip(dream_input, n=3)
            cols = st.columns(3)
            for col, img_name in zip(cols, matched_images):
                img_path = f"dreamscope_backend/data/abstract_art_512/{img_name}"
                col.image(img_path, use_container_width=True)

            st.subheader("Symbol interpretations")
            for r in results:
                st.write(f"**{r['Dream Symbol']}** (score: {r['score']})")
                st.caption(r.get('Context', ''))
                st.write(str(r['Interpretation']))
                st.divider()
        else:
            st.warning("Please describe your dream first.")