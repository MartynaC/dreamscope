import streamlit as st
import requests
from dreamscope_backend.dreamscope import match_dream, match_emotions
from dreamscope_backend.clip_matcher import match_images_clip
from emotion_waves import plot_emotion_waves

API_URL = "https://dreamscope-api-356964226060.europe-west1.run.app/"

st.set_page_config(page_title="DreamScope", page_icon="🌙")

tab = st.sidebar.radio("Navigation", ["🌙 MVP", "✨ Extended", "🧪 Visualization Lab", "🔮 RAG"], label_visibility="collapsed")

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
                # emotions = match_emotions(dream_input)
                # results = match_dream(dream_input)

                # API - comment out when working locally
                response = requests.get(url, params=params).json()
                emotions = response['emotions']
                results = response['descriptions']

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
                # LOCAL - comment out when using API
                # emotions = match_emotions(dream_input)
                # results = match_dream(dream_input)
                # image_urls = match_images_clip(dream_input, n=3)

                # API - comment out when working locally
                response = requests.get(url, params=params).json()
                emotions = response['emotions']
                results = response['descriptions']
                img_response = requests.get(f"{API_URL}/images", params=params).json()
                image_urls = img_response['images']

            st.subheader("Emotions detected")
            for emotion in emotions:
                st.write(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

            st.subheader("Dream images — CLIP matched to dream description")
            cols = st.columns(3)
            for col, img_url in zip(cols, image_urls):
                col.image(img_url, width='stretch')

            st.subheader("Symbol interpretations")
            for r in results:
                st.write(f"**{r['Dream Symbol']}** (score: {r['score']})")
                st.caption(r.get('Context', ''))
                st.write(str(r['Interpretation']))
                st.divider()
        else:
            st.warning("Please describe your dream first.")

elif tab == "🧪 Visualization Lab":
    if st.button("Interpret my dream"):
        if dream_input:
            with st.spinner("Analysing your dream..."):
                # LOCAL - comment out when using API
                # emotions = match_emotions(dream_input)
                # results = match_dream(dream_input)
                # image_urls = match_images_clip(dream_input, n=3)

                # API - comment out when working locally
                response = requests.get(url, params=params).json()
                emotions = response['emotions']
                results = response['descriptions']
                img_response = requests.get(f"{API_URL}/images", params=params).json()
                image_urls = img_response['images']

            st.subheader("Emotions detected")
            with st.container():
                st.markdown(
                    """
                    <style>
                        div[class*="stPlotlyChart"] { height: 600px !important; }
                        figure { height: 600px !important; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
            fig = plot_emotion_waves(emotions)
            st.pyplot(fig, width='stretch')

            st.subheader("Dream images — CLIP matched to dream description")
            cols = st.columns(3)
            for col, img_url in zip(cols, image_urls):
                col.image(img_url, width='stretch')

            st.subheader("Symbol interpretations")
            for r in results:
                st.write(f"**{r['Dream Symbol']}** (score: {r['score']})")
                st.caption(r.get('Context', ''))
                st.write(str(r['Interpretation']))
                st.divider()
        else:
            st.warning("Please describe your dream first.")

elif tab == "🔮 RAG":
    if st.button("Interpret my dream"):
        if dream_input:
            with st.spinner("Analysing your dream... (this may take a minute)"):
                # LOCAL - comment out when using API
                emotions = match_emotions(dream_input)
                results = match_dream(dream_input)
                image_urls = match_images_clip(dream_input, n=3)

                # API - comment out when working locally
                # response = requests.get(url, params=params).json()
                # emotions = response['emotions']
                # results = response['descriptions']
                # img_response = requests.get(f"{API_URL}/images", params=params).json()
                # image_urls = img_response['images']

            st.subheader("Emotions detected")
            with st.container():
                st.markdown(
                    """
                    <style>
                        div[class*="stPlotlyChart"] { height: 600px !important; }
                        figure { height: 600px !important; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
            fig = plot_emotion_waves(emotions)
            st.pyplot(fig, width='stretch')

            st.subheader("Dream images — CLIP matched to dream description")
            cols = st.columns(3)
            for col, img_url in zip(cols, image_urls):
                col.image(img_url, width='stretch')

            st.subheader("Dream interpretation")
            st.write(results)
        else:
            st.warning("Please describe your dream first.")