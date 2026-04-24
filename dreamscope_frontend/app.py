import streamlit as st
import requests

from emotion_waves import plot_emotion_waves
from concurrent.futures import ThreadPoolExecutor

### LOCAL - Switch to 'USE_LOCAL = True' to make local tests without API ###
USE_LOCAL = False

if USE_LOCAL:
    from dreamscope_backend.dreamscope import match_dream_symbols, match_dream, match_emotions
    from dreamscope_backend.clip_matcher import match_images_clip

### Helper functions to handle Local vs API mode, and protect from API failures ###
def safe_api_request(url, params=None):
    """Makes a GET request and returns (data, error) tuple."""
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        error_msg = f'''
        Request to {url} failed. Exception message:\n
        ```{str(e)}```\n
        '''
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"Status: {e.response.status_code} | Response: {e.response.text}"
        return None, error_msg

def fetch_dream_data(dream_input, api_url=None, params=None):
    """
    Returns (inter_data, inter_error, rag_data, rag_error, img_data, img_error).
    Switches between local functions and API calls based on USE_LOCAL.
    """
    if USE_LOCAL:
        try:
            emotions = match_emotions(dream_input)
            results_mvp = match_dream_symbols(dream_input)
            results = match_dream(dream_input)
            image_urls = match_images_clip(dream_input, n=4)
            return {"emotions": emotions, "descriptions": results_mvp}, None, {"emotions": emotions, "rag": results}, None, {"images": image_urls}, None
        except Exception as e:
            error_msg = f'''
            Local processing failed:\n
            ```{str(e)}```
            '''
            return None, error_msg, None, error_msg
    else:
        with ThreadPoolExecutor() as executor:
            future_inter = executor.submit(safe_api_request, f"{api_url}interpretations", params)
            future_rag = executor.submit(safe_api_request, f"{api_url}rag", params)
            future_img = executor.submit(safe_api_request, f"{api_url}images", params)
            inter_data, inter_error = future_inter.result()
            rag_data, rag_error = future_rag.result()
            img_data, img_error = future_img.result()
        return inter_data, inter_error, rag_data, rag_error, img_data, img_error

API_URL = "https://dreamscope-api-356964226060.europe-west1.run.app/"

st.set_page_config(page_title="DreamScope", page_icon="🌙")

is_dark_mode = st.context.theme.type == "dark"



tab_options = ["🌙 MVP", "✨ Extended", "🧪 Visualization Lab", "🔮 RAG"]
tab_slugs = {
    "mvp": "🌙 MVP",
    "extended": "✨ Extended",
    "viz": "🧪 Visualization Lab",
    "rag": "🔮 RAG",
}
slug_map = {v: k for k, v in tab_slugs.items()}

slug = st.query_params.get("tab", "mvp")
tab = tab_slugs.get(slug, "🌙 MVP")

st.markdown("""
<style>
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.35rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: #cfcfd6;
    font-weight: 700;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.08);
    color: #ffffff;
    border-color: rgba(255,255,255,0.16);
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #c8a2ff, #ff9acb, #ffe0b3);
    color: #111111;
    border: none;
    box-shadow: 0 0 22px rgba(255,154,203,0.28);
}
</style>
""", unsafe_allow_html=True)

for tab_name in tab_options:
    slug_val = slug_map[tab_name]
    is_active = tab_name == tab

    if st.sidebar.button(
        tab_name,
        key=f"nav_{slug_val}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        st.query_params["tab"] = slug_val
        st.rerun()



st.markdown("""
<style>
.gradient-text {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    text-shadow: 0 0 20px rgba(255, 180, 220, 0.4);
    background: linear-gradient(
        270deg,
        #c8a2ff,  /* lavender */
        #ff9acb,  /* pink */
        #ffe0b3,  /* peach */
        #c8a2ff
    );

    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: gradientMove 8s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>

<h1 class="gradient-text">🌙 DREAMSCOPE</h1>
""", unsafe_allow_html=True)
st.subheader("Dream interpretation")

dream_input = st.text_area("Describe your dream", placeholder="Describe your dream in as much detail as you can remember...")
params = {'dream_text': dream_input}


if tab == "🌙 MVP":

    if st.button("Interpret my dream"):
        if dream_input:
            with st.spinner("Analysing your dream..."):
                inter_data, inter_error, rag_data, rag_error, img_data, img_error = fetch_dream_data(
                    dream_input, api_url=API_URL, params=params
                )

            st.subheader("Emotions detected")
            emotions = inter_data.get('emotions')
            for emotion in emotions:
                st.write(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

            st.subheader("Symbol interpretations")
            results = inter_data.get('descriptions')
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
                inter_data, inter_error, rag_data, rag_error, img_data, img_error = fetch_dream_data(
                    dream_input, api_url=API_URL, params=params
                )

            st.subheader("Emotions detected")
            emotions = inter_data.get('emotions')
            for emotion in emotions:
                st.write(f"**{emotion['label']}** — {round(emotion['score'] * 100)}%")

            st.subheader("Dream images — CLIP matched to dream description")
            image_urls = img_data.get('images', [])
            cols = st.columns(3)
            for col, img in zip(cols, image_urls):
                col.image(img['url'], width='stretch')

            st.subheader("Symbol interpretations")
            results = inter_data.get('descriptions')
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
                inter_data, inter_error, rag_data, rag_error, img_data, img_error = fetch_dream_data(
                    dream_input, api_url=API_URL, params=params
                )

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
            emotions = inter_data.get('emotions')
            fig = plot_emotion_waves(emotions, is_dark_mode=is_dark_mode)
            st.pyplot(fig, width='content')

            st.subheader("Dream images — CLIP matched to dream description")
            images = img_data.get('images', [])
            if images:
                for i in range(0, len(images), 2):
                    row = images[i:i+2]
                    cols = st.columns(2, gap='medium')
                    for col, img in zip(cols, row):
                        col.image(img['url'], width='content')
                        col.caption(f"**{img['artist']}** \n*{img['title']}*")

            st.subheader("Symbol interpretations")
            results = inter_data.get('descriptions')
            for r in results:
                st.write(f"**{r['Dream Symbol']}** (score: {r['score']})")
                st.caption(r.get('Context', ''))
                st.write(str(r['Interpretation']))
                st.divider()
        else:
            st.warning("Please describe your dream first.")


elif tab == "🔮 RAG":

    if st.button("Interpret my dream"):
        if not dream_input:
            st.warning("Please describe your dream first.")
            st.stop()

        with st.spinner("Analysing your dream... (this may take a minute)"):
            inter_data, inter_error, rag_data, rag_error, img_data, img_error = fetch_dream_data(
                dream_input, api_url=API_URL, params=params
            )

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
        if rag_error:
            st.write("_Looks like we encoutered a problem... Here are some technical details:_")
            st.error(rag_error)
        else:
            emotions = rag_data.get('emotions')
            if emotions:
                fig = plot_emotion_waves(emotions, is_dark_mode=is_dark_mode)
                st.pyplot(fig, width='content')
            else:
                st.warning("No emotion data found in response.")

        st.subheader("Dream images — CLIP matched to dream description")
        if img_error:
            st.write("_Looks like we encoutered a problem... Here are some technical details:_")
            st.error(img_error)
        else:
            images = img_data.get('images', [])
            if images:
                for i in range(0, len(images), 2):
                    row = images[i:i+2]
                    cols = st.columns(2, gap='medium')
                    for col, img in zip(cols, row):
                        col.image(img['url'], width='content')
                        col.caption(f"**{img['artist']}** \n*{img['title']}*")
            else:
                st.warning("No images returned.")


        st.subheader("Dream interpretation")
        if rag_error:
            st.write("_Looks like we encoutered a problem... Here are some technical details:_")
            st.error(rag_error)
        else:
            results = rag_data.get('rag')
            if results:
                st.write(results)
            else:
                st.warning("No interpretation available.")
