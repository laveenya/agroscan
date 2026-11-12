"""AgroScan web page (Streamlit). It only talks to the API, it never loads the model itself."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AgroScan", page_icon="🌿")
st.title("🌿 AgroScan - crop disease detection")
st.write("Upload a close-up photo of **one leaf** (tomato, potato or bell pepper) and AgroScan suggests the most likely disease.")

try:
    info = requests.get(f"{API_URL}/health", timeout=5).json()
    if not info.get("model_loaded"):
        st.error(f"The API is running but the model is not loaded: {info.get('error')}")
        st.stop()
    st.sidebar.success("API connected")
    st.sidebar.write(f"Model version: `{str(info['model_version'])[:8]}`")
    st.sidebar.write(f"'Not sure' below **{info['confidence_threshold']:.0%}** confidence")
    st.sidebar.write("Classes:")
    for name in info["classes"]:
        st.sidebar.write("- " + name.replace("_", " ").strip())
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Is it running?")
    st.stop()

upload = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
if upload is not None:
    st.image(upload, caption="Your photo", use_container_width=True)
    with st.spinner("Analysing..."):
        r = requests.post(f"{API_URL}/predict", files={"file": (upload.name, upload.getvalue(), upload.type)}, timeout=60)
    if r.status_code != 200:
        st.error(r.json().get("detail", "Something went wrong."))
    else:
        res = r.json()
        if res["confident"]:
            st.success(f"Most likely: **{res['prediction'].replace('_', ' ')}**")
        else:
            st.warning(res["message"])
        st.subheader("Top 3 suggestions")
        for item in res["top3"]:
            st.write(f"{item['disease'].replace('_', ' ')} - {item['probability']:.0%}")
            st.progress(min(1.0, item["probability"]))
        st.caption(f"Model {str(res['model_version'])[:8]} - {res['latency_ms']} ms")

st.divider()
st.caption("AgroScan is a student project trained on 8 classes. Real-world accuracy is about 57-62% (top-3 about 93%), so treat the "
           "answer as a suggestion and confirm with an agronomist.")
