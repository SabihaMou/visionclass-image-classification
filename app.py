"""Streamlit UI for VisionClass."""
import streamlit as st
from PIL import Image, UnidentifiedImageError
from visionclass.predict import classify, prepare_image

st.set_page_config(page_title="VisionClass | Image Classification", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1050px; padding-top: 2.3rem}
.hero {padding: 1.8rem 2rem; background: linear-gradient(110deg,#15253d,#28546b); border-radius: 18px; color: white; margin-bottom: 1.5rem}
.hero h1 {color: white; font-size: 2.3rem; margin-bottom: .3rem}
.hero p {color: #d9edf4; margin: 0}
.small-note {color: #64748b; font-size: .9rem}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔎 VisionClass</h1><p>Image classification with MobileNetV3 and interactive top-five predictions</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("About this project")
    st.write("**Model:** MobileNetV3 Small")
    st.write("**Classification categories:** 1,000 ImageNet classes")
    st.write("**Output:** Top 5 predicted ImageNet labels")
    st.caption("ImageNet categories are limited and may not describe every image accurately.")

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("1 · Upload an image")
    uploaded = st.file_uploader("Choose a JPG, PNG, or WEBP image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        try:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("Please upload an image smaller than 10 MB.")
                st.stop()
            image = prepare_image(Image.open(uploaded))
            st.image(image, caption="Your uploaded image", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("This file could not be opened as an image. Try another file.")
            st.stop()
    else:
        st.info("Upload a photo to begin. For example: a dog, bicycle, or coffee mug.")

with right:
    st.subheader("2 · Model predictions")
    if uploaded is None:
        st.write("Your top predictions will appear here after you upload an image.")
    else:
        if st.button("Classify image", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading model and analyzing image (first run may download weights)..."):
                    results = classify(image)
                st.session_state["results"] = results
                st.session_state["image_key"] = (uploaded.name, uploaded.size)
            except Exception as exc:
                st.error("Classification failed. Check your internet connection on first run and verify PyTorch/TorchVision installation.")
                with st.expander("Technical details"):
                    st.code(str(exc))
        if st.session_state.get("image_key") == (uploaded.name, uploaded.size):
            for index, item in enumerate(st.session_state.get("results", []), 1):
                st.write(f"**{index}. {item['label'].replace('_', ' ').title()}** — {item['confidence']:.1%}")
                st.progress(item["confidence"])
            if st.session_state.get("results"):
                st.caption("Scores are model probabilities over ImageNet labels, not guarantees of correctness.")

st.divider()
st.subheader("How it works")
st.write("**Upload → Preprocess → MobileNetV3 → Softmax → Top five labels.** The model uses official pretrained TorchVision weights; the uploaded image is processed locally by the running app.")
st.caption("Educational demonstration. Do not use predictions for medical, safety-critical, or other high-stakes decisions.")
