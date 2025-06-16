import streamlit as st
from utils import extract_features, predict_accent, load_model
import tempfile
import os

st.title("Accent Classifier AI Agent")
st.write("Upload an audio file and get the predicted accent.")
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
model = load_model()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix = ".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name
        st.audio(uploaded_file, format='audio/wav')
        if st.button("predict Accent"):
            try:
                prediction = predict_accent(file_path, model)
                st.success(f"🗣️Predicted Accent: **{prediction}**")
            except Exception as e:
                st.error(f"Error: {e}")



