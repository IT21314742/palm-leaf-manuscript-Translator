import streamlit as st
import os
import json
from PIL import Image

def main():
    st.set_page_config(page_title="Palm Leaf Manuscript Labeling Tool", layout="centered")
    st.title("Palm Leaf Manuscript Labeling Tool")
    st.write("Label each image with the correct Sinhala text for OCR training.")

    uploaded_files = st.file_uploader("Upload manuscript images to label", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    label_data = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=uploaded_file.name, width=400)
            label = st.text_input(f"Enter Sinhala text for {uploaded_file.name}", key=uploaded_file.name)
            if label:
                label_data[uploaded_file.name] = label
        if st.button("Export Labels"):
            if label_data:
                label_json = json.dumps(label_data, ensure_ascii=False, indent=2)
                st.download_button("Download Label JSON", label_json, file_name="labels.json", mime="application/json")
            else:
                st.warning("No labels entered yet.")

if __name__ == "__main__":
    main()
