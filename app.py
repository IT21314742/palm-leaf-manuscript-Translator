import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from pipeline import preprocess_image, segment_lines, recognize_characters, translate_to_modern_sinhala, export_to_pdf

st.set_page_config(page_title="Palm Leaf Manuscript OCR", layout="centered")
st.title("Palm Leaf Manuscript OCR & PDF Export")
st.write("Upload manuscript images (historic Sinhala), recognize text, and export as a structured PDF.")

uploaded_files = st.file_uploader("Upload manuscript images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    img_paths = []
    for uploaded_file in uploaded_files:
        img_path = os.path.join(temp_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_paths.append(img_path)
    st.success(f"{len(img_paths)} images uploaded.")
    if st.button("Process Images"):
        all_lines = []
        for img_path in img_paths:
            img = preprocess_image(img_path)
            lines = segment_lines(img)
            st.info(f"{os.path.basename(img_path)}: {len(lines)} lines segmented.")
            recognized = recognize_characters(lines)
            st.write(f"Raw OCR output for {os.path.basename(img_path)}:")
            for ocr_line in recognized:
                st.code(ocr_line)
            translated = translate_to_modern_sinhala(recognized)
            all_lines.extend(translated)
        st.subheader("Translated Preview")
        if all_lines:
            for line in all_lines:
                st.write(line)
        else:
            st.warning("No text was recognized. Check if the OCR model is working or if segmentation is correct.")
        st.success("Processing complete! Preview above.")
