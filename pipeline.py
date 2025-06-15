import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List
from PIL import Image

# --- Preprocessing ---
def preprocess_image(image_path: str) -> np.ndarray:
    """Apply denoising, adaptive thresholding, and basic enhancement."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    img = cv2.equalizeHist(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 11)
    return img

def save_preprocessed(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(input_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        img = preprocess_image(img_path)
        cv2.imwrite(out_path, img)

# --- Segmentation (placeholder) ---
def segment_lines(img: np.ndarray) -> List[np.ndarray]:
    """Segment the image into lines (simple projection-based, placeholder)."""
    # TODO: Implement robust line segmentation
    return [img]

# --- OCR (integrated) ---
def recognize_characters(line_imgs: List[np.ndarray]) -> List[str]:
    """Run OCR on segmented lines using custom model."""
    from ocr_model import ocr_infer
    return [ocr_infer(line_img) for line_img in line_imgs]

# --- Translation (placeholder) ---
def translate_to_modern_sinhala(text_lines: List[str]) -> List[str]:
    """Translate historic script to modern Sinhala. Placeholder."""
    # TODO: Implement mapping or translation model
    return text_lines

# --- PDF Generation ---
def export_to_pdf(text_lines: List[str], output_pdf: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    y = height - 40
    for line in text_lines:
        c.drawString(40, y, line)
        y -= 20
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()

# --- Pipeline Entrypoint ---
def process_manuscript_folder(input_dir: str, output_pdf: str, temp_dir: str = "preprocessed"):
    save_preprocessed(input_dir, temp_dir)
    all_lines = []
    for fname in sorted(os.listdir(temp_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(os.path.join(temp_dir, fname), cv2.IMREAD_GRAYSCALE)
        line_imgs = segment_lines(img)
        recognized = recognize_characters(line_imgs)
        translated = translate_to_modern_sinhala(recognized)
        all_lines.extend(translated)
    export_to_pdf(all_lines, output_pdf)

if __name__ == "__main__":
    # Example usage: process training Sinhala set
    process_manuscript_folder(
        input_dir="dataset/training_dataset/sinhala",
        output_pdf="output_sinhala_training.pdf",
        temp_dir="preprocessed_sinhala"
    )
