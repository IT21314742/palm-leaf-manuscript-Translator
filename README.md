# PalmLeaf-ML-Translator

A machine learning-based application that scans and translates ancient **palm leaf manuscripts** written in **Pāli**, **Classical Sinhala**, and **Sanskrit** into **English** and **Modern Sinhala**. The system also provides an option to download the translated content as a **PDF file**, helping preserve and modernize ancient knowledge.

---

## 🌟 Features

- 📷 **Image Preprocessing**: Enhances palm leaf manuscript images using denoising, contrast adjustment, and segmentation.
- 🔤 **OCR Engine**: Extracts text from ancient scripts using a trained OCR model.
- 🌐 **Multilingual Translation**:
  - Input Languages: Pāli, Sanskrit, Classical Sinhala
  - Output Languages: English, Modern Sinhala
- 📄 **PDF Generation**: Save scanned and translated content as a formatted PDF file.
- 💡 **Cultural Preservation**: Bridges ancient languages with modern accessibility for research and learning.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**:
  - OpenCV (Image processing)
  - Tesseract OCR or Custom OCR
  - TensorFlow / Keras (ML Models)
  - MarianMT or mBART (Translation)
  - ReportLab / FPDF (PDF generation)

---

## 📂 Project Structure

PalmLeaf-ML-Translator/
├── images/ # Sample palm leaf manuscripts
├── models/ # Trained OCR and translation models
├── src/
│ ├── preprocessing.py
│ ├── ocr.py
│ ├── translator.py
│ ├── pdf_generator.py
│ └── app.py
├── requirements.txt
├── README.md
└── LICENSE

### Prerequisites

- Python 3.8+
- pip
- Tesseract installed locally (if used)

### Installation

```bash
git clone https://github.com/IT21314742/palm-leaf-manuscript-Translator.git
cd palm-leaf-manuscript-Translator
pip install -r requirements.txt

