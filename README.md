# 🩺 MediScan

MediScan is a decision-support prototype that predicts likely medical conditions from user-described symptoms. It accepts symptoms as **plain text**, a **scanned image**, or a **PDF medical report**, and returns the top predicted conditions with confidence scores — all through a simple Flask web interface.

> ⚠️ **Disclaimer:** MediScan is an educational/research prototype only. It is **not** a medical device and does **not** provide clinical diagnoses. Always consult a licensed healthcare professional for medical advice.

---

## How It Works

1. **Input** — Users can submit symptoms in three ways:
   - Typing them directly as text
   - Uploading a scanned image (PNG, JPG, JPEG, TIFF, BMP, WEBP)
   - Uploading a PDF medical report

2. **Text Extraction**
   - Native PDFs are parsed directly with `pdfplumber`.
   - Scanned PDFs and images are processed with **Tesseract OCR** to extract readable text.

3. **NLP Preprocessing**
   - Text is lowercased, stripped of punctuation/numbers, cleaned of stopwords, and stemmed using **NLTK**.

4. **Prediction**
   - A **TF-IDF + Logistic Regression** pipeline (scikit-learn) classifies the cleaned symptom text.
   - The model is trained on a built-in dataset covering 20 common conditions, including Influenza, COVID-19, Pneumonia, Tuberculosis, Malaria, Dengue, Typhoid, Diabetes, Hypertension, Heart Disease, Stroke, Asthma, GERD, Appendicitis, UTI, Anemia, Migraine, Depression, Anxiety Disorder, and Chickenpox.

5. **Output**
   - Returns the **top 3 predicted conditions** with confidence percentages, along with a short excerpt of the extracted symptom text.

---

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (TF-IDF + Logistic Regression), NLTK
- **OCR / Document Parsing:** Tesseract OCR, pdfplumber, pdf2image
- **Frontend:** HTML, CSS, JavaScript

---

## Project Structure

MediScan/
├── app.py # Flask web server & API routes
├── disease_detector.py # Core ML pipeline: training, prediction, text/OCR extraction
├── disease_model.pkl # Trained classification model
├── label_encoder.pkl # Label encoder for disease classes
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Web UI
└── .gitignore

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system (required for image/scanned-PDF input)

### Installation

```bash
git clone https://github.com/MohidWebDev/MediScan.git
cd MediScan
pip install -r requirements.txt
```

### Running the Web App

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### Using the CLI

```bash
# Predict from plain text
python disease_detector.py --text "fever, headache, body ache, fatigue"

# Predict from a PDF report
python disease_detector.py --pdf report.pdf

# Predict from an image
python disease_detector.py --image scan.png

# Retrain the model
python disease_detector.py --train
```

---

## API Endpoints

| Method | Endpoint        | Description                                 |
| ------ | --------------- | ------------------------------------------- |
| GET    | `/`             | Serves the web UI                           |
| POST   | `/predict/text` | Predicts from JSON body `{ "text": "..." }` |
| POST   | `/predict/file` | Predicts from an uploaded image or PDF file |

---

## License

MIT
