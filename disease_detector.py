#!/usr/bin/env python3
"""
============================================================
  Disease Detection System — Decision Support Prototype
  Author  : AI-generated prototype (NOT a medical device)
  License : MIT
============================================================

DISCLAIMER
----------
This tool is a research/educational prototype only.
It does NOT replace professional medical advice, diagnosis,
or treatment. Always consult a qualified healthcare provider.

REQUIRED PACKAGES  (install once)
----------------------------------
    pip install scikit-learn numpy pandas joblib pillow \
                pytesseract pdfplumber pdf2image nltk

SYSTEM DEPENDENCY  (for OCR)
-----------------------------
  macOS  : brew install tesseract
  Ubuntu : sudo apt install tesseract-ocr
  Windows: https://github.com/UB-Mannheim/tesseract/wiki
           Then set: TESSDATA_PREFIX / add to PATH

USAGE
-----
  python disease_detector.py --text "fever, headache, body ache, fatigue"
  python disease_detector.py --pdf  report.pdf
  python disease_detector.py --image scan.png
  python disease_detector.py --train          # retrain & save model
"""

# ── Standard library ──────────────────────────────────────
import os
import re
import sys
import json
import argparse
import warnings
import textwrap
from pathlib import Path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Third-party ───────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
except ImportError as e:
    sys.exit(f"[ERROR] Missing package: {e}\n"
             "Run: pip install scikit-learn numpy pandas joblib")

try:
    import nltk
    from nltk.stem import PorterStemmer

    def _ensure_nltk_resource(resource_name: str, resource_path: str):
        """
        Ensure an NLTK resource is present and not corrupted.
        Deletes and re-downloads if the file is a bad zip.
        """
        import zipfile, glob
        try:
            nltk.data.find(resource_path)
            # Also verify the zip is not corrupted
            found_path = nltk.data.find(resource_path)
            # If it's a zip, try opening it
            for nltk_dir in nltk.data.path:
                pattern = os.path.join(nltk_dir, "**", resource_name + ".zip")
                for zf in glob.glob(pattern, recursive=True):
                    try:
                        zipfile.ZipFile(zf).close()
                    except zipfile.BadZipFile:
                        print(f"[INFO] Corrupted NLTK file detected: {zf} — deleting and redownloading…")
                        os.remove(zf)
                        nltk.download(resource_name, quiet=False, force=True)
                        return
        except LookupError:
            nltk.download(resource_name, quiet=False, force=True)

    _ensure_nltk_resource("stopwords", "corpora/stopwords")
    _ensure_nltk_resource("punkt",     "tokenizers/punkt")
    _ensure_nltk_resource("punkt_tab", "tokenizers/punkt_tab")

    from nltk.corpus import stopwords   # import AFTER ensuring resources exist

except ImportError:
    sys.exit("[ERROR] nltk not found. Run: pip install nltk")
except Exception as _nltk_err:
    sys.exit(
        f"[ERROR] NLTK setup failed: {_nltk_err}\n"
        "Fix: delete the NLTK data folder and rerun.\n"
        "  Windows : C:\\Users\\<you>\\AppData\\Roaming\\nltk_data\\\n"
        "  macOS   : ~/nltk_data/\n"
        "  Linux   : ~/nltk_data/\n"
        "Then run the script again and it will redownload cleanly."
    )

try:
    from PIL import Image
    import pytesseract
except ImportError:
    pytesseract = None   # OCR disabled gracefully

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────
MODEL_PATH   = Path("disease_model.pkl")
ENCODER_PATH = Path("label_encoder.pkl")
TOP_K        = 3   # number of top predictions to show

# ─────────────────────────────────────────────────────────
#  BUILT-IN TRAINING DATA
#  (symptom text  →  disease label)
#  Expanded synthetic dataset — covers 20 common conditions.
# ─────────────────────────────────────────────────────────
TRAINING_RECORDS = [
    # ── Influenza ─────────────────────────────────────────
    ("fever chills body ache headache fatigue sore throat runny nose cough", "Influenza"),
    ("high fever muscle pain tiredness sneezing nasal congestion", "Influenza"),
    ("sudden fever severe headache joint pain exhaustion dry cough", "Influenza"),
    ("flu like symptoms fever chills sweating aches weakness", "Influenza"),
    ("seasonal flu fever body aches fatigue loss of appetite", "Influenza"),
    ("fever 102 body pain headache cold sweats chills weakness", "Influenza"),

    # ── Common Cold ───────────────────────────────────────
    ("runny nose sneezing mild sore throat watery eyes", "Common Cold"),
    ("nasal congestion mild cough low grade fever stuffy nose", "Common Cold"),
    ("blocked nose sneezing throat irritation mild headache", "Common Cold"),
    ("cold symptoms runny nose watery eyes sneezing", "Common Cold"),
    ("sore throat runny nose sneezing no fever fatigue", "Common Cold"),

    # ── COVID-19 ──────────────────────────────────────────
    ("loss of smell loss of taste fever dry cough fatigue", "COVID-19"),
    ("shortness of breath fever cough chest pain anosmia", "COVID-19"),
    ("covid symptoms fever cough breathlessness fatigue body pain", "COVID-19"),
    ("anosmia ageusia dry cough fever sore throat diarrhea", "COVID-19"),
    ("persistent dry cough fever loss of smell muscle aches", "COVID-19"),
    ("difficulty breathing high fever loss of taste cough fatigue", "COVID-19"),

    # ── Pneumonia ─────────────────────────────────────────
    ("productive cough chest pain fever chills shortness of breath", "Pneumonia"),
    ("wet cough green sputum high fever chest tightness breathing difficulty", "Pneumonia"),
    ("pneumonia cough blood sputum fever rapid breathing chest pain", "Pneumonia"),
    ("high temperature difficulty breathing crackling sound lungs fatigue", "Pneumonia"),
    ("persistent cough chest pain fever rigor sweating breathless", "Pneumonia"),

    # ── Tuberculosis ──────────────────────────────────────
    ("chronic cough blood sputum night sweats weight loss fatigue", "Tuberculosis"),
    ("hemoptysis prolonged cough fever chest pain weight loss", "Tuberculosis"),
    ("tb cough three weeks blood sputum night sweating loss weight", "Tuberculosis"),
    ("night sweats persistent cough unexplained weight loss low fever", "Tuberculosis"),

    # ── Malaria ───────────────────────────────────────────
    ("cyclic fever chills sweating headache nausea vomiting", "Malaria"),
    ("high fever shivering anemia jaundice splenomegaly headache", "Malaria"),
    ("periodic fever rigor rigidity sweating malaise joint pain", "Malaria"),
    ("malarial fever every 48 hours chills headache body aches", "Malaria"),

    # ── Dengue ────────────────────────────────────────────
    ("dengue high fever severe headache pain behind eyes rash", "Dengue"),
    ("breakbone fever muscle pain joint pain rash low platelet", "Dengue"),
    ("fever rash thrombocytopenia myalgia nausea vomiting headache", "Dengue"),
    ("sudden high fever rash behind eyes pain hemorrhagic rash", "Dengue"),

    # ── Typhoid ───────────────────────────────────────────
    ("prolonged fever abdominal pain rose spots constipation splenomegaly", "Typhoid"),
    ("enteric fever high temperature abdominal tenderness diarrhea", "Typhoid"),
    ("step ladder fever headache slow pulse abdominal discomfort", "Typhoid"),
    ("typhoid fever nausea vomiting diarrhea abdominal cramps", "Typhoid"),

    # ── Diabetes ──────────────────────────────────────────
    ("increased thirst frequent urination fatigue blurred vision", "Diabetes"),
    ("polyuria polydipsia polyphagia weight loss tingling feet", "Diabetes"),
    ("high blood sugar frequent urination slow healing wounds", "Diabetes"),
    ("excessive hunger weight loss fatigue hyperglycemia", "Diabetes"),
    ("numbness hands feet thirst urination blurry vision fatigue", "Diabetes"),

    # ── Hypertension ──────────────────────────────────────
    ("high blood pressure headache dizziness shortness of breath", "Hypertension"),
    ("hypertension headache nosebleed chest pain palpitations", "Hypertension"),
    ("elevated bp vision changes chest discomfort shortness breath", "Hypertension"),
    ("blood pressure 160 100 headache nausea dizziness", "Hypertension"),

    # ── Heart Disease ─────────────────────────────────────
    ("chest pain pressure squeezing shortness of breath left arm pain", "Heart Disease"),
    ("angina chest tightness radiating pain jaw shoulder fatigue", "Heart Disease"),
    ("myocardial infarction crushing chest pain sweating nausea", "Heart Disease"),
    ("palpitations irregular heartbeat chest discomfort breathlessness", "Heart Disease"),
    ("coronary artery disease chest pain exertion shortness breath", "Heart Disease"),

    # ── Stroke ────────────────────────────────────────────
    ("sudden numbness face arm leg confusion slurred speech", "Stroke"),
    ("sudden headache vision loss weakness one side face droop", "Stroke"),
    ("cerebrovascular accident sudden weakness speech difficulty", "Stroke"),
    ("fast stroke face arm speech time facial drooping arm weakness", "Stroke"),

    # ── Asthma ────────────────────────────────────────────
    ("wheezing shortness of breath chest tightness coughing night", "Asthma"),
    ("bronchospasm breathlessness wheezing cough cold air trigger", "Asthma"),
    ("asthma attack dyspnea wheezing chest tightness inhaler", "Asthma"),
    ("difficulty breathing cough nighttime wheeze allergic trigger", "Asthma"),

    # ── GERD / Acid Reflux ────────────────────────────────
    ("heartburn acid reflux regurgitation sour taste chest burning", "GERD"),
    ("gastroesophageal reflux burning chest throat discomfort", "GERD"),
    ("acid indigestion heartburn bloating belching after meals", "GERD"),
    ("burning sensation stomach food coming back throat", "GERD"),

    # ── Appendicitis ──────────────────────────────────────
    ("right lower abdomen pain nausea vomiting fever loss appetite", "Appendicitis"),
    ("sharp pain right iliac fossa guarding rigidity fever", "Appendicitis"),
    ("periumbilical pain migrating right lower quadrant nausea fever", "Appendicitis"),

    # ── UTI ───────────────────────────────────────────────
    ("burning urination frequent urge cloudy urine lower back pain", "UTI"),
    ("dysuria frequent urination pelvic pain dark urine fever", "UTI"),
    ("urinary tract infection pain urination blood urine fever", "UTI"),
    ("burning sensation urination urgency incomplete emptying", "UTI"),

    # ── Anemia ────────────────────────────────────────────
    ("fatigue pale skin dizziness shortness of breath cold hands", "Anemia"),
    ("low hemoglobin weakness tiredness pallor palpitations", "Anemia"),
    ("iron deficiency fatigue brittle nails pale conjunctiva", "Anemia"),
    ("breathlessness on exertion pallor fatigue tachycardia", "Anemia"),

    # ── Migraine ──────────────────────────────────────────
    ("severe throbbing headache nausea vomiting light sensitivity", "Migraine"),
    ("unilateral headache aura photophobia phonophobia", "Migraine"),
    ("migraine pulsating pain nausea sensitivity light sound", "Migraine"),
    ("visual aura blind spot severe one sided headache vomiting", "Migraine"),

    # ── Depression / Anxiety ──────────────────────────────
    ("persistent sadness hopelessness loss of interest fatigue sleep issues", "Depression"),
    ("low mood anhedonia insomnia appetite change concentration poor", "Depression"),
    ("anxiety worry restlessness rapid heartbeat sweating panic", "Anxiety Disorder"),
    ("panic attack chest pain racing heart shortness breath fear", "Anxiety Disorder"),

    # ── Chickenpox ────────────────────────────────────────
    ("itchy red blisters rash fever fatigue chickenpox", "Chickenpox"),
    ("vesicular rash all over body fever itching scabs", "Chickenpox"),
    ("blister rash starting face spreading body mild fever itching", "Chickenpox"),
]

# ─────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────
_stemmer    = PorterStemmer()
_stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """
    Lowercase → remove punctuation/numbers → remove stopwords
    → stem tokens → return cleaned string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens
              if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


def summarise_text(text: str, max_chars: int = 300) -> str:
    """Return a brief excerpt of extracted text for display."""
    text = " ".join(text.split())   # collapse whitespace
    return (text[:max_chars] + "…") if len(text) > max_chars else text

# ─────────────────────────────────────────────────────────
#  BUILD TRAINING DATAFRAME
# ─────────────────────────────────────────────────────────

def build_dataframe() -> pd.DataFrame:
    texts, labels = zip(*TRAINING_RECORDS)
    df = pd.DataFrame({"symptoms": texts, "disease": labels})
    return df


def load_csv_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load an external CSV dataset.
    Expected columns: 'symptoms' and 'disease'.
    Falls back gracefully to the built-in dataset.
    """
    try:
        df = pd.read_csv(csv_path)
        required = {"symptoms", "disease"}
        if not required.issubset(df.columns):
            print(f"[WARN] CSV missing columns {required}. Using built-in data.")
            return build_dataframe()
        print(f"[INFO] Loaded {len(df)} records from {csv_path}")
        return df
    except Exception as ex:
        print(f"[WARN] Could not load CSV ({ex}). Using built-in data.")
        return build_dataframe()

# ─────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────

def train_model(csv_path: str = None) -> tuple:
    """
    Train a TF-IDF + Logistic Regression pipeline.
    Saves model and label encoder to disk.
    Returns (pipeline, label_encoder).
    """
    df = load_csv_dataset(csv_path) if csv_path else build_dataframe()

    df["clean"] = df["symptoms"].apply(preprocess_text)

    le = LabelEncoder()
    y  = le.fit_transform(df["disease"])
    X  = df["clean"]

    n_classes    = len(le.classes_)
    can_stratify = int(len(df) * 0.2) >= n_classes   # test set must fit all classes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
        )),
    ])

    pipeline.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = pipeline.predict(X_test)
        # Only report on classes that actually appear in the test split
        present_labels = np.unique(y_test)
        present_names  = le.classes_[present_labels]
        print("\n── Training complete ──────────────────────────────")
        print(classification_report(
            y_test, y_pred,
            labels=present_labels,
            target_names=present_names,
            zero_division=0,
        ))

    # Persist to disk
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    print(f"[INFO] Encoder saved → {ENCODER_PATH}")

    return pipeline, le


def load_or_train_model() -> tuple:
    """Load saved model if available; otherwise train from scratch."""
    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        try:
            pipeline = joblib.load(MODEL_PATH)
            le       = joblib.load(ENCODER_PATH)
            print("[INFO] Loaded saved model.")
            return pipeline, le
        except Exception as ex:
            print(f"[WARN] Could not load saved model ({ex}). Retraining…")

    print("[INFO] Training model on built-in dataset…")
    return train_model()

# ─────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────

def predict(text: str, pipeline, le, top_k: int = TOP_K) -> list[dict]:
    """
    Returns a list of dicts:
      [{"disease": str, "confidence": float}, …]
    sorted by confidence descending, length = top_k.
    """
    if not text.strip():
        return []

    clean     = preprocess_text(text)
    proba     = pipeline.predict_proba([clean])[0]
    top_idx   = np.argsort(proba)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "disease":    le.classes_[idx],
            "confidence": float(proba[idx]),
        })
    return results

# ─────────────────────────────────────────────────────────
#  INPUT EXTRACTORS
# ─────────────────────────────────────────────────────────

def extract_from_text(raw: str) -> str:
    """Plain text — return as-is after basic sanity check."""
    if not raw or not raw.strip():
        raise ValueError("Input text is empty.")
    return raw.strip()


def extract_from_image(image_path: str) -> str:
    """Use Tesseract OCR to extract text from an image file."""
    if pytesseract is None:
        raise ImportError(
            "pytesseract / Pillow not installed.\n"
            "Run: pip install pytesseract pillow\n"
            "And install Tesseract OCR on your system."
        )
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    img  = Image.open(path)
    text = pytesseract.image_to_string(img)
    if not text.strip():
        raise RuntimeError("OCR returned empty text. Image may be too low-resolution.")
    return text


def extract_from_pdf(pdf_path: str) -> str:
    """
    Try pdfplumber for native PDF text.
    If result is blank (scanned PDF), fall back to OCR via pdf2image.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    # ── Attempt 1: native text ────────────────────────────
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages_text).strip()
            if text:
                print("[INFO] Extracted text via pdfplumber.")
                return text
        except Exception as ex:
            print(f"[WARN] pdfplumber failed ({ex}). Trying OCR fallback…")
    else:
        print("[WARN] pdfplumber not installed. Trying OCR fallback…")

    # ── Attempt 2: OCR fallback ───────────────────────────
    if convert_from_path is None or pytesseract is None:
        raise ImportError(
            "PDF appears to be scanned/image-based but OCR dependencies are missing.\n"
            "Run: pip install pdf2image pytesseract pillow\n"
            "     sudo apt install poppler-utils tesseract-ocr"
        )

    print("[INFO] Performing OCR on PDF pages…")
    images = convert_from_path(pdf_path, dpi=200)
    pages_text = [pytesseract.image_to_string(img) for img in images]
    text = "\n".join(pages_text).strip()

    if not text:
        raise RuntimeError("OCR returned no text from PDF. The file may be corrupted.")
    return text

# ─────────────────────────────────────────────────────────
#  PRETTY OUTPUT
# ─────────────────────────────────────────────────────────

def _bar(confidence: float, width: int = 20) -> str:
    filled = int(round(confidence * width))
    return "█" * filled + "░" * (width - filled)


def print_results(results: list[dict], extracted_text: str, source: str) -> None:
    """Print a clean, human-readable report to stdout."""
    divider = "─" * 60

    print(f"\n{divider}")
    print("  🩺  DISEASE DETECTION — DECISION SUPPORT REPORT")
    print(divider)

    print(f"\n  Source : {source}")
    print(f"  Excerpt: {summarise_text(extracted_text)}")

    if not results:
        print("\n  [!] Could not produce predictions (insufficient text).")
    else:
        print(f"\n  Top {len(results)} Possible Condition(s):\n")
        for rank, r in enumerate(results, 1):
            pct = r["confidence"] * 100
            bar = _bar(r["confidence"])
            print(f"  {rank}. {r['disease']:<28} "
                  f"{pct:5.1f}%  {bar}")

    print(f"\n{divider}")
    print(textwrap.fill(
        "⚠️  DISCLAIMER: This tool is a prototype for educational "
        "purposes only. Predictions are based on keyword patterns "
        "and should NOT be used for clinical diagnosis. "
        "Consult a licensed healthcare professional.",
        width=60, initial_indent="  ", subsequent_indent="  ",
    ))
    print(divider + "\n")

# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="disease_detector.py",
        description="Disease Detection — Decision Support Prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python disease_detector.py --text "fever, headache, body ache"
              python disease_detector.py --pdf  report.pdf
              python disease_detector.py --image scan.png
              python disease_detector.py --train
              python disease_detector.py --train --csv mydata.csv
        """),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  type=str, metavar="SYMPTOMS",
                       help="Symptom description as plain text")
    group.add_argument("--pdf",   type=str, metavar="FILE",
                       help="Path to a medical report PDF")
    group.add_argument("--image", type=str, metavar="FILE",
                       help="Path to a scanned image (PNG/JPG/…)")
    group.add_argument("--train", action="store_true",
                       help="(Re)train and save the model")

    parser.add_argument("--csv",  type=str, metavar="FILE", default=None,
                        help="Optional CSV dataset for training "
                             "(columns: symptoms, disease)")
    parser.add_argument("--top",  type=int, default=TOP_K,
                        help=f"Number of top predictions (default: {TOP_K})")
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Train-only mode ───────────────────────────────────
    if args.train:
        train_model(csv_path=args.csv)
        return

    # ── Load / train model ────────────────────────────────
    pipeline, le = load_or_train_model()

    # ── Extract text from chosen input ───────────────────
    extracted_text = ""
    source_label   = ""

    try:
        if args.text:
            extracted_text = extract_from_text(args.text)
            source_label   = "Plain text input"

        elif args.pdf:
            print(f"[INFO] Reading PDF: {args.pdf}")
            extracted_text = extract_from_pdf(args.pdf)
            source_label   = f"PDF  → {args.pdf}"

        elif args.image:
            print(f"[INFO] Running OCR on image: {args.image}")
            extracted_text = extract_from_image(args.image)
            source_label   = f"Image → {args.image}"

    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as ex:
        sys.exit(f"[ERROR] {ex}")

    # ── Predict ───────────────────────────────────────────
    results = predict(extracted_text, pipeline, le, top_k=args.top)

    # ── Display ───────────────────────────────────────────
    print_results(results, extracted_text, source_label)


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────
#  HOW TO UPGRADE TO A TRANSFORMER MODEL (optional notes)
# ─────────────────────────────────────────────────────────
#
#  Replace the TF-IDF + LogReg pipeline with:
#
#  from transformers import pipeline as hf_pipeline
#
#  classifier = hf_pipeline(
#      "zero-shot-classification",
#      model="facebook/bart-large-mnli",   # or "typeform/distilbart-mnli-12-3"
#  )
#
#  candidate_labels = list(le.classes_)
#  result = classifier(extracted_text, candidate_labels, multi_label=False)
#
#  This requires: pip install transformers torch
#  and ~1–2 GB of disk space for the model weights.
#  The zero-shot approach needs NO retraining on your own data.