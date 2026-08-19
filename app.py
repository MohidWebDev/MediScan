"""
app.py  —  Flask Web UI for Disease Detection System
=====================================================
Place this file in the SAME folder as disease_detector.py

Install Flask:
    pip install flask

Run:
    python app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# ── Import the core logic from disease_detector ──────────
# Both files must be in the same directory.
sys.path.insert(0, str(Path(__file__).parent))

try:
    import disease_detector as dd
except ImportError as e:
    sys.exit(
        f"[ERROR] Cannot import disease_detector.py: {e}\n"
        "Make sure app.py and disease_detector.py are in the same folder."
    )

# ── Load / train model once at startup ───────────────────
print("[INFO] Loading model…")
_pipeline, _le = dd.load_or_train_model()
print("[INFO] Model ready.")

# ── Flask app ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
ALLOWED_PDF_EXT   = {".pdf"}


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/text", methods=["POST"])
def predict_text():
    """Accept JSON body: {"text": "fever, headache, …"}"""
    try:
        data = request.get_json(force=True)
        if not data or "text" not in data:
            return jsonify({"error": "No text provided."}), 400

        raw_text = data["text"].strip()
        if not raw_text:
            return jsonify({"error": "Text is empty."}), 400

        extracted = dd.extract_from_text(raw_text)
        results   = dd.predict(extracted, _pipeline, _le, top_k=3)
        excerpt   = dd.summarise_text(extracted, max_chars=400)

        return jsonify({
            "source":  "Plain text input",
            "excerpt": excerpt,
            "results": results,
        })

    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/predict/file", methods=["POST"])
def predict_file():
    """
    Accept a multipart file upload.
    Auto-detects PDF vs image by extension.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        f    = request.files["file"]
        name = f.filename or ""
        ext  = Path(name).suffix.lower()

        if ext not in ALLOWED_IMAGE_EXT | ALLOWED_PDF_EXT:
            return jsonify({
                "error": f"Unsupported file type '{ext}'. "
                         "Allowed: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP."
            }), 400

        # Save to a temp file so our extractor functions can open it
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        try:
            if ext in ALLOWED_PDF_EXT:
                extracted = dd.extract_from_pdf(tmp_path)
                source    = f"PDF → {name}"
            else:
                extracted = dd.extract_from_image(tmp_path)
                source    = f"Image → {name}"
        finally:
            os.unlink(tmp_path)   # always clean up

        results = dd.predict(extracted, _pipeline, _le, top_k=3)
        excerpt = dd.summarise_text(extracted, max_chars=400)

        return jsonify({
            "source":  source,
            "excerpt": excerpt,
            "results": results,
        })

    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as ex:
        return jsonify({"error": str(ex)}), 422
    except Exception as ex:
        return jsonify({"error": f"Unexpected error: {ex}"}), 500


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)