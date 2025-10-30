# src/app.py
"""
Combined FastAPI + Streamlit app for Fake News Detection.

Usage:
  # Run FastAPI server (recommended for production)
  uvicorn src.app:app --host 0.0.0.0 --port 8000

  # Run Streamlit frontend (will start a Streamlit server that calls the local API)
  python src/app.py --streamlit

Notes:
 - Models and features expected under ./outputs/
   - outputs/models/rf_tfidf_gltr.joblib
   - outputs/features/feats.pkl  (tfidf vectorizer)
   - outputs/models/bert_finetuned/ (HF saved)
   - outputs/models/gpt2_finetuned/ (HF saved)
 - Running both BERT and GPT-2 may require GPU / large RAM.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel

# ---- Lightweight imports used by both sides ----
import joblib
import numpy as np
import torch

# ---- FastAPI setup ----
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Fake News Detection - Multi-model API (RF + BERT + GPT-2)")

ROOT = Path(".")
MODELS_DIR = ROOT / "outputs" / "models"
FEATURES_FILE = ROOT / "outputs" / "features" / "feats.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_tfidf_gltr.joblib"
BERT_DIR = MODELS_DIR / "bert_finetuned"
GPT2_DIR = MODELS_DIR / "gpt2_finetuned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load TF-IDF and RandomForest
# -------------------------
rf_model = None
tfv = None
X_train = y_train = X_val = y_val = None

try:
    # feats.pkl expected: (X_train, y_train, X_val, y_val, tfv)
    X_train, y_train, X_val, y_val, tfv = joblib.load(FEATURES_FILE)
except Exception:
    tfv = None

if RF_MODEL_PATH.exists():
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
    except Exception:
        rf_model = None

# -------------------------
# Load BERT
# -------------------------
bert_tokenizer = None
bert_model = None
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if BERT_DIR.exists():
        bert_tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))
        bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR)).to(DEVICE)
    else:
        # fallback to tokenizer only (model not fine-tuned/present)
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = None
except Exception:
    bert_tokenizer = None
    bert_model = None

# -------------------------
# Load GPT-2
# -------------------------
gpt2_tokenizer = None
gpt2_model = None
try:
    from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification
    if GPT2_DIR.exists():
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(str(GPT2_DIR))
        gpt2_model = GPT2ForSequenceClassification.from_pretrained(str(GPT2_DIR)).to(DEVICE)
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # if vocabulary changed, try resizing embeddings
            try:
                gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
            except Exception:
                pass
    else:
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        gpt2_model = None
except Exception:
    gpt2_tokenizer = None
    gpt2_model = None

# -------------------------
# Prediction utilities
# -------------------------
def logits_to_label_prob(logits: np.ndarray) -> Dict[str, Any]:
    # Safely convert logits -> softmax -> predicted label + prob
    try:
        import scipy.special
        probs = scipy.special.softmax(logits, axis=-1)
        pred = int(np.argmax(probs))
        return {"label": pred, "prob": float(probs[pred])}
    except Exception:
        # fallback: argmax only
        pred = int(np.argmax(logits))
        return {"label": pred, "prob": None}


def rf_predict(text: str) -> Dict[str, Any]:
    if rf_model is None or tfv is None:
        raise RuntimeError("RandomForest or TF-IDF vectorizer not loaded.")
    vec = tfv.transform([text])
    pred = int(rf_model.predict(vec)[0])
    prob = None
    if hasattr(rf_model, "predict_proba"):
        try:
            proba = rf_model.predict_proba(vec)[0]
            prob = float(proba[pred])
        except Exception:
            prob = None
    return {"label": pred, "prob": prob}


def bert_predict(text: str) -> Dict[str, Any]:
    if bert_tokenizer is None:
        raise RuntimeError("BERT tokenizer/model not loaded.")
    inputs = bert_tokenizer([text], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if bert_model is None:
        raise RuntimeError("BERT model not loaded (fine-tuned).")
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits.cpu().numpy().squeeze(0)
    return logits_to_label_prob(logits)


def gpt2_predict(text: str) -> Dict[str, Any]:
    if gpt2_tokenizer is None:
        raise RuntimeError("GPT-2 tokenizer/model not loaded.")
    inputs = gpt2_tokenizer([text], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if gpt2_model is None:
        raise RuntimeError("GPT-2 model not loaded (fine-tuned).")
    gpt2_model.eval()
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        logits = outputs.logits.cpu().numpy().squeeze(0)
    return logits_to_label_prob(logits)

# -------------------------
# FastAPI endpoints
# -------------------------
class TextIn(BaseModel):
    text: str

class BatchTextIn(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {
        "service": "Fake News Detection (RF + BERT + GPT-2)",
        "models": {
            "random_forest": bool(rf_model and tfv),
            "bert_finetuned": bool(bert_model),
            "gpt2_finetuned": bool(gpt2_model),
        },
        "device": DEVICE,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rf_loaded": bool(rf_model and tfv),
        "bert_loaded": bool(bert_model),
        "gpt2_loaded": bool(gpt2_model),
    }

@app.post("/predict/single")
def predict_single(item: TextIn):
    text = item.text
    results = {}
    # RF
    try:
        results["random_forest"] = rf_predict(text)
    except Exception as e:
        results["random_forest"] = {"error": str(e)}
    # BERT
    try:
        results["bert"] = bert_predict(text)
    except Exception as e:
        results["bert"] = {"error": str(e)}
    # GPT-2
    try:
        results["gpt2"] = gpt2_predict(text)
    except Exception as e:
        results["gpt2"] = {"error": str(e)}
    return {"text": text, "results": results}

@app.post("/predict/batch")
def predict_batch(batch: BatchTextIn):
    texts = batch.texts
    outputs = []
    for t in texts:
        outputs.append(predict_single(TextIn(text=t)))
    return {"batch_results": outputs}

# -------------------------
# Streamlit UI (defined as a function)
# -------------------------
def run_streamlit_ui(api_url: str = "http://localhost:8000/predict/single"):
    """
    Launch a simple Streamlit UI that calls the FastAPI /predict/single endpoint.
    To start this UI run:
        python src/app.py --streamlit
    or (if you have streamlit installed) python -m streamlit run src/app.py -- --streamlit
    """
    try:
        import streamlit as st
    except Exception as e:
        raise RuntimeError("Streamlit not installed. Install `streamlit` to run the UI.") from e

    st.set_page_config(page_title="Fake News Detection (Demo)", layout="wide")
    st.title("Fake News Detection — Demo Frontend")

    st.markdown(
        """This demo calls the local FastAPI endpoint to get predictions from three models:
        **Random Forest**, **BERT**, and **GPT-2**."""
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        text = st.text_area("Enter news article or headline:", height=300)
        if st.button("Predict"):
            if not text.strip():
                st.warning("Please enter text to classify.")
            else:
                payload = {"text": text}
                try:
                    resp = st.experimental_singleton.clear()  # noop to ensure experimental functions available
                except Exception:
                    pass
                try:
                    import requests
                    r = requests.post(api_url, json=payload, timeout=20)
                    r.raise_for_status()
                    data = r.json()
                    # Show raw output
                    st.subheader("Raw API Output")
                    st.json(data)
                    # Present clean cards
                    st.subheader("Model Predictions")
                    rf = data["results"].get("random_forest", {})
                    bert = data["results"].get("bert", {})
                    gpt2 = data["results"].get("gpt2", {})
                    def pretty_label(d):
                        if isinstance(d, dict) and "label" in d:
                            lbl = int(d["label"])
                            return f"{lbl} (REAL)" if lbl == 1 else f"{lbl} (FAKE)"
                        return "N/A"
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Random Forest", pretty_label(rf))
                    c2.metric("BERT", pretty_label(bert))
                    c3.metric("GPT-2", pretty_label(gpt2))
                except Exception as ex:
                    st.error(f"Call to API failed: {ex}")

    with col2:
        st.markdown("### API / System Info")
        st.write(f"- API URL: `{api_url}`")
        loaded = {
            "RF loaded": bool(rf_model and tfv),
            "BERT loaded": bool(bert_model),
            "GPT-2 loaded": bool(gpt2_model),
        }
        st.json(loaded)
        st.markdown("### Notes")
        st.write(
            "- Labels: `0` = FAKE, `1` = REAL\n"
            "- If a model is not loaded the card will show `N/A`.\n"
            "- Running both transformer models may require a machine with sufficient GPU/CPU and RAM."
        )

# -------------------------
# CLI entry: support --streamlit
# -------------------------
if __name__ == "__main__":
    # If run as script: allow launching Streamlit via --streamlit
    if "--streamlit" in sys.argv:
        # Programmatic invocation using the Streamlit CLI bootstrap.
        # This will start a Streamlit server that imports this module (so the run_streamlit_ui will be executed).
        # To ensure Streamlit executes the UI code we bootstrap the CLI with args that include this file and the flag.
        try:
            # Set up args for streamlit to run this file and pass a sentinel that triggers the UI.
            # We pass an extra sentinel so inside the module Streamlit's internal runner loads the file normally.
            sys.argv = ["streamlit", "run", __file__, "--", "--embedded"]
            from streamlit.web import cli as stcli
            stcli.main()
        except Exception as e:
            print("Error launching Streamlit via CLI:", e)
            raise
    elif "--run-ui" in sys.argv:
        # Direct python execution without streamlit CLI: call the function that uses streamlit API
        # (Not the typical way; prefer `python src/app.py --streamlit` or `streamlit run src/app.py`.)
        run_streamlit_ui()
    else:
        print("\nThis module contains both a FastAPI app and a Streamlit demo UI.")
        print("Run the FastAPI server with:")
        print("  uvicorn src.app:app --host 0.0.0.0 --port 8000")
        print("Run the Streamlit UI with:")
        print("  python src/app.py --streamlit")
        print("  (or) streamlit run src/app.py -- --embedded")
        print("\nNote: Streamlit will call the running FastAPI endpoint to obtain predictions.\n")

