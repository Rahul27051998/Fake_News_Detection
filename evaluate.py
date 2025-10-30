# src/evaluate.py
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    GPT2TokenizerFast, GPT2ForSequenceClassification, Trainer
)

# Directories
MODELS_DIR = Path("./outputs/models")
DATA_DIR = Path("./outputs")
PRED_DIR = Path("./outputs/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Metrics
# --------------------------
def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    val_df["text_clean"] = val_df["text_clean"].fillna("").astype(str)
    y_true = val_df["label"].tolist()

    results = {}

    # --------------------------
    # 1. RandomForest (TF-IDF + GLTR features)
    # --------------------------
    print(f"📊 Evaluating RandomForest on {len(val_df)} entries...")
    try:
        # Load precomputed features (train + val)
        X_train, y_train, X_val, y_val, tfv = joblib.load("./outputs/features/feats.pkl")
        rf_model = joblib.load(MODELS_DIR / "rf_tfidf_gltr.joblib")

        # Predict all at once (faster, correct shape)
        preds = rf_model.predict(X_val)

        # Show progress bar as simulation
        for _ in tqdm(range(len(preds)), desc="🔄 RF predicting", ncols=100):
            pass

        metrics = get_metrics(y_val, preds)
        results["rf"] = metrics
        val_df["rf_pred"] = preds
        print(f"✅ RF metrics: {metrics}")
    except Exception as e:
        print(f"⚠️ Skipping RF evaluation (error: {e})")

    # --------------------------
    # 2. BERT
    # --------------------------
    print(f"📊 Evaluating BERT on {len(val_df)} entries...")
    try:
        bert_model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "bert_finetuned")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_bert(batch):
            texts = [str(x) for x in batch["text_clean"]]
            return bert_tokenizer(texts, truncation=True, padding="max_length", max_length=256)

        val_ds = Dataset.from_pandas(val_df)
        val_ds = val_ds.map(tokenize_bert, batched=True, batch_size=32, desc="🔄 Tokenizing val set for BERT")

        trainer = Trainer(model=bert_model, tokenizer=bert_tokenizer)
        preds_logits = trainer.predict(val_ds).predictions
        preds = np.argmax(preds_logits, axis=-1)

        metrics = get_metrics(y_true, preds)
        results["bert"] = metrics
        val_df["bert_pred"] = preds
        print(f"✅ BERT metrics: {metrics}")
    except Exception as e:
        print(f"⚠️ Skipping BERT evaluation (error: {e})")

    # --------------------------
    # 3. GPT-2
    # --------------------------
    print(f"📊 Evaluating GPT-2 on {len(val_df)} entries...")
    try:
        gpt2_model = GPT2ForSequenceClassification.from_pretrained(MODELS_DIR / "gpt2_finetuned")
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        def tokenize_gpt2(batch):
            texts = [str(x) for x in batch["text_clean"]]
            return gpt2_tokenizer(texts, truncation=True, padding="max_length", max_length=256)

        val_ds = Dataset.from_pandas(val_df)
        val_ds = val_ds.map(tokenize_gpt2, batched=True, batch_size=32, desc="🔄 Tokenizing val set for GPT-2")

        trainer = Trainer(model=gpt2_model, tokenizer=gpt2_tokenizer)
        preds_logits = trainer.predict(val_ds).predictions
        preds = np.argmax(preds_logits, axis=-1)

        metrics = get_metrics(y_true, preds)
        results["gpt2"] = metrics
        val_df["gpt2_pred"] = preds
        print(f"✅ GPT-2 metrics: {metrics}")
    except Exception as e:
        print(f"⚠️ Skipping GPT-2 evaluation (error: {e})")

    # --------------------------
    # Save results
    # --------------------------
    metrics_file = PRED_DIR / "metrics_summary.csv"
    preds_file = PRED_DIR / "val_predictions.csv"

    pd.DataFrame(results).to_csv(metrics_file)
    val_df.to_csv(preds_file, index=False)

    print(f"✅ Saved metrics summary to {metrics_file}")
    print(f"✅ Saved detailed predictions to {preds_file}")






