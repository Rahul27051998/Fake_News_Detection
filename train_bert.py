# src/train_bert.py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

MODELS_DIR = Path("./outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv("./outputs/train_clean.csv")
    val_df = pd.read_csv("./outputs/val_clean.csv")

    # Ensure clean text is string
    train_df['text_clean'] = train_df['text_clean'].fillna("").astype(str)
    val_df['text_clean'] = val_df['text_clean'].fillna("").astype(str)

    # Load tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function (ensures string input)
    def tokenize(batch):
        texts = [str(x) if x is not None else "" for x in batch["text_clean"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

    # Convert to Hugging Face Datasets
    print(f"📊 Tokenizing {len(train_df)} training entries...")
    train_ds = Dataset.from_pandas(train_df)
    train_ds = train_ds.map(tokenize, batched=True, batch_size=32, desc="🔄 Tokenizing train set")

    print(f"📊 Tokenizing {len(val_df)} validation entries...")
    val_ds = Dataset.from_pandas(val_df)
    val_ds = val_ds.map(tokenize, batched=True, batch_size=32, desc="🔄 Tokenizing val set")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "bert_checkpoint"),
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        save_total_limit=2,
        logging_dir=str(MODELS_DIR / "logs"),
        logging_steps=50,
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("🚀 Starting BERT training...")
    trainer.train()
    trainer.save_model(MODELS_DIR / "bert_finetuned")
    print("✅ Saved BERT model")


