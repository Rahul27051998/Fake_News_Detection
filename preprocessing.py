# src/preprocessing.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").replace("\r", " ").strip()

def preprocess(file, out_file):
    df = pd.read_csv(DATA_DIR / file)
    df['text'] = df['text'].fillna("").astype(str)
    df['text_clean'] = df['text'].apply(clean_text)
    if df['label'].dtype == object:
        df['label'] = df['label'].str.lower().map({'real': 0, 'fake': 1})
    df.to_csv(OUTPUT_DIR / out_file, index=False)
    print(f"Saved preprocessed {out_file}")

if __name__ == "__main__":
    preprocess("train.csv", "train_clean.csv")
    preprocess("val.csv", "val_clean.csv")
