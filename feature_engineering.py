# src/feature_engineering.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
from pathlib import Path
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    Path("./outputs/features").mkdir(parents=True, exist_ok=True)

    # Load cleaned + GLTR data
    train = pd.read_csv("./outputs/train_clean.csv")
    val = pd.read_csv("./outputs/val_clean.csv")
    gltr_train = pd.read_csv("./outputs/features/gltr_train.csv")
    gltr_val = pd.read_csv("./outputs/features/gltr_val.csv")

    print(f"📊 Building TF-IDF features for {len(train)} training and {len(val)} validation entries...")

    # -----------------------------
    # Ensure text_clean is string
    # -----------------------------
    train['text_clean'] = train['text_clean'].fillna("").astype(str)
    val['text_clean'] = val['text_clean'].fillna("").astype(str)

    # -----------------------------
    # Stylometric Features
    # -----------------------------
    def extract_stylometric(text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        words = text.split()
        word_count = len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
        return pd.Series([word_count, sentence_count, avg_word_len])

    tqdm.pandas(desc="🔄 Extracting stylometric features (train)")
    train[['word_count', 'sentence_count', 'avg_word_len']] = train['text_clean'].progress_apply(extract_stylometric)

    tqdm.pandas(desc="🔄 Extracting stylometric features (val)")
    val[['word_count', 'sentence_count', 'avg_word_len']] = val['text_clean'].progress_apply(extract_stylometric)

    train[['text_clean', 'label', 'word_count', 'sentence_count', 'avg_word_len']].to_csv(
        "./outputs/features/stylometric_train.csv", index=False)
    val[['text_clean', 'label', 'word_count', 'sentence_count', 'avg_word_len']].to_csv(
        "./outputs/features/stylometric_val.csv", index=False)
    print("✅ Stylometric feature CSVs saved")

    # -----------------------------
    # TF-IDF Vectorization
    # -----------------------------
    tfv = TfidfVectorizer(max_features=5000, stop_words='english')

    tqdm.pandas(desc="🔄 Processing train texts (TF-IDF)")
    X_train_tfidf = tfv.fit_transform(train['text_clean'].progress_apply(lambda x: str(x)))

    tqdm.pandas(desc="🔄 Processing val texts (TF-IDF)")
    X_val_tfidf = tfv.transform(val['text_clean'].progress_apply(lambda x: str(x)))

    # Export TF-IDF scores to CSV
    tfidf_tokens = tfv.get_feature_names_out()

    train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_tokens)
    train_tfidf_df['label'] = train['label'].values
    train_tfidf_df.to_csv("./outputs/features/tfidf_train.csv", index=False)

    val_tfidf_df = pd.DataFrame(X_val_tfidf.toarray(), columns=tfidf_tokens)
    val_tfidf_df['label'] = val['label'].values
    val_tfidf_df.to_csv("./outputs/features/tfidf_val.csv", index=False)
    print("✅ TF-IDF feature CSVs saved")

    # -----------------------------
    # Combine TF-IDF + GLTR
    # -----------------------------
    print("🔗 Combining TF-IDF + GLTR features...")
    X_train = hstack([X_train_tfidf, gltr_train.fillna(0).values])
    X_val = hstack([X_val_tfidf, gltr_val.fillna(0).values])

    joblib.dump((X_train, train['label'].values, X_val, val['label'].values, tfv),
                "./outputs/features/feats.pkl")
    print("✅ Feature engineering completed — saved to ./outputs/features/feats.pkl")



