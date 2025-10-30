# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EDA_DIR = Path("./outputs/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Load cleaned datasets
    train = pd.read_csv("./outputs/train_clean.csv")
    val = pd.read_csv("./outputs/val_clean.csv")

    # Ensure text_clean is always a string
    train['text_clean'] = train['text_clean'].fillna("").astype(str)
    val['text_clean'] = val['text_clean'].fillna("").astype(str)

    # --------------------------
    # Label distribution
    # --------------------------
    label_counts_train = train['label'].value_counts().rename_axis("label").reset_index(name="count")
    label_counts_val = val['label'].value_counts().rename_axis("label").reset_index(name="count")

    # Save label distributions
    label_counts_train.to_csv(EDA_DIR / "train_label_distribution.csv", index=False)
    label_counts_val.to_csv(EDA_DIR / "val_label_distribution.csv", index=False)

    # Plot label distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(data=label_counts_train, x="label", y="count", palette="viridis")
    plt.title("Train Label Distribution")
    plt.savefig(EDA_DIR / "train_label_distribution.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=label_counts_val, x="label", y="count", palette="magma")
    plt.title("Validation Label Distribution")
    plt.savefig(EDA_DIR / "val_label_distribution.png")
    plt.close()

    # --------------------------
    # Text length analysis
    # --------------------------
    train['text_len'] = train['text_clean'].str.split().apply(len)
    val['text_len'] = val['text_clean'].str.split().apply(len)

    # Save descriptive stats
    desc_train = train['text_len'].describe().to_frame().reset_index().rename(columns={"index": "stat", "text_len": "value"})
    desc_val = val['text_len'].describe().to_frame().reset_index().rename(columns={"index": "stat", "text_len": "value"})
    desc_train.to_csv(EDA_DIR / "train_text_length_stats.csv", index=False)
    desc_val.to_csv(EDA_DIR / "val_text_length_stats.csv", index=False)

    # Plot histogram of text lengths
    plt.figure(figsize=(8, 5))
    sns.histplot(train['text_len'], bins=50, kde=True, color="blue", label="Train", alpha=0.6)
    sns.histplot(val['text_len'], bins=50, kde=True, color="red", label="Validation", alpha=0.6)
    plt.legend()
    plt.title("Text Length Distribution (Train vs Val)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig(EDA_DIR / "text_length_distribution.png")
    plt.close()

    # --------------------------
    # Correlation heatmap (optional)
    # --------------------------
    plt.figure(figsize=(6, 4))
    corr = train[['label', 'text_len']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Train)")
    plt.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close()

    print("EDA completed. Results saved in ./outputs/eda/")


