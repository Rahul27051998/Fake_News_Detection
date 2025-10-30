# src/gltr_features.py
import pandas as pd
import numpy as np
import torch, math
from tqdm import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_gltr(texts, model_name='gpt2', max_length=1024):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    feats = []
    vocab_size = model.config.vocab_size

    for text in tqdm(texts, desc="🔄 Computing GLTR features", ncols=100):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        toks = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(toks) < 2:
            feats.append({"mean_rank": np.nan, "n_tokens": 0})
            continue

        input_ids = torch.tensor(toks[:-1], device=DEVICE).unsqueeze(0)
        target_ids = torch.tensor(toks[1:], device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(input_ids).logits, dim=-1).squeeze(0)

        ranks = []
        for i, true_id in enumerate(target_ids.squeeze(0).tolist()):
            true_prob = probs[i][true_id].item()
            higher = (probs[i] > true_prob).sum().item()
            ranks.append((higher + 1) / vocab_size)

        feats.append({"mean_rank": np.mean(ranks), "n_tokens": len(ranks)})

    return pd.DataFrame(feats)

if __name__ == "__main__":
    Path("./outputs/features").mkdir(parents=True, exist_ok=True)

    train = pd.read_csv("./outputs/train_clean.csv")
    val = pd.read_csv("./outputs/val_clean.csv")

    print(f"📊 Processing {len(train)} entries from train_clean.csv...")
    gltr_train = compute_gltr(train['text_clean'].tolist())
    gltr_train.to_csv("./outputs/features/gltr_train.csv", index=False)

    print(f"📊 Processing {len(val)} entries from val_clean.csv...")
    gltr_val = compute_gltr(val['text_clean'].tolist())
    gltr_val.to_csv("./outputs/features/gltr_val.csv", index=False)

    print("✅ GLTR features saved in ./outputs/features/")

