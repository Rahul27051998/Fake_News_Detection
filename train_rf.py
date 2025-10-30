# src/train_rf.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from tqdm import tqdm

MODELS_DIR = Path("./outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

if __name__ == "__main__":
    # Load engineered features
    X_train, y_train, X_val, y_val, tfv = joblib.load("./outputs/features/feats.pkl")

    print(f"📊 Training RandomForest on {X_train.shape[0]} entries...")

    # Simulate epochs by training incrementally
    rf = RandomForestClassifier(warm_start=True, random_state=42, n_jobs=-1)

    n_estimators_per_epoch = 50
    total_epochs = 4  # 50 * 4 = 200 total trees
    progress = tqdm(range(1, total_epochs + 1), desc="🔄 Training RandomForest", ncols=100)

    for epoch in progress:
        rf.set_params(n_estimators=epoch * n_estimators_per_epoch)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)
        metrics = get_metrics(y_val, preds)
        progress.set_postfix(metrics)

    print("✅ RandomForest metrics:", metrics)

    # Save model
    joblib.dump(rf, MODELS_DIR / "rf_tfidf_gltr.joblib")
    print("✅ Saved RandomForest model")

