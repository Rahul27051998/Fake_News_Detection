# src/predict.py
import joblib
import pandas as pd

if __name__ == "__main__":
    # Load RandomForest model and TF-IDF vectorizer
    rf = joblib.load("./outputs/models/rf_tfidf_gltr.joblib")
    X_train, y_train, X_val, y_val, tfv = joblib.load("./outputs/features/feats.pkl")

    # Example: Predict on new unseen text
    new_texts = [
        "Breaking news: The Prime Minister announced new reforms today.",
        "Scientists discovered that chocolate can cure all diseases instantly."
    ]

    X_new = tfv.transform(new_texts)
    preds = rf.predict(X_new)

    results = pd.DataFrame({
        "text": new_texts,
        "prediction": ["real" if p == 0 else "fake" for p in preds]
    })

    print(results)
    results.to_csv("./outputs/predictions/new_predictions.csv", index=False)
    print("Saved predictions to outputs/predictions/new_predictions.csv")
