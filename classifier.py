"""Support ticket classifier: TF-IDF + Logistic Regression.

Usage:
    python classifier.py train            # trains on tickets.csv, saves model
    python classifier.py predict "text"   # classify one ticket

Replace tickets.csv (columns: text,category) with real or public ticket data
to improve it. The bundled file is small and hand-written for demo purposes.
"""
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = "ticket_model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train(csv_path: str = "tickets.csv") -> None:
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.25, random_state=42, stratify=df["category"]
    )
    pipe = build_pipeline().fit(X_train, y_train)
    print(classification_report(y_test, pipe.predict(X_test), zero_division=0))
    joblib.dump(pipe.fit(df["text"], df["category"]), MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


def predict(text: str) -> dict:
    pipe = joblib.load(MODEL_PATH)
    probs = pipe.predict_proba([text])[0]
    best = probs.argmax()
    return {"category": pipe.classes_[best], "confidence": round(float(probs[best]), 2)}


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "predict":
        print(predict(" ".join(sys.argv[2:])))
    else:
        sys.exit(__doc__)
