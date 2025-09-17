# train_model.py
"""
Train a baseline fake-news detector using TF-IDF + Linear SVM (SGDClassifier with hinge loss).
Saves the trained pipeline with joblib.

Usage:
    python train_model.py --data data/train.csv --out models/fake_news_clf.joblib
"""

import os
import joblib
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training CSV not found at: {path}")
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have columns: 'text' and 'label'")
    df['text'] = df['text'].astype(str).fillna("")
    df['label'] = df['label'].astype(str).str.upper().str.strip().replace({'0':'FAKE','1':'REAL'})
    return df

def build_pipeline() -> Pipeline:
    # TF-IDF vectorizer using unigrams + bigrams; linear SVM via SGDClassifier (hinge)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1,2),
            max_df=0.9,
            min_df=1,           # for tiny sample; increase min_df for big datasets
            sublinear_tf=True
        )),
        ("clf", SGDClassifier(
            loss="hinge",
            alpha=1e-4,
            max_iter=2000,
            random_state=42
        ))
    ])

def main(data_path: str, out_path: str):
    df = load_data(data_path)
    X = df['text']
    y = df['label']
    # stratify for balanced splits if possible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    pipe = build_pipeline()
    print("Training model... (this may take a moment)")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"\n✅ Model saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news detector (TF-IDF + Linear SVM)")
    parser.add_argument("--data", default="data/train.csv", help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--out",  default="models/fake_news_clf.joblib", help="Output joblib model path")
    args = parser.parse_args()
    main(args.data, args.out)