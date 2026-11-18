"""Phase 3a: Train the Fast/Think Classifier Gate.

The Gate is a lightweight, low-latency (<5ms) System-1 classifier that inspects
incoming customer queries and routes them to either:
  1. 'fast'  -> Direct label emission (<fast>{"status":"resolved","intent":"x"})
  2. 'think' -> Chain-of-Thought reasoning (<think>...</think>{payload})

Data:
  data/processed/gate_labels.jsonl (10,003 labeled train queries)
    - 'think': 2,056 queries (hard probe < 0.8, hard tier, or ambiguous)
    - 'fast':  7,947 queries (easy, clean)

Architecture:
  - SentenceTransformer: BAAI/bge-small-en-v1.5 (dense features)
  - TF-IDF Vectorizer (n-gram 1-2, sublinear)
  - Calibrated Logistic Regression / Linear Classifier
  - Exports trained pipeline to data/processed/gate_classifier.joblib
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "gate_labels.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "gate_classifier.joblib"
SEED = 77


def main() -> None:
    print("Loading gate training labels from:", DATA_PATH)
    rows = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8") if line.strip()]
    texts = [r["text"] for r in rows]
    # 1 for 'think', 0 for 'fast'
    labels = np.array([1 if r["mode"] == "think" else 0 for r in rows])

    print(f"Total gate samples: {len(labels)}")
    print(f"  Class 0 (fast):  {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    print(f"  Class 1 (think): {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")

    print("\nComputing bge-small dense embeddings...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    dense_features = encoder.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)

    print("\nEvaluating 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(labels))
    oof_probs = np.zeros(len(labels))

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        train_sparse = tfidf.fit_transform([texts[i] for i in train_idx])
        val_sparse = tfidf.transform([texts[i] for i in val_idx])

        X_train = hstack([train_sparse, csr_matrix(dense_features[train_idx] * 2.0)]).tocsr()
        X_val = hstack([val_sparse, csr_matrix(dense_features[val_idx] * 2.0)]).tocsr()

        # Class weight balanced to handle 80/20 class imbalance
        clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(X_train, labels[train_idx])

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = (val_probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, oof_probs)
    print("\n--- OUT-OF-FOLD PERFORMANCE ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(labels, oof_preds, target_names=["fast (0)", "think (1)"], digits=4))

    # Determine optimal decision threshold to prioritize high recall on 'think'
    # In banking, routing a hard query to 'fast' is risky, so recall on 'think' should be high (>90%)
    precisions, recalls, thresholds = precision_recall_curve(labels, oof_probs)
    # Find threshold with at least 90% recall on 'think'
    valid_idx = np.where(recalls >= 0.90)[0]
    best_idx = valid_idx[-1] if len(valid_idx) > 0 else len(thresholds) // 2
    recommended_thresh = round(float(thresholds[best_idx]), 3)
    print(f"Recommended Decision Threshold for ~90% Think-Recall: {recommended_thresh}")

    # Retrain on 100% of data for production export
    print("\nFitting final Gate model on full dataset...")
    final_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    full_sparse = final_tfidf.fit_transform(texts)
    X_full = hstack([full_sparse, csr_matrix(dense_features * 2.0)]).tocsr()

    final_clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=SEED)
    final_clf.fit(X_full, labels)

    gate_artifact = {
        "tfidf": final_tfidf,
        "classifier": final_clf,
        "encoder_name": "BAAI/bge-small-en-v1.5",
        "threshold": recommended_thresh,
        "classes": ["fast", "think"],
        "auc": round(float(auc), 4),
    }

    joblib.dump(gate_artifact, OUT_PATH)
    print(f"Gate classifier artifact saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
