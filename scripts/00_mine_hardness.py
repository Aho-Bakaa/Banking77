"""Phase 1, step 0: hardness mining + data-driven sibling graph.

Fits a cheap "System-1" probe (bge-small embeddings + TF-IDF -> logistic regression)
with 5-fold cross-validation on the Banking77 *train* split, and records for every
training query its out-of-fold (OOF) probabilities. From these we derive:

  * per-example hardness: gold prob, top-1/top-2 margin, entropy, OOF correctness
  * an empirical confusion graph between intents (which intents a cheap model mixes up)
  * the top-k plausible competitor intents per query -> fed to the teacher as
    candidate siblings, so traces compare against *real* confusers, not guesses.

Outputs
  data/raw/train_hardness.jsonl   one row per train query
  configs/confusion_pairs.json    intent pairs ranked by symmetric OOF confusion mass
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CONFIG_DIR = ROOT / "configs"
SEED = 77
TOP_K_COMPETITORS = 4


def main() -> None:
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    intent_names = sorted(train_df["category"].unique())
    intent_to_index = {name: idx for idx, name in enumerate(intent_names)}
    gold_indices = train_df["category"].map(intent_to_index).to_numpy()

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    dense_features = encoder.encode(
        train_df["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=True
    )

    oof_probabilities = np.zeros((len(train_df), len(intent_names)))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_index, (fit_rows, holdout_rows) in enumerate(folds.split(train_df, gold_indices)):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        sparse_fit = tfidf.fit_transform(train_df["text"].iloc[fit_rows])
        sparse_holdout = tfidf.transform(train_df["text"].iloc[holdout_rows])
        fit_matrix = hstack([sparse_fit, csr_matrix(dense_features[fit_rows] * 2.0)]).tocsr()
        holdout_matrix = hstack([sparse_holdout, csr_matrix(dense_features[holdout_rows] * 2.0)]).tocsr()
        probe = LogisticRegression(C=8.0, max_iter=3000)
        probe.fit(fit_matrix, gold_indices[fit_rows])
        oof_probabilities[holdout_rows] = probe.predict_proba(holdout_matrix)
        fold_accuracy = (oof_probabilities[holdout_rows].argmax(1) == gold_indices[holdout_rows]).mean()
        print(f"fold {fold_index}: acc={fold_accuracy:.4f}")

    predicted_indices = oof_probabilities.argmax(1)
    print(f"OOF accuracy (System-1 probe): {(predicted_indices == gold_indices).mean():.4f}")

    sorted_probabilities = np.sort(oof_probabilities, axis=1)[:, ::-1]
    margins = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]
    entropies = -(oof_probabilities * np.log(oof_probabilities + 1e-12)).sum(1)
    gold_probabilities = oof_probabilities[np.arange(len(train_df)), gold_indices]

    rows = []
    for row_index, text in enumerate(train_df["text"]):
        ranked = np.argsort(-oof_probabilities[row_index])
        gold_name = intent_names[gold_indices[row_index]]
        competitors = [intent_names[i] for i in ranked if intent_names[i] != gold_name][:TOP_K_COMPETITORS]
        rows.append(
            {
                "id": f"train-{row_index:05d}",
                "text": text,
                "gold_intent": gold_name,
                "probe_pred": intent_names[predicted_indices[row_index]],
                "probe_correct": bool(predicted_indices[row_index] == gold_indices[row_index]),
                "gold_prob": round(float(gold_probabilities[row_index]), 4),
                "margin": round(float(margins[row_index]), 4),
                "entropy": round(float(entropies[row_index]), 4),
                "competitors": competitors,
            }
        )
    with open(RAW_DIR / "train_hardness.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Symmetric confusion mass: sum of OOF probability that gold A leaks to B and vice versa.
    confusion_mass = np.zeros((len(intent_names), len(intent_names)))
    for row_index, gold_index in enumerate(gold_indices):
        confusion_mass[gold_index] += oof_probabilities[row_index]
    np.fill_diagonal(confusion_mass, 0.0)
    class_counts = np.bincount(gold_indices, minlength=len(intent_names))[:, None]
    confusion_rate = confusion_mass / class_counts
    symmetric_rate = confusion_rate + confusion_rate.T
    pairs = []
    for first in range(len(intent_names)):
        for second in range(first + 1, len(intent_names)):
            pairs.append((symmetric_rate[first, second], intent_names[first], intent_names[second]))
    pairs.sort(reverse=True)
    confusion_pairs = [
        {"a": first_name, "b": second_name, "confusion": round(float(score), 4)}
        for score, first_name, second_name in pairs[:150]
    ]
    CONFIG_DIR.mkdir(exist_ok=True)
    (CONFIG_DIR / "confusion_pairs.json").write_text(json.dumps(confusion_pairs, indent=1))

    hardness = pd.DataFrame(rows)
    print("probe-wrong:", (~hardness.probe_correct).sum())
    print("margin<0.3:", (hardness.margin < 0.3).sum(), " gold_prob<0.5:", (hardness.gold_prob < 0.5).sum())
    for pair in confusion_pairs[:25]:
        print(pair)


if __name__ == "__main__":
    main()
