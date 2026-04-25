"""Train TF-IDF + LR on relevant_priors_public.json and write artifacts/relevance_tfidf_lr.joblib"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from relevance_model import (
    build_pipeline,
    default_public_json_path,
    load_public_training_rows,
    save_pipeline,
)


def main() -> int:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_public_json_path()
    if not data_path.is_file():
        print(f"Missing {data_path}. Place relevant_priors_public.json next to train.py or pass path.", file=sys.stderr)
        return 1
    doc = json.loads(data_path.read_text(encoding="utf-8"))
    X, y = load_public_training_rows(doc)
    print(f"Loaded {len(y)} labeled pairs from {data_path.name}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred, zero_division=0)
    print(f"Holdout accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_va, pred, zero_division=0))

    pipe.fit(X, y)
    out = Path(__file__).resolve().parent / "artifacts" / "relevance_tfidf_lr.joblib"
    save_pipeline(pipe, out)
    print(f"Wrote {out} (full-data fit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
