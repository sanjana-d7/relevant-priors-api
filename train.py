"""Train TF-IDF + LR on relevant_priors_public.json and write artifacts/relevance_tfidf_lr.joblib"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from relevance_model import (
    build_pipeline,
    default_public_json_path,
    load_public_training_rows,
    save_artifact,
)


def _best_threshold(
    pipeline, X_va: list[str], y_va: list[int]
) -> tuple[float, float]:
    proba = pipeline.predict_proba(X_va)[:, 1]
    y_arr = np.asarray(y_va, dtype=int)
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.005):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_arr, pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


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

    best_val, best_c, best_thr = -1.0, 0.5, 0.5
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        pipe = build_pipeline()
        pipe.set_params(clf__C=c)
        pipe.fit(X_tr, y_tr)
        thr, _ = _best_threshold(pipe, X_va, y_va)
        pred = (pipe.predict_proba(X_va)[:, 1] >= thr).astype(int)
        acc = accuracy_score(y_va, pred)
        print(f"  C={c:<4} val_acc={acc:.4f}  thr={thr:.3f}")
        if acc > best_val:
            best_val, best_c, best_thr = acc, c, thr

    pipe = build_pipeline()
    pipe.set_params(clf__C=best_c)
    pipe.fit(X_tr, y_tr)
    pred = (pipe.predict_proba(X_va)[:, 1] >= best_thr).astype(int)
    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred, zero_division=0)
    print(f"Chosen C={best_c}  threshold={best_thr:.3f}")
    print(f"Holdout accuracy @t: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_va, pred, zero_division=0))

    pipe.set_params(clf__C=best_c)
    pipe.fit(X, y)
    out = Path(__file__).resolve().parent / "artifacts" / "relevance_tfidf_lr.joblib"
    save_artifact(pipe, best_thr, out)
    print(f"Wrote {out} (full-data fit, C={best_c}, threshold={best_thr:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
