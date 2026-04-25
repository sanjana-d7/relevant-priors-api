"""Train TF-IDF + LR on relevant_priors_public.json and write artifacts/relevance_tfidf_lr.joblib"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    for t in np.arange(0.18, 0.83, 0.004):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_arr, pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _cv_score_for_c(
    C: float,
    X: list[str],
    y: list[int],
    skf: StratifiedKFold,
) -> tuple[float, list[float], list[float]]:
    """Return mean val accuracy, per-fold accs, per-fold best thresholds."""
    y_arr = np.asarray(y, dtype=int)
    fold_acc: list[float] = []
    fold_thr: list[float] = []
    idx = np.arange(len(y))

    for tr_i, va_i in skf.split(idx, y_arr):
        X_tr = [X[i] for i in tr_i]
        y_tr = y_arr[tr_i].tolist()
        X_va = [X[i] for i in va_i]
        y_va = y_arr[va_i].tolist()
        pipe = build_pipeline()
        pipe.set_params(clf__C=C)
        pipe.fit(X_tr, y_tr)
        thr, _ = _best_threshold(pipe, X_va, y_va)
        pred = (np.asarray(pipe.predict_proba(X_va)[:, 1], dtype=np.float64) >= thr).astype(
            int
        )
        fold_acc.append(float(accuracy_score(y_va, pred)))
        fold_thr.append(thr)
    return float(np.mean(fold_acc)), fold_acc, fold_thr


def main() -> int:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_public_json_path()
    if not data_path.is_file():
        print(f"Missing {data_path}. Place relevant_priors_public.json next to train.py or pass path.", file=sys.stderr)
        return 1
    doc = json.loads(data_path.read_text(encoding="utf-8"))
    X, y = load_public_training_rows(doc)
    print(f"Loaded {len(y)} labeled pairs from {data_path.name}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_grid = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)

    best_mean, best_c = -1.0, 4.0
    best_folds: list[float] = []
    best_thrs: list[float] = []

    print("5-fold CV (per C: mean val acc, fold thrs):")
    for c in c_grid:
        mean_acc, f_acc, f_thr = _cv_score_for_c(c, X, y, skf)
        print(
            f"  C={c:<4}  mean_val_acc={mean_acc:.4f}  "
            f"folds={[round(x, 3) for x in f_acc]}  "
            f"thr_median={float(np.median(f_thr)):.3f}"
        )
        if mean_acc > best_mean:
            best_mean, best_c, best_folds, best_thrs = mean_acc, c, f_acc, f_thr

    final_thr = float(np.median(best_thrs))
    print(
        f"\nBest C={best_c}  (5-fold mean acc={best_mean:.4f})  "
        f"threshold=median of fold thrs={final_thr:.3f}"
    )

    # One reporting holdout (same split as before) for human-readable F1
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.set_params(clf__C=best_c)
    pipe.fit(X_tr, y_tr)
    pred = (pipe.predict_proba(X_va)[:, 1] >= final_thr).astype(int)
    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred, zero_division=0)
    print(f"Single 20% holdout @CV median thr: acc={acc:.4f}  F1={f1:.4f}")
    print(classification_report(y_va, pred, zero_division=0))

    pipe = build_pipeline()
    pipe.set_params(clf__C=best_c)
    pipe.fit(X, y)
    out = Path(__file__).resolve().parent / "artifacts" / "relevance_tfidf_lr.joblib"
    save_artifact(pipe, final_thr, out)
    print(
        f"Wrote {out} (full data, C={best_c}, thr={final_thr:.3f}, 5foldCV-selected, +shingle)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
