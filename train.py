"""Train TF-IDF + LR on relevant_priors_public.json; optional ST blend tuned on a holdout."""
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


def _best_threshold_vec(
    proba: np.ndarray, y_va: list[int] | np.ndarray
) -> tuple[float, float]:
    y_arr = np.asarray(y_va, dtype=int)
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.15, 0.90, 0.003):
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
        p_va = np.asarray(pipe.predict_proba(X_va)[:, 1], dtype=np.float64)
        thr, _ = _best_threshold_vec(p_va, y_va)
        pred = (p_va >= thr).astype(int)
        fold_acc.append(float(accuracy_score(y_va, pred)))
        fold_thr.append(thr)
    return float(np.mean(fold_acc)), fold_acc, fold_thr


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    lr_only = "--lr-only" in argv
    argv = [a for a in argv if a != "--lr-only"]
    data_path = Path(argv[0]) if argv else default_public_json_path()
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

    print("5-fold CV (LR only, per C):")
    for c in c_grid:
        mean_acc, f_acc, f_thr = _cv_score_for_c(c, X, y, skf)
        print(
            f"  C={c:<4}  mean_val_acc={mean_acc:.4f}  "
            f"folds={[round(x, 3) for x in f_acc]}  "
            f"thr_median={float(np.median(f_thr)):.3f}"
        )
        if mean_acc > best_mean:
            best_mean, best_c, best_folds, best_thrs = mean_acc, c, f_acc, f_thr

    print(f"\nBest C={best_c}  5-fold mean acc={best_mean:.4f}\n")

    # Holdout: tune threshold; optional MiniLM blend (install requirements-train.txt)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.set_params(clf__C=best_c)
    pipe.fit(X_tr, y_tr)
    p_lr = np.asarray(pipe.predict_proba(X_val)[:, 1], dtype=np.float64)
    y_val_arr = np.asarray(y_val, dtype=int)

    t_lr, acc_lr = _best_threshold_vec(p_lr, y_val)
    print(f"LR only on val: acc={acc_lr:.4f}  thr={t_lr:.3f}")

    best_b, best_t = 0.0, t_lr
    if lr_only:
        print("(--lr-only) Skipping sentence-transformer hybrid; st_blend=0 for deploy size.")
    else:
        try:
            from relevance_model import lr_and_st_for_pair_texts
        except ImportError:
            print(
                "MiniLM not available (install: pip install -r requirements-train.txt). "
                "Using LR-only.",
                file=sys.stderr,
            )
            best_b, best_t = 0.0, t_lr
        else:
            _, st = lr_and_st_for_pair_texts(X_val, pipe)
            best_h = (acc_lr, 0.0, t_lr)
            for b in (0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35):
                comb = (1.0 - b) * p_lr + b * st
                for t in np.arange(0.12, 0.92, 0.002):
                    pred = (comb >= t).astype(int)
                    acc = accuracy_score(y_val_arr, pred)
                    if acc > best_h[0]:
                        best_h = (float(acc), float(b), float(t))
            h_acc, best_b, best_t = best_h
            print(
                f"Best hybrid on val: acc={h_acc:.4f}  st_blend={best_b:.3f}  thr={best_t:.3f}"
            )
            if h_acc <= acc_lr:
                print("Using LR-only (hybrid did not beat LR on this holdout).")
                best_b, best_t = 0.0, t_lr

    pipe = build_pipeline()
    pipe.set_params(clf__C=best_c)
    pipe.fit(X, y)
    out = Path(__file__).resolve().parent / "artifacts" / "relevance_tfidf_lr.joblib"
    save_artifact(pipe, best_t, out, st_blend=best_b)
    print(
        f"Wrote {out}  C={best_c}  thr={best_t:.3f}  st_blend={best_b:.3f}  "
        f"(+MiniLM on predict when st_blend>0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
