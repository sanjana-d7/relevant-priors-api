"""
Pair classification: is prior study relevant to current study?
Train on (current_description, prior_description) text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "relevance_tfidf_lr.joblib"


def default_public_json_path() -> Path:
    """Public eval file from the challenge bundle, next to this package (portable for zip/submit)."""
    return Path(__file__).resolve().parent / "relevant_priors_public.json"


def _normalize_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pair_texts(current: str, prior: str) -> str:
    c = _normalize_text(current)
    p = _normalize_text(prior)
    return f"CURRENT {c} [SEP] PRIOR {p}"


def _jaccard_features(pairs: list[str]) -> np.ndarray:
    out = np.zeros((len(pairs), 1), dtype=np.float64)
    for i, t in enumerate(pairs):
        if "[SEP]" not in t:
            continue
        left, right = t.split("[SEP]", 1)
        a = {x for x in left.split() if len(x) > 2}
        b = {x for x in right.split() if len(x) > 2}
        if not a and not b:
            continue
        out[i, 0] = len(a & b) / max(1, len(a | b))
    return out


def _jaccard_transformer(X) -> np.ndarray:  # noqa: ANN001
    """Picklable entry point for FunctionTransformer (no lambdas)."""
    arr = np.asarray(X)
    if arr.size == 0:
        return np.zeros((0, 1), dtype=np.float64)
    if arr.ndim == 0:
        seq = [str(arr.item())]
    else:
        seq = [str(s) for s in arr.ravel().tolist()]
    return _jaccard_features(seq)


def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50_000,
    )
    union = FeatureUnion(
        [
            (
                "tfidf",
                tfidf,
            ),
            (
                "jacc",
                Pipeline(
                    [
                        (
                            "ft",
                            FunctionTransformer(
                                _jaccard_transformer, validate=False
                            ),
                        )
                    ]
                ),
            ),
        ]
    )
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="saga",
        n_jobs=None,
        random_state=42,
    )
    return Pipeline([("feats", union), ("clf", clf)])


def load_public_training_rows(doc: dict[str, Any]) -> tuple[list[str], list[int]]:
    """Expand truth + cases into (pair_text, label)."""
    truth = {(r["case_id"], r["study_id"]): 1 if r["is_relevant_to_current"] else 0 for r in doc["truth"]}
    case_by_id = {c["case_id"]: c for c in doc["cases"]}
    X: list[str] = []
    y: list[int] = []
    for (case_id, study_id), lab in truth.items():
        c = case_by_id.get(case_id)
        if c is None:
            continue
        cur = c["current_study"]["study_description"]
        prior_row = next((p for p in c["prior_studies"] if p["study_id"] == study_id), None)
        if prior_row is None:
            continue
        X.append(_pair_texts(cur, prior_row["study_description"]))
        y.append(lab)
    return X, y


def save_pipeline(pipe: Pipeline, path: Path | None = None) -> None:
    path = path or _ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_pipeline(path: Path | None = None) -> Pipeline:
    path = path or _ARTIFACT
    return joblib.load(path)


def predict_batch(pipe: Pipeline, current: str, priors: list[str]) -> list[bool]:
    if not priors:
        return []
    texts = [_pair_texts(current, p) for p in priors]
    proba = pipe.predict_proba(texts)[:, 1]
    return [bool(p >= 0.5) for p in proba]
