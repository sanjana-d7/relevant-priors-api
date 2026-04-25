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


def _parse_pair_line(t: str) -> tuple[str, str]:
    if "[SEP]" not in t:
        return "", ""
    a, b = t.split("[SEP]", 1)
    cur = a[8:].strip() if a.startswith("CURRENT ") else a.strip()
    pri = b[6:].strip() if b.startswith("PRIOR ") else b.strip()
    return cur, pri


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


def _token_overlap_features(pairs: list[str]) -> np.ndarray:
    """Hand features for when TF-IDF misses semantic overlap in short rads text."""
    out = np.zeros((len(pairs), 6), dtype=np.float64)
    for i, t in enumerate(pairs):
        if "[SEP]" not in t:
            continue
        cur, pri = _parse_pair_line(t)
        wc = [x for x in cur.split() if len(x) > 2]
        wp = [x for x in pri.split() if len(x) > 2]
        sc, sp = set(wc), set(wp)
        if sc and sp:
            inter = len(sc & sp)
            out[i, 0] = min(inter / 8.0, 1.0)  # cap scale
            out[i, 1] = inter / max(1, min(len(sc), len(sp)))
            out[i, 2] = inter / max(1, len(sc | sp))
        lmin, lmax = min(len(cur), len(pri)), max(len(cur), len(pri))
        out[i, 3] = lmin / max(1, lmax)
        shorter, longer = (cur, pri) if len(cur) <= len(pri) else (pri, cur)
        if len(shorter) >= 3 and shorter in longer:
            out[i, 4] = 1.0
        if wc and wp and wc[0] == wp[0]:
            out[i, 5] = 1.0
    return out


def _token_overlap_transformer(X) -> np.ndarray:  # noqa: ANN001
    arr = np.asarray(X)
    if arr.size == 0:
        return np.zeros((0, 6), dtype=np.float64)
    if arr.ndim == 0:
        seq = [str(arr.item())]
    else:
        seq = [str(s) for s in arr.ravel().tolist()]
    return _token_overlap_features(seq)


def _char_shingle_jaccard(pairs: list[str], *, k: int = 2) -> np.ndarray:
    """Jaccard on k-character shingles (ignoring pair template); good for short rads text."""

    def shingles(s: str) -> set[str]:
        s = s.replace(" ", "")
        if len(s) < k:
            return {s} if s else set()
        return {s[i : i + k] for i in range(len(s) - k + 1)}

    out = np.zeros((len(pairs), 1), dtype=np.float64)
    for i, t in enumerate(pairs):
        cur, pri = _parse_pair_line(t)
        a, b = shingles(cur), shingles(pri)
        if not a and not b:
            continue
        out[i, 0] = len(a & b) / max(1, len(a | b))
    return out


def _shingle_jaccard_transformer(X) -> np.ndarray:  # noqa: ANN001
    arr = np.asarray(X)
    if arr.size == 0:
        return np.zeros((0, 1), dtype=np.float64)
    if arr.ndim == 0:
        seq = [str(arr.item())]
    else:
        seq = [str(s) for s in arr.ravel().tolist()]
    return _char_shingle_jaccard(seq, k=2)


def build_pipeline() -> Pipeline:
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 4),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
        max_features=85_000,
    )
    subword = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=20_000,
    )
    union = FeatureUnion(
        [
            ("w_tfidf", word_tfidf),
            ("c_tfidf", subword),
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
            (
                "ovlp",
                Pipeline(
                    [
                        (
                            "ft2",
                            FunctionTransformer(
                                _token_overlap_transformer, validate=False
                            ),
                        )
                    ]
                ),
            ),
            (
                "shg",
                Pipeline(
                    [
                        (
                            "s",
                            FunctionTransformer(
                                _shingle_jaccard_transformer, validate=False
                            ),
                        )
                    ]
                ),
            ),
        ]
    )
    clf = LogisticRegression(
        class_weight="balanced",
        C=0.5,
        max_iter=3000,
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


def save_artifact(
    pipeline: Any,
    threshold: float,
    path: Path | None = None,
    *,
    st_blend: float = 0.0,
) -> None:
    path = path or _ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": float(threshold),
            "st_blend": float(st_blend),
        },
        path,
    )


def save_pipeline(pipe: Pipeline, path: Path | None = None) -> None:
    """Back-compat: same as save_artifact with 0.5 threshold."""
    save_artifact(pipe, 0.5, path=path)


def load_pipeline(path: Path | None = None) -> Any:
    path = path or _ARTIFACT
    p, _, _ = load_artifact(path)
    return p


def load_artifact(
    path: Path | None = None,
) -> tuple[Any, float, float]:
    path = path or _ARTIFACT
    obj: Any = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        p = obj["pipeline"]
        return p, float(obj.get("threshold", 0.5)), float(
            obj.get("st_blend", 0.0) or 0.0
        )
    if hasattr(obj, "predict_proba"):
        return obj, 0.5, 0.0
    raise TypeError(f"Unrecognized artifact at {path}")


def load_pipeline_and_threshold(
    path: Path | None = None,
) -> tuple[Any, float, float]:
    return load_artifact(path)


def _st_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


_ST_MODEL: Any = None


def get_st_model() -> Any:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = _st_model()
    return _ST_MODEL


def lr_and_st_for_pair_texts(
    X: list[str], pipe: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Vector of LR proba and ST [0,1] alignment for each pair string; ST grouped by current."""
    p_lr = np.asarray(pipe.predict_proba(X)[:, 1], dtype=np.float64)
    n = len(X)
    st_out = np.zeros(n, dtype=np.float64)
    by_cur: dict[str, list[tuple[int, str]]] = {}
    for i, t in enumerate(X):
        cur, pri = _parse_pair_line(t)
        by_cur.setdefault(cur, []).append((i, pri))
    for cur, items in by_cur.items():
        idxs: list[int] = []
        pris: list[str] = []
        for i, p in items:
            idxs.append(i)
            pris.append(p)
        s = st_alignment_scores(cur, pris)
        for j, ii in enumerate(idxs):
            st_out[int(ii)] = s[j]
    return p_lr, st_out


def st_alignment_scores(
    current: str, priors: list[str]
) -> np.ndarray:
    """Cosine-based alignment in [0, 1] from a small sentence encoder."""
    if not priors:
        return np.zeros((0,), dtype=np.float64)
    m = get_st_model()
    c_emb = m.encode(
        [current], normalize_embeddings=True, show_progress_bar=False
    )
    p_emb = m.encode(
        priors, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    )
    # cosine similarity in [-1,1] for L2-normalized rows
    cos = (c_emb @ p_emb.T).ravel()
    return ((cos.astype(np.float64) + 1.0) / 2.0).clip(0.0, 1.0)


def predict_batch(
    pipe: Any,
    current: str,
    priors: list[str],
    *,
    threshold: float = 0.5,
    st_blend: float = 0.0,
) -> list[bool]:
    if not priors:
        return []
    texts = [_pair_texts(current, p) for p in priors]
    p_lr = np.asarray(pipe.predict_proba(texts)[:, 1], dtype=np.float64)
    b = float(st_blend)
    if b <= 0.0:
        proba = p_lr
    else:
        b = min(b, 1.0)
        st = st_alignment_scores(current, priors)
        proba = (1.0 - b) * p_lr + b * st
    t = float(threshold)
    return [bool(p >= t) for p in proba]
