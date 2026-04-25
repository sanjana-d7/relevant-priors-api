"""
POST /predict — relevant priors API (relevant-priors-v1).
"""
from __future__ import annotations

import hashlib
import logging
import time

from fastapi import FastAPI, Request
from pydantic import BaseModel

from relevance_model import _ARTIFACT, load_pipeline_and_threshold, predict_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("relevant_priors")

_CACHE: dict[tuple[str, str, str], bool] = {}
_PIPE = None
_THRESH: float = 0.5


def get_pipeline_and_threshold():
    global _PIPE, _THRESH
    if _PIPE is None:
        p = _ARTIFACT
        if not p.is_file():
            raise RuntimeError(
                f"Model not found at {p}. Run: python train.py  "
                "(or: python train.py path/to/relevant_priors_public.json)"
            )
        _PIPE, _THRESH = load_pipeline_and_threshold(p)
        logger.info("Loaded model from %s threshold=%.3f", p, _THRESH)
    return _PIPE, _THRESH


class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class Case(BaseModel):
    case_id: str
    patient_id: str = ""
    patient_name: str = ""
    current_study: Study
    prior_studies: list[Study]


class PredictRequest(BaseModel):
    challenge_id: str = "relevant-priors-v1"
    schema_version: int = 1
    generated_at: str | None = None
    cases: list[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]


app = FastAPI(title="Relevant priors v1", version="1.0.0")

def _artifact_fingerprint() -> dict[str, str | int]:
    """Stable id for the on-disk .joblib so you can confirm Render = your latest train."""
    p = _ARTIFACT
    if not p.is_file():
        return {"artifact": "missing"}
    b = p.read_bytes()
    h = hashlib.sha256(b).hexdigest()[:20]
    return {
        "artifact_bytes": len(b),
        "artifact_sha20": h,
    }


@app.get("/health")
def health() -> dict[str, str | int]:
    out: dict[str, str | int] = {"status": "ok", "model": "v2-ensemble-tfidf"}
    out.update(_artifact_fingerprint())
    return out


def _run_predict(body: PredictRequest, req_id: str) -> PredictResponse:
    t0 = time.perf_counter()
    pipe, thr = get_pipeline_and_threshold()
    n_cases = len(body.cases)
    n_priors = sum(len(c.prior_studies) for c in body.cases)
    logger.info("request_id=%s cases=%d priors=%d", req_id, n_cases, n_priors)

    predictions: list[Prediction] = []
    for case in body.cases:
        cur = case.current_study.study_description
        priors = [p.study_description for p in case.prior_studies]
        pids = [p.study_id for p in case.prior_studies]
        if not pids:
            continue

        out_flags: list[bool] = [False] * len(pids)
        uncached_i: list[int] = []
        uncached_prior: list[str] = []
        for i, (pid, pd) in enumerate(zip(pids, priors, strict=True)):
            key = (case.case_id, case.current_study.study_id, pid)
            if key in _CACHE:
                out_flags[i] = _CACHE[key]
            else:
                uncached_i.append(i)
                uncached_prior.append(pd)

        if uncached_prior:
            batch = predict_batch(pipe, cur, uncached_prior, threshold=thr)
            for idx, b in zip(uncached_i, batch, strict=True):
                out_flags[idx] = b
                pid = pids[idx]
                key = (case.case_id, case.current_study.study_id, pid)
                _CACHE[key] = b

        for pid, rel in zip(pids, out_flags, strict=True):
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=pid,
                    predicted_is_relevant=rel,
                )
            )

    ms = (time.perf_counter() - t0) * 1000
    logger.info("request_id=%s done in %.1fms predictions=%d", req_id, ms, len(predictions))
    return PredictResponse(predictions=predictions)


@app.post("/predict", response_model=PredictResponse)
def predict_post(body: PredictRequest, request: Request) -> PredictResponse:
    req_id = request.headers.get("x-request-id", "-")
    return _run_predict(body, req_id)


@app.post("/", response_model=PredictResponse)
def predict_root(body: PredictRequest, request: Request) -> PredictResponse:
    req_id = request.headers.get("x-request-id", "-")
    return _run_predict(body, req_id)

