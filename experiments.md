# Experiments — Relevant Priors v1

## Baseline

- **Pair text**: each `(current_study, prior_study)` is normalized (uppercase, alphanumerics + space) and joined as `CURRENT <desc> [SEP] PRIOR <desc>`.
- **Model**: `TfidfVectorizer (word 1–2 grams)` + token **Jaccard** + **LogisticRegression** (`class_weight="balanced"`, default threshold 0.5).
- **Local public set (27,614 pairs)**: ~**87%** accuracy, F1 (positive class) ~**0.76**.
- **Platform Quick API check (10 cases / 173 priors)**: **87.28%**.

## What worked

1. **Richer text features** — added **word 1–3 / 1–4 grams**, **char `char_wb` 3–5 grams**, **token-overlap** features (counts, length ratio, substring, first-token match), and **2-char shingle Jaccard**.
   - Public eval went from ~87% → ~92%.
   - Smoke test improved from **87.28% → 96.53%**.
2. **Threshold tuning on validation** — instead of 0.5, scan ~0.18–0.83 and pick the threshold that maximizes accuracy on a held-out split. Saved with the model in the joblib artifact.
3. **5-fold StratifiedKFold C search** — replaced a single 80/20 split with 5-fold CV to pick `C` more stably (best `C ≈ 8`, mean CV accuracy ~**0.93**).
4. **Hybrid LR + MiniLM** (sentence-transformers `all-MiniLM-L6-v2`) — at predict time, blend the LR probability with a `[0,1]`-rescaled cosine of MiniLM embeddings:
   `p = (1 − blend) * p_lr + blend * st`, then threshold. Blend is tuned on a 20% holdout; if it fails to beat LR-only there, the model falls back to `blend = 0`. Modest val improvement (~+0.05 acc).
5. **Production hygiene** — batched `predict_proba` per case, in-memory cache keyed by `(case_id, current_study_id, prior_study_id)`, threshold + blend persisted in the joblib so the API loads them with the model. `/health` returns a `model` tag and a SHA-256 fingerprint of the artifact so we can verify which build is actually running on Render.

## What failed (or didn’t generalize)

1. **Soft-voting ensemble of two LRs** (different TF-IDF setups) — improved public in-sample accuracy slightly but **dropped** the platform smoke test from **96.53% → 95.38%**. The averaged probabilities moved several near-threshold pairs the wrong way. **Reverted.**
2. **Hand modality / region keyword features** (`BREAST`, `CT`, `MR`, `SPINE`, …) — neutral on validation, did not help smoke; removed from the recipe used to recover 96.53%.
3. **Naive “increase C until val accuracy peaks”** without separate selection of threshold — overfit the threshold to one fold; replaced with 5-fold CV + threshold scan per fold.
4. **Cannot target the 6 hidden errors directly** — the platform’s Quick API check reports accuracy but not the per-pair labels, so the model has to be improved generically on the public file and re-tested live.

## Lessons / pitfalls

- **Public in-sample accuracy ≠ smoke (10-case) accuracy ≠ private split accuracy.** A change that helps the public file can hurt the small smoke set, especially anything that shifts borderline decisions (ensemble, calibration shifts, threshold edits).
- **Keep a verifiable identity** for the deployed artifact (`/health` exposes `model` tag + bytes + sha20 prefix). This caught one case where the same score appeared after a “redeploy” because the new model wasn’t actually live yet.

## How I would improve it next

- **Cross-encoder** (e.g. a small BERT trained on the public pairs) instead of plain MiniLM cosine; expected to help short, ambiguous radiology descriptions.
- **Probability calibration** (isotonic on a held-out split) before threshold selection.
- **Per-modality thresholds** if validation supports it (e.g. one for MAM/breast, one for CT/MR), with strict guardrails to avoid overfitting.
- **Domain-specific encoder** (BioClinical / Rad-BERT) if deploy size and cold-start budget on the host allow.
- **Smarter caching key** — normalize description text in the key so equivalent descriptions hit the same cache entry across cases.
