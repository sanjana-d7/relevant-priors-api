# Experiments

## Baseline

- Trained a **logistic regression** on the public `truth` labels using **TF-IDF (1–2 grams)** on a normalized `CURRENT [desc] [SEP] PRIOR [desc]` string plus a **token Jaccard** overlap feature between the two descriptions.
- **Holdout (20% stratified)**: accuracy about **0.87**, F1 (positive class) about **0.76**.
- **Full public training set (in-sample)**: same ballpark (labels are the training signal; linear model does not memorize).

## What worked

- Batching: one `predict_proba` call per case over **all** uncached priors (never one HTTP round-trip per study).
- In-memory **cache** keyed by `(case_id, current_study_id, prior_study_id)` so repeat evaluations do not recompute.
- `class_weight='balanced'` to handle imbalanced positive/negative priors in the public split.

## What failed (or is limited)

- A pure string model cannot encode full clinical reasoning (protocol, reason for exam, findings). The private set may differ in distribution.
- The toy example in the problem statement (MRI brain vs prior CT head → false) may not match a purely data-driven model; the scorer uses hidden labels, not the example.

## How I would improve it

- **Calibration** (Platt or isotonic) on a validation split, then choose threshold to maximize F1 or match operational precision/recall.
- **Entity features**: modality (CT/MR/US/NM), body region, contrast flags — extracted with simple rules and concatenated to TF-IDF or fed into a small tree model.
- **Batched LLM** with a **single prompt per case** listing all priors (as required by the challenge hint), with a compact JSON-only response and strict post-validation — only if latency and key budget allow within the 360s cap.
