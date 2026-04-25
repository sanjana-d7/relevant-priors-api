"""List misclassified pairs on the public labeled set (debug / error analysis)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from relevance_model import (
    _ARTIFACT,
    _pair_texts,
    default_public_json_path,
    load_artifact,
    lr_and_st_for_pair_texts,
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_public_json_path()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    if not path.is_file():
        print("Usage: analyze_misclassifications.py [json] [limit]", file=sys.stderr)
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    truth = {
        (r["case_id"], r["study_id"]): bool(r["is_relevant_to_current"])
        for r in doc["truth"]
    }
    case_by_id = {c["case_id"]: c for c in doc["cases"]}
    pipe, thr, blend = load_artifact(_ARTIFACT)

    rows: list[tuple[str, str, str, str, bool]] = []
    for (case_id, study_id), lab in truth.items():
        c = case_by_id.get(case_id)
        if c is None:
            continue
        cur = c["current_study"]["study_description"]
        prior_row = next(
            (p for p in c["prior_studies"] if p["study_id"] == study_id), None
        )
        if prior_row is None:
            continue
        prior = prior_row["study_description"]
        rows.append((case_id, study_id, cur, prior, lab))

    Xs = [_pair_texts(c, p) for _, _, c, p, _ in rows]
    p_lr, st = lr_and_st_for_pair_texts(Xs, pipe)
    if float(blend) > 0.0:
        proba = (1.0 - float(blend)) * p_lr + float(blend) * st
    else:
        proba = p_lr

    wrong: list[tuple[str, str, str, str, bool, bool, float]] = []
    for i, (case_id, study_id, cur, prior, lab) in enumerate(rows):
        p_pos = float(proba[i])
        pred = p_pos >= thr
        if pred != lab:
            wrong.append((case_id, study_id, cur, prior, lab, pred, p_pos))

    print(f"Misclassified (public): {len(wrong)} / {len(truth)}")
    print(f"Threshold: {thr:.4f}\n--- sample (first {limit}) ---")
    for row in wrong[:limit]:
        cid, sid, cur, prior, lab, pred, p_pos = row
        print(
            f"\ncase={cid} prior_study={sid} truth={lab} pred={pred} p={p_pos:.3f}\n"
            f"  current: {cur[:200]}{'...' if len(cur) > 200 else ''}\n"
            f"  prior:   {prior[:200]}{'...' if len(prior) > 200 else ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
