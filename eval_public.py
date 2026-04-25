"""Full-dataset accuracy on public truth (in-sample for trained model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from relevance_model import (
    _ARTIFACT,
    default_public_json_path,
    load_pipeline_and_threshold,
    load_public_training_rows,
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_public_json_path()
    if not path.is_file():
        print("Usage: eval_public.py [relevant_priors_public.json]", file=sys.stderr)
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    X, y = load_public_training_rows(doc)
    pipe, thr = load_pipeline_and_threshold(_ARTIFACT)
    y_hat = [int(p >= thr) for p in pipe.predict_proba(X)[:, 1]]
    print("accuracy", accuracy_score(y, y_hat))
    print("f1", f1_score(y, y_hat, zero_division=0))
    print(classification_report(y, y_hat, zero_division=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
