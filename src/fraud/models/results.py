"""Running record of ladder-step results, written to reports/model_ladder.json.

Each (step key, feature set) pair keeps its single best result, so re-running a sweep overwrites its
own row rather than appending duplicates. This file is the source for the README comparison table
and for picking the champion candidate that `params.yaml`'s `model` section then pins for the
deterministic `train`/`evaluate` dvc.yaml stages.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from fraud.params import REPO_ROOT

LADDER_RESULTS_PATH = REPO_ROOT / "reports" / "model_ladder.json"


@dataclass
class LadderResult:
    step_id: int
    key: str
    name: str
    feature_set: str
    hyperparams: dict[str, Any]
    recall_at_precision: float
    precision_at_threshold: float
    pr_auc: float
    met_budget: bool
    n_trials: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def load_ladder_results(path: Any = LADDER_RESULTS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def append_ladder_result(result: LadderResult, path: Any = LADDER_RESULTS_PATH) -> None:
    results = load_ladder_results(path)
    results = [
        r
        for r in results
        if not (r["key"] == result.key and r["feature_set"] == result.feature_set)
    ]
    results.append(asdict(result))
    results.sort(key=lambda r: (r["step_id"], r["key"], r["feature_set"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
