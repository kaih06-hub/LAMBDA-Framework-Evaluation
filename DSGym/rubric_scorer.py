"""
Numerical rubric scorer for the 12-step pipeline experiment from run_lambda.py.

Reads:  ./results/run_*.md          (per-run logs from run_lambda.py)
Writes: ./rubric_scores/per_run.csv      one row per run, raw signals + scores
        ./rubric_scores/summary.csv      mean / stdev per criterion
        ./rubric_scores/run_details.json detailed evidence per run

Implements numerical proxies for seven of the qualitative criteria from the
prior LAMBDA-vs-ChatGPT-vs-Copilot study:

  3. Ability to complete the task     (steps_completed / 12) × 5
  4. Statefulness                     cross-step variable reuse density
  5. Ease of tuning model parameters  GridSearchCV success + best-params reported
  6. Stability (Data Decisions)       cross-run consistency on preprocessing
  7. Stability (Method Choices)       cross-run consistency on model defaults
  8. Continuity across steps          1 − (variable-redefinition penalty)
  9. Transparency / Auditability      explanation density per code chunk

Stability scores (6, 7) are computed across the full set of runs and reported
as a single number per criterion; per-run rows mirror the per-decision values
that fed the stability calculation.

Usage:
    python rubric_scorer.py
    python rubric_scorer.py --results-dir ./results --out ./rubric_scores
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Step detection — patterns mapped to each of the 12 prompt steps
# ---------------------------------------------------------------------------
STEP_PATTERNS: Dict[int, List[str]] = {
    1: [r"read_csv|read_excel|pd\.read_|load_dataset|\.shape|columns\s*=", r"loaded.*data|data\s+loaded|dataset\s+loaded"],
    2: [r"\.describe\(\)|\.info\(\)|\.value_counts|seaborn|sns\.|plt\.|histogram|boxplot|correlation\s*matrix|heatmap"],
    3: [r"train_test_split|test_size\s*=|X_train|X_test|y_train|y_test"],
    4: [r"feature[_\s]?selection|select(?:ed)?\s+features|drop\(|drop_duplicates|X\s*=\s*[A-Za-z_]+\["],
    5: [r"LogisticRegression|logistic\s*regression"],
    6: [r"\bSVC\b|\bSVM\b|svm\.|support\s*vector"],
    7: [r"MLPClassifier|MLPRegressor|\bMLP\b|multi-?layer\s*perceptron"],
    8: [r"DecisionTreeClassifier|DecisionTreeRegressor|decision\s*tree"],
    9: [r"RandomForestClassifier|RandomForestRegressor|random\s*forest"],
    10: [r"GridSearchCV|grid_search|param_grid|best_params_|best_score_"],
    11: [r"joblib\.dump|pickle\.dump|\.pkl|\.joblib|model\.save|save_model"],
    12: [r"summary\s+report|generate\s+(?:a\s+)?report|##?\s*Conclusion|##?\s*Summary|##?\s*Report"],
}

ERROR_PATTERNS = [
    r"\bTraceback\b", r"\bException:", r"\bError:", r"\bFAILED\b",
    r"NameError", r"ValueError", r"TypeError", r"KeyError", r"AttributeError",
]

# ---------------------------------------------------------------------------
# Data-decision detectors (Criterion 6)
# ---------------------------------------------------------------------------
DATA_DECISIONS: Dict[str, Dict[str, str]] = {
    "missing_value_strategy": {
        "drop": r"dropna\(",
        "mean_imputation": r"SimpleImputer.*?strategy\s*=\s*['\"]mean['\"]|fillna\([^)]*mean\(",
        "median_imputation": r"SimpleImputer.*?strategy\s*=\s*['\"]median['\"]|fillna\([^)]*median\(",
        "mode_imputation": r"SimpleImputer.*?strategy\s*=\s*['\"]most_frequent['\"]|fillna\([^)]*mode\(",
        "constant_fill": r"fillna\([^)]*\b(?:0|-?\d+)\b\)",
    },
    "categorical_encoding": {
        "one_hot": r"OneHotEncoder|get_dummies",
        "label": r"LabelEncoder",
        "ordinal": r"OrdinalEncoder",
    },
    "scaling": {
        "standard": r"StandardScaler",
        "minmax": r"MinMaxScaler",
        "robust": r"RobustScaler",
        "none": r"^$",  # filled in if no scaler detected
    },
}

# ---------------------------------------------------------------------------
# Method-choice detectors (Criterion 7)
# ---------------------------------------------------------------------------
METHOD_CHOICES: Dict[str, Dict[str, str]] = {
    "logreg_solver": {
        "lbfgs": r"LogisticRegression\([^)]*solver\s*=\s*['\"]lbfgs['\"]",
        "liblinear": r"LogisticRegression\([^)]*solver\s*=\s*['\"]liblinear['\"]",
        "saga": r"LogisticRegression\([^)]*solver\s*=\s*['\"]saga['\"]",
        "default": r"LogisticRegression\(\)",
    },
    "svm_kernel": {
        "rbf": r"SVC\([^)]*kernel\s*=\s*['\"]rbf['\"]",
        "linear": r"SVC\([^)]*kernel\s*=\s*['\"]linear['\"]",
        "poly": r"SVC\([^)]*kernel\s*=\s*['\"]poly['\"]",
        "default": r"SVC\(\)",
    },
    "rf_n_estimators": {
        "100": r"RandomForestClassifier\([^)]*n_estimators\s*=\s*100",
        "200": r"RandomForestClassifier\([^)]*n_estimators\s*=\s*200",
        "default": r"RandomForestClassifier\(\)",
    },
    "test_size": {
        "0.2": r"test_size\s*=\s*0\.2\b",
        "0.3": r"test_size\s*=\s*0\.3\b",
        "0.25": r"test_size\s*=\s*0\.25\b",
        "other": r"test_size\s*=\s*0?\.\d+",
    },
}

VARIABLE_NAMES = [
    "X_train", "X_test", "y_train", "y_test",
    "X", "y", "df", "model", "scaler", "encoder",
    "pipeline", "preprocessor",
]


# ===========================================================================
# Per-run extraction
# ===========================================================================
def _read_run(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[rubric] could not read {path}: {e}")
        return ""


def detect_step_completion(text: str) -> Tuple[List[bool], int]:
    """Return per-step completion booleans and how many errors were observed."""
    completed = []
    for i in range(1, 13):
        patterns = STEP_PATTERNS[i]
        completed.append(any(re.search(p, text, re.IGNORECASE) for p in patterns))
    n_errors = sum(
        len(re.findall(p, text, re.IGNORECASE)) for p in ERROR_PATTERNS
    )
    return completed, n_errors


def score_completion(steps: List[bool], n_errors: int) -> float:
    """Criterion 3: ability to complete (1-5)."""
    completed = sum(steps)
    base = (completed / 12.0) * 5.0
    # Per-error penalty up to -1
    penalty = min(1.0, n_errors * 0.1)
    return round(max(0.0, min(5.0, base - penalty)), 2)


def score_statefulness(text: str) -> Tuple[float, Dict[str, int]]:
    """Criterion 4: variable reuse across the run.

    Counts references to canonical pipeline variables (X_train, model, etc.)
    that exceed their *first* appearance — i.e. evidence the variable persisted
    and was reused.
    """
    counts: Dict[str, int] = {}
    reuse_total = 0
    references_total = 0
    for v in VARIABLE_NAMES:
        n = len(re.findall(rf"\b{re.escape(v)}\b", text))
        counts[v] = n
        references_total += n
        # "reuse" = references beyond the first
        reuse_total += max(0, n - 1)
    if references_total == 0:
        return 0.0, counts
    # Density of reuse, scaled to [0, 5]
    density = reuse_total / max(1, references_total)
    score = round(min(5.0, density * 5.0 + 1.0), 2)
    return score, counts


def score_tuning(text: str, errors_in_step10_window: bool) -> Tuple[float, Dict[str, Any]]:
    """Criterion 5: ease of tuning (success of GridSearchCV step)."""
    has_grid = bool(re.search(r"GridSearchCV|param_grid", text))
    has_best_params = bool(re.search(r"best_params_|best estimator|best score|best_score_", text, re.IGNORECASE))
    has_fit = bool(re.search(r"\.fit\(", text))
    score = 0.0
    if has_grid:
        score += 2.0
    if has_grid and has_fit:
        score += 1.0
    if has_best_params:
        score += 2.0
    if errors_in_step10_window:
        score = max(0.0, score - 1.5)
    return round(min(5.0, score), 2), {
        "has_grid_search": has_grid,
        "has_best_params": has_best_params,
        "has_fit_call": has_fit,
    }


def detect_data_decisions(text: str) -> Dict[str, str]:
    """For each data-decision dimension, return the chosen variant or 'absent'."""
    decisions: Dict[str, str] = {}
    for dim, variants in DATA_DECISIONS.items():
        chosen = "absent"
        for variant, pat in variants.items():
            if variant == "none":
                continue
            if re.search(pat, text, re.IGNORECASE):
                chosen = variant
                break
        # Special case for scaling: if no scaler matched, mark "none" if MLP/SVM
        # appeared (those usually need scaling, so absence is itself a choice)
        if dim == "scaling" and chosen == "absent":
            if re.search(r"MLP|SVC|SVM|StandardScaler|MinMaxScaler", text):
                chosen = "none"
        decisions[dim] = chosen
    return decisions


def detect_method_choices(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dim, variants in METHOD_CHOICES.items():
        chosen = "absent"
        for variant, pat in variants.items():
            if re.search(pat, text, re.IGNORECASE):
                chosen = variant
                break
        out[dim] = chosen
    return out


def score_continuity(text: str) -> Tuple[float, Dict[str, int]]:
    """Criterion 8: continuity across steps.

    Penalize redefinitions of canonical variables. A variable defined more than
    once (`X_train = ...` appearing 3 times) suggests later steps re-derived it
    from scratch instead of building on the prior step.
    """
    redefs: Dict[str, int] = {}
    for v in VARIABLE_NAMES:
        # `name =` not preceded by `==` and not part of a larger token
        n = len(re.findall(rf"\b{re.escape(v)}\s*=(?!=)", text))
        if n > 0:
            redefs[v] = n
    n_keys = sum(1 for v in redefs if redefs[v] >= 1)
    if n_keys == 0:
        return 0.0, redefs
    n_overdef = sum(1 for v, n in redefs.items() if n > 1)
    # 0 overdefined → 5; everything overdefined → 1
    if n_keys == 0:
        score = 0.0
    else:
        score = 5.0 - 4.0 * (n_overdef / n_keys)
    return round(max(1.0, min(5.0, score)), 2), redefs


def score_transparency(text: str) -> Tuple[float, Dict[str, int]]:
    """Criterion 9: transparency / auditability.

    Density of explanation relative to code: counts comments, prose between
    code blocks, and rationale markers ("because", "we chose", "in order to").
    """
    code_chars = sum(len(b) for b in re.findall(r"```.*?```", text, re.DOTALL))
    code_chars += sum(len(b) for b in re.findall(r"<pre>.*?</pre>", text, re.DOTALL))
    inline_comments = len(re.findall(r"^\s*#[^\n]+", text, re.MULTILINE))
    rationale = len(re.findall(
        r"\b(?:because|in order to|so that|we (?:choose|chose|use|used|will|prefer)|"
        r"this allows|we need|to (?:ensure|make|avoid))\b",
        text, re.IGNORECASE,
    ))
    total_chars = max(1, len(text))
    prose_chars = total_chars - code_chars
    prose_ratio = prose_chars / total_chars
    # Composite signal
    signal = (
        prose_ratio  # 0..1
        + min(1.0, inline_comments / 20.0)
        + min(1.0, rationale / 10.0)
    )
    score = round(max(0.0, min(5.0, signal * 5.0 / 3.0)), 2)
    return score, {
        "inline_comments": inline_comments,
        "rationale_markers": rationale,
        "prose_ratio": round(prose_ratio, 3),
    }


def score_one_run(path: Path) -> Dict[str, Any]:
    text = _read_run(path)
    if not text:
        return {"run": path.name, "error": "empty"}
    steps, n_errors = detect_step_completion(text)

    # Were there errors near the GridSearchCV section? (rough heuristic)
    grid_idx = text.lower().find("gridsearchcv")
    errors_in_step10_window = False
    if grid_idx >= 0:
        window = text[grid_idx: grid_idx + 4000]
        errors_in_step10_window = any(
            re.search(p, window, re.IGNORECASE) for p in ERROR_PATTERNS
        )

    completion = score_completion(steps, n_errors)
    statefulness, var_counts = score_statefulness(text)
    tuning, tuning_evidence = score_tuning(text, errors_in_step10_window)
    continuity, redefs = score_continuity(text)
    transparency, trans_evidence = score_transparency(text)
    data_decisions = detect_data_decisions(text)
    method_choices = detect_method_choices(text)

    return {
        "run": path.name,
        "n_chars": len(text),
        "n_errors": n_errors,
        "steps_completed": sum(steps),
        "step_flags": "".join("1" if s else "0" for s in steps),
        # Numerical scores
        "score_3_completion": completion,
        "score_4_statefulness": statefulness,
        "score_5_tuning": tuning,
        "score_8_continuity": continuity,
        "score_9_transparency": transparency,
        # Decisions used for stability scoring (6, 7) across runs
        **{f"data_{k}": v for k, v in data_decisions.items()},
        **{f"method_{k}": v for k, v in method_choices.items()},
        # Evidence for QA
        "var_counts": json.dumps(var_counts),
        "redefinitions": json.dumps(redefs),
        "tuning_evidence": json.dumps(tuning_evidence),
        "transparency_evidence": json.dumps(trans_evidence),
    }


# ===========================================================================
# Stability scores (cross-run)
# ===========================================================================
def stability_score(values: List[str]) -> Tuple[float, Dict[str, int]]:
    """Higher score = lower variance.

    A criterion gets 5 if every run made the same decision and 1 if every
    run made a different decision. Linear interpolation between.
    """
    values = [v for v in values if v and v != "absent"]
    if not values:
        return 0.0, {}
    counts = Counter(values)
    most_common_n = counts.most_common(1)[0][1]
    consistency = most_common_n / len(values)  # in [1/k, 1]
    score = round(1.0 + 4.0 * consistency, 2)
    return score, dict(counts)


def compute_stability_scores(per_run: List[Dict[str, Any]]
                             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data_dims = list(DATA_DECISIONS.keys())
    method_dims = list(METHOD_CHOICES.keys())

    data_summary: Dict[str, Any] = {}
    method_summary: Dict[str, Any] = {}

    for dim in data_dims:
        values = [r.get(f"data_{dim}", "absent") for r in per_run]
        score, counts = stability_score(values)
        data_summary[dim] = {"score": score, "counts": counts}

    for dim in method_dims:
        values = [r.get(f"method_{dim}", "absent") for r in per_run]
        score, counts = stability_score(values)
        method_summary[dim] = {"score": score, "counts": counts}

    # Aggregate to single number per criterion (mean over its sub-dimensions)
    s6 = round(statistics.mean([d["score"] for d in data_summary.values()]), 2) \
        if data_summary else 0.0
    s7 = round(statistics.mean([d["score"] for d in method_summary.values()]), 2) \
        if method_summary else 0.0

    return (
        {"score_6_stability_data_decisions": s6, "by_dimension": data_summary},
        {"score_7_stability_method_choices": s7, "by_dimension": method_summary},
    )


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results",
                    help="Directory containing run_XX.md files")
    ap.add_argument("--out", default="./rubric_scores",
                    help="Output directory for CSVs")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    runs = sorted(results_dir.glob("run_*.md"))
    if not runs:
        print(f"[rubric] No run_*.md files in {results_dir}")
        return
    print(f"[rubric] scoring {len(runs)} runs from {results_dir}")

    per_run = [score_one_run(p) for p in runs]
    per_run = [r for r in per_run if "error" not in r]

    # Write per-run CSV
    if per_run:
        keys: List[str] = []
        seen = set()
        for r in per_run:
            for k in r.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        with open(out_dir / "per_run.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in per_run:
                w.writerow(r)
        print(f"[rubric] {out_dir / 'per_run.csv'}: {len(per_run)} rows")

    # Stability + summary
    data_sum, method_sum = compute_stability_scores(per_run)

    summary_rows = []
    score_cols = [
        ("3_completion", "score_3_completion"),
        ("4_statefulness", "score_4_statefulness"),
        ("5_tuning", "score_5_tuning"),
        ("8_continuity", "score_8_continuity"),
        ("9_transparency", "score_9_transparency"),
    ]
    for label, col in score_cols:
        vals = [r[col] for r in per_run if r.get(col) is not None]
        if not vals:
            continue
        summary_rows.append({
            "criterion": label,
            "n_runs": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        })
    summary_rows.append({
        "criterion": "6_stability_data_decisions",
        "n_runs": len(per_run),
        "mean": data_sum["score_6_stability_data_decisions"],
        "median": data_sum["score_6_stability_data_decisions"],
        "stdev": 0.0,
        "min": data_sum["score_6_stability_data_decisions"],
        "max": data_sum["score_6_stability_data_decisions"],
    })
    summary_rows.append({
        "criterion": "7_stability_method_choices",
        "n_runs": len(per_run),
        "mean": method_sum["score_7_stability_method_choices"],
        "median": method_sum["score_7_stability_method_choices"],
        "stdev": 0.0,
        "min": method_sum["score_7_stability_method_choices"],
        "max": method_sum["score_7_stability_method_choices"],
    })

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["criterion", "n_runs", "mean", "median",
                           "stdev", "min", "max"],
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"[rubric] {out_dir / 'summary.csv'}: {len(summary_rows)} rows")

    # Detailed JSON
    with open(out_dir / "run_details.json", "w") as f:
        json.dump({
            "per_run": per_run,
            "stability_data": data_sum,
            "stability_method": method_sum,
            "summary": summary_rows,
        }, f, indent=2, default=str)
    print(f"[rubric] {out_dir / 'run_details.json'}")

    # Print headline table
    print("\n=== RUBRIC SUMMARY (1–5) ===")
    print(f"{'criterion':<40} {'mean':>6} {'stdev':>7}")
    print("-" * 56)
    for r in summary_rows:
        print(f"{r['criterion']:<40} {r['mean']:>6} {r['stdev']:>7}")


if __name__ == "__main__":
    main()