"""
Drift taxonomy + agent divergence analyzer for the task-intent drift paper.

Reads all DSGym evaluation result JSONs under ./evaluation_results/ produced by
main.py and emits CSVs under ./drift_analysis/ summarizing:

Per sample (per_sample.csv)
  Drift category, scalars, nudge events, plan/code alignment, drift onset,
  per-trial bookkeeping.

Aggregations
  category_counts.csv       drift category × nudge_mode crosstab
  recovery_rate.csv         recovery rate by mode and trigger reason
  divergence.csv            verbal-vs-execution disagreement summary
  consistency.csv           per-sample stability across trials
  agent_divergence.csv      orchestrator-vs-programmer disagreement stats
  drift_trajectory.csv      one row per (sample, trial, turn) — feeds line graph
  drift_onset.csv           per-sample drift onset (turn / tokens / sec) summary
  trial_summary.csv         strict + lenient accuracy per (mode, trial)

Optional: if a Gemini-judge JSON exists with the same run_name + "_gemini.json"
suffix, lenient labels are used instead of strict labels.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

RESULTS_DIR = Path("./evaluation_results")
OUT_DIR = Path("./drift_analysis")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _find_result_files() -> List[Path]:
    files: List[Path] = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        name = p.name
        if "_gemini" in name or "_judged" in name or name.endswith("_metrics.json"):
            continue
        files.append(p)
    return files


def _load_gemini_labels(run_name: str) -> Dict[str, bool]:
    candidates = (
        list(RESULTS_DIR.glob(f"{run_name}*gemini*.json"))
        + list(RESULTS_DIR.glob(f"{run_name}*judged*.json"))
    )
    for cand in candidates:
        try:
            with open(cand) as f:
                data = json.load(f)
        except Exception:
            continue
        labels: Dict[str, bool] = {}
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        if isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                key = str(item.get("sample_id", item.get("id", i)))
                # Prefer Gemini verdict if present
                if isinstance(item.get("gemini_judgment"), dict):
                    v = str(item["gemini_judgment"].get("verdict", "")).lower()
                    if v in {"correct", "incorrect"}:
                        labels[key] = (v == "correct")
                        continue
                if "correct" in item:
                    labels[key] = bool(item["correct"])
                elif "lenient_correct" in item:
                    labels[key] = bool(item["lenient_correct"])
        if labels:
            return labels
    return {}


def _iter_samples(result_doc: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(result_doc, dict):
        for key in ("predictions", "samples", "results", "data"):
            if key in result_doc and isinstance(result_doc[key], list):
                yield from result_doc[key]
                return
        if "metadata" in result_doc or "solution" in result_doc:
            yield result_doc
            return
    elif isinstance(result_doc, list):
        yield from result_doc


def _get_run_name(path: Path, doc: Any) -> str:
    if isinstance(doc, dict):
        cfg = doc.get("config") or {}
        if isinstance(cfg, dict) and cfg.get("run_name"):
            return cfg["run_name"]
        if doc.get("run_name"):
            return doc["run_name"]
    return path.stem.replace("_results", "")


def _correct_label(sample: Dict[str, Any],
                   gemini_labels: Dict[str, bool],
                   sample_idx: int) -> Optional[bool]:
    sample_id = str(
        sample.get("sample_id") or sample.get("id") or sample.get("idx", sample_idx)
    )
    if sample_id in gemini_labels:
        return gemini_labels[sample_id]
    em = sample.get("metrics", {})
    if isinstance(em, dict):
        v = em.get("exact_match")
        if isinstance(v, dict):
            try:
                return float(v.get("score", 0)) >= 1.0
            except (TypeError, ValueError):
                return False
        if isinstance(v, (int, float, bool)):
            return float(v) >= 0.5
    for key in ("exact_match", "is_correct", "correct", "score"):
        if key in sample:
            v = sample[key]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v >= 0.5
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_sample(sample: Dict[str, Any], correct: Optional[bool]) -> str:
    meta = (sample.get("metadata") or {})
    drift = meta.get("drift") or {}
    if drift.get("crashed"):
        return "crash"
    if drift.get("verbal_execution_disagree") is True:
        return "verbal_execution_divergence"
    nudged = bool(drift.get("nudge_triggered"))
    if correct is True and not nudged:
        return "clean"
    if correct is True and nudged:
        return "early_exit_recovered"
    if correct is False and nudged:
        return "early_exit_failed"
    if correct is False and not nudged:
        return "other_failure"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-sample row builder
# ---------------------------------------------------------------------------
def build_per_sample_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = _find_result_files()
    print(f"[drift] found {len(files)} result files")

    for path in files:
        try:
            with open(path) as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[drift] skip {path.name}: {e}")
            continue

        run_name = _get_run_name(path, doc)
        gemini = _load_gemini_labels(run_name)

        for i, sample in enumerate(_iter_samples(doc)):
            if not isinstance(sample, dict):
                continue
            meta = sample.get("metadata") or {}
            drift = meta.get("drift") or {}
            correct = _correct_label(sample, gemini, i)
            category = classify_sample(sample, correct)

            sample_id = (
                sample.get("sample_id") or sample.get("id")
                or sample.get("idx") or i
            )
            extra = sample.get("extra_info") or {}
            agent_turns = drift.get("agent_turns") or []

            rows.append({
                "run_name": run_name,
                "sample_id": str(sample_id),
                "trial_id": drift.get("trial_id"),
                "nudge_mode": drift.get("nudge_mode"),
                "correct": correct,
                "category": category,
                # Nudge stats
                "nudge_triggered": drift.get("nudge_triggered"),
                "nudge_count": drift.get("nudge_count"),
                "answer_detected_at_turn": drift.get("answer_detected_at_turn"),
                "crashed": drift.get("crashed"),
                # Verbal-execution
                "execution_scalar": drift.get("execution_scalar"),
                "verbal_scalar": drift.get("verbal_scalar"),
                "verbal_execution_disagree": drift.get("verbal_execution_disagree"),
                # Plan / O→P
                "plan_item_count": len(drift.get("plan_items") or []),
                "plan_code_alignment": drift.get("plan_code_alignment"),
                "o_p_disagreement": drift.get("o_p_disagreement"),
                # Drift onset
                "drift_onset_turn": drift.get("drift_onset_turn"),
                "drift_onset_tokens": drift.get("drift_onset_tokens"),
                "drift_onset_sec": drift.get("drift_onset_sec"),
                "re_anchor_turn": drift.get("re_anchor_turn"),
                "re_anchor_tokens": drift.get("re_anchor_tokens"),
                # Counts useful for sanity-checking
                "n_turns": len(agent_turns),
                "total_tokens": (agent_turns[-1]["cumulative_tokens"]
                                 if agent_turns else None),
                "elapsed_sec": meta.get("elapsed_sec"),
                # Optional dataset metadata
                "question_type": extra.get("question_type"),
                "difficulty": extra.get("difficulty"),
                # First nudge reason
                "nudge_reason": (
                    drift.get("nudge_events", [{}])[0].get("reason")
                    if drift.get("nudge_events") else None
                ),
            })
    return rows


def build_trajectory_rows() -> List[Dict[str, Any]]:
    """One row per (run, sample, turn) — feeds the drift line graph."""
    rows: List[Dict[str, Any]] = []
    for path in _find_result_files():
        try:
            with open(path) as f:
                doc = json.load(f)
        except Exception:
            continue
        run_name = _get_run_name(path, doc)
        for i, sample in enumerate(_iter_samples(doc)):
            if not isinstance(sample, dict):
                continue
            meta = sample.get("metadata") or {}
            drift = meta.get("drift") or {}
            sample_id = str(
                sample.get("sample_id") or sample.get("id")
                or sample.get("idx") or i
            )
            for t in (drift.get("agent_turns") or []):
                rows.append({
                    "run_name": run_name,
                    "sample_id": sample_id,
                    "trial_id": drift.get("trial_id"),
                    "nudge_mode": drift.get("nudge_mode"),
                    "turn_index": t.get("turn_index"),
                    "phase": t.get("phase"),
                    "elapsed_sec": t.get("elapsed_sec"),
                    "phase_sec": t.get("phase_sec"),
                    "tokens_in_turn": t.get("tokens_in_turn"),
                    "cumulative_tokens": t.get("cumulative_tokens"),
                    "task_anchoring": t.get("task_anchoring"),
                    "has_code": t.get("has_code"),
                    "has_execution_output": t.get("has_execution_output"),
                    "has_inspector_signals": t.get("has_inspector_signals"),
                })
    return rows


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]],
              fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("")
        print(f"[drift] {path}: 0 rows")
        return
    if fieldnames is None:
        # Stable union of keys, preserving first-seen order
        seen: List[str] = []
        seen_set: set = set()
        for r in rows:
            for k in r.keys():
                if k not in seen_set:
                    seen.append(k)
                    seen_set.add(k)
        fieldnames = seen
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[drift] {path}: {len(rows)} rows")


def category_counts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str], int] = Counter()
    for r in rows:
        counts[(r["nudge_mode"] or "?", r["category"])] += 1
    modes = sorted({r["nudge_mode"] or "?" for r in rows})
    cats = ["clean", "early_exit_recovered", "early_exit_failed",
            "verbal_execution_divergence", "crash", "other_failure", "unknown"]
    out = []
    for cat in cats:
        row: Dict[str, Any] = {"category": cat}
        total = 0
        for mode in modes:
            n = counts[(mode, cat)]
            row[mode] = n
            total += n
        row["total"] = total
        out.append(row)
    totals: Dict[str, Any] = {"category": "TOTAL"}
    grand = 0
    for mode in modes:
        n = sum(counts[(mode, c)] for c in cats)
        totals[mode] = n
        grand += n
    totals["total"] = grand
    out.append(totals)
    return out


def recovery_rate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
    for r in rows:
        if not r.get("nudge_triggered"):
            continue
        mode = r["nudge_mode"] or "?"
        reason = r.get("nudge_reason") or "unknown"
        recovered = r["category"] == "early_exit_recovered"
        bucket[(mode, reason)].append(recovered)
    out = []
    for (mode, reason), vals in sorted(bucket.items()):
        if not vals:
            continue
        out.append({
            "nudge_mode": mode,
            "nudge_reason": reason,
            "n_nudged": len(vals),
            "n_recovered": sum(vals),
            "recovery_rate": round(sum(vals) / len(vals), 3),
        })
    by_mode: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        if r.get("nudge_triggered"):
            by_mode[r["nudge_mode"] or "?"].append(
                r["category"] == "early_exit_recovered"
            )
    for mode, vals in sorted(by_mode.items()):
        if vals:
            out.append({
                "nudge_mode": mode,
                "nudge_reason": "ALL",
                "n_nudged": len(vals),
                "n_recovered": sum(vals),
                "recovery_rate": round(sum(vals) / len(vals), 3),
            })
    return out


def divergence_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mode: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "n_with_both_scalars": 0,
        "n_disagree": 0,
        "n_disagree_correct": 0,
    })
    for r in rows:
        mode = r["nudge_mode"] or "?"
        if (r.get("execution_scalar") is None
                or r.get("verbal_scalar") is None):
            continue
        by_mode[mode]["n_with_both_scalars"] += 1
        if r.get("verbal_execution_disagree"):
            by_mode[mode]["n_disagree"] += 1
            if r.get("correct") is True:
                by_mode[mode]["n_disagree_correct"] += 1
    out = []
    for mode, d in sorted(by_mode.items()):
        n = d["n_with_both_scalars"]
        out.append({
            "nudge_mode": mode,
            "n_with_both_scalars": n,
            "n_disagree": d["n_disagree"],
            "disagreement_rate": round(d["n_disagree"] / n, 3) if n else 0.0,
            "n_disagree_correct": d["n_disagree_correct"],
        })
    return out


def consistency_across_trials(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        bucket[(r["nudge_mode"] or "?", r["sample_id"])].append(r)
    out = []
    for (mode, sid), trials in sorted(bucket.items()):
        if len(trials) < 2:
            continue
        correct = [t.get("correct") for t in trials]
        cats = sorted({t["category"] for t in trials})
        out.append({
            "nudge_mode": mode,
            "sample_id": sid,
            "n_trials": len(trials),
            "n_correct": sum(1 for c in correct if c is True),
            "frac_correct": round(
                sum(1 for c in correct if c is True) / len(trials), 3
            ),
            "categories_observed": "|".join(cats),
            "category_count": len(cats),
            "stable": len(cats) == 1,
        })
    return out


# ---------------------------------------------------------------------------
# NEW: agent-level divergence stats
# ---------------------------------------------------------------------------
def agent_divergence_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Orchestrator vs programmer (and inspector-proxy) disagreement.

    For each nudge_mode, reports:
      - mean / median plan_code_alignment (O→P agreement)
      - mean / median o_p_disagreement   (1 - alignment)
      - inspector_signal_rate            fraction of workflow turns that
                                         contained inspector-style language
      - mean turns per sample
      - mean tokens per sample
    """
    by_mode: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Pull per-turn inspector-signal info from the trajectory rows
    traj = build_trajectory_rows()
    inspector_by_sample: Dict[Tuple[str, str], Tuple[int, int]] = defaultdict(
        lambda: (0, 0)  # (n_workflow_turns, n_with_inspector_signal)
    )
    for t in traj:
        if t["phase"] != "workflow":
            continue
        key = ((t.get("nudge_mode") or "?"), t["sample_id"])
        n, k = inspector_by_sample[key]
        inspector_by_sample[key] = (
            n + 1,
            k + (1 if t.get("has_inspector_signals") else 0),
        )

    for r in rows:
        mode = r["nudge_mode"] or "?"
        if r.get("plan_code_alignment") is not None:
            by_mode[mode]["alignment"].append(float(r["plan_code_alignment"]))
            by_mode[mode]["disagreement"].append(float(r["o_p_disagreement"]))
        if r.get("n_turns") is not None:
            by_mode[mode]["n_turns"].append(float(r["n_turns"]))
        if r.get("total_tokens") is not None:
            by_mode[mode]["total_tokens"].append(float(r["total_tokens"]))
        n, k = inspector_by_sample.get((mode, r["sample_id"]), (0, 0))
        if n > 0:
            by_mode[mode]["inspector_signal_rate"].append(k / n)

    def _stats(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {"n": 0, "mean": None, "median": None, "stdev": None}
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3),
            "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        }

    out = []
    for mode, d in sorted(by_mode.items()):
        a = _stats(d["alignment"])
        dis = _stats(d["disagreement"])
        ins = _stats(d["inspector_signal_rate"])
        nt = _stats(d["n_turns"])
        tt = _stats(d["total_tokens"])
        out.append({
            "nudge_mode": mode,
            "n_samples_with_plan": a["n"],
            "alignment_mean": a["mean"],
            "alignment_median": a["median"],
            "o_p_disagreement_mean": dis["mean"],
            "o_p_disagreement_median": dis["median"],
            "inspector_signal_rate_mean": ins["mean"],
            "inspector_signal_rate_median": ins["median"],
            "n_turns_mean": nt["mean"],
            "total_tokens_mean": tt["mean"],
        })
    return out


def drift_onset_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drift onset distribution per nudge_mode (turn / token / second axes)."""
    by_mode: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        mode = r["nudge_mode"] or "?"
        if r.get("drift_onset_turn") is not None:
            by_mode[mode]["onset_turn"].append(float(r["drift_onset_turn"]))
        if r.get("drift_onset_tokens") is not None:
            by_mode[mode]["onset_tokens"].append(float(r["drift_onset_tokens"]))
        if r.get("drift_onset_sec") is not None:
            by_mode[mode]["onset_sec"].append(float(r["drift_onset_sec"]))
        # Total samples in mode (for drift_rate denominator)
        by_mode[mode]["_all"].append(1.0)

    def _stats(xs: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not xs:
            return None, None, None
        m = round(statistics.mean(xs), 2)
        med = round(statistics.median(xs), 2)
        sd = round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0
        return m, med, sd

    out = []
    for mode, d in sorted(by_mode.items()):
        n_total = len(d["_all"])
        n_drifted = len(d["onset_turn"])
        m_t, med_t, sd_t = _stats(d["onset_turn"])
        m_k, med_k, sd_k = _stats(d["onset_tokens"])
        m_s, med_s, sd_s = _stats(d["onset_sec"])
        out.append({
            "nudge_mode": mode,
            "n_samples": n_total,
            "n_drifted": n_drifted,
            "drift_rate": round(n_drifted / n_total, 3) if n_total else 0.0,
            "onset_turn_mean": m_t,
            "onset_turn_median": med_t,
            "onset_turn_stdev": sd_t,
            "onset_tokens_mean": m_k,
            "onset_tokens_median": med_k,
            "onset_tokens_stdev": sd_k,
            "onset_sec_mean": m_s,
            "onset_sec_median": med_s,
            "onset_sec_stdev": sd_s,
        })
    return out


def trial_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_run: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[(r["nudge_mode"] or "?", r.get("trial_id"))].append(r)
    out = []
    for (mode, trial), batch in sorted(by_run.items(),
                                       key=lambda kv: (kv[0][0], str(kv[0][1]))):
        n = len(batch)
        n_correct = sum(1 for r in batch if r.get("correct") is True)
        n_nudged = sum(1 for r in batch if r.get("nudge_triggered"))
        n_recovered = sum(1 for r in batch if r["category"] == "early_exit_recovered")
        n_crash = sum(1 for r in batch if r.get("crashed"))
        out.append({
            "nudge_mode": mode,
            "trial_id": trial,
            "n_samples": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 3) if n else 0.0,
            "n_nudged": n_nudged,
            "nudge_rate": round(n_nudged / n, 3) if n else 0.0,
            "n_recovered_by_nudge": n_recovered,
            "recovery_rate": (round(n_recovered / n_nudged, 3)
                              if n_nudged else 0.0),
            "n_crash": n_crash,
        })
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    rows = build_per_sample_rows()
    if not rows:
        print("[drift] No rows produced. Did main.py run successfully?")
        return

    write_csv(OUT_DIR / "per_sample.csv", rows)
    write_csv(OUT_DIR / "category_counts.csv", category_counts(rows))
    write_csv(OUT_DIR / "recovery_rate.csv", recovery_rate(rows))
    write_csv(OUT_DIR / "divergence.csv", divergence_summary(rows))
    write_csv(OUT_DIR / "consistency.csv", consistency_across_trials(rows))
    write_csv(OUT_DIR / "agent_divergence.csv", agent_divergence_summary(rows))
    write_csv(OUT_DIR / "drift_onset.csv", drift_onset_summary(rows))
    write_csv(OUT_DIR / "trial_summary.csv", trial_summary(rows))
    write_csv(OUT_DIR / "drift_trajectory.csv", build_trajectory_rows())

    # Headline stats
    cats = Counter(r["category"] for r in rows)
    print("\n=== DRIFT TAXONOMY ===")
    for cat, n in cats.most_common():
        print(f"  {cat:<30} {n:>4}")
    print(f"  {'TOTAL':<30} {sum(cats.values()):>4}")

    both = [r for r in rows if r.get("execution_scalar") is not None
            and r.get("verbal_scalar") is not None]
    disagree = [r for r in both if r.get("verbal_execution_disagree")]
    if both:
        print("\n=== VERBAL-EXECUTION DIVERGENCE ===")
        print(f"  samples with both scalars: {len(both)}")
        print(f"  disagreements:             {len(disagree)} "
              f"({100 * len(disagree) / len(both):.1f}%)")

    aligned = [r for r in rows if r.get("plan_code_alignment") is not None]
    if aligned:
        mean_align = statistics.mean(r["plan_code_alignment"] for r in aligned)
        print("\n=== ORCHESTRATOR → PROGRAMMER ===")
        print(f"  samples with extracted plan: {len(aligned)}")
        print(f"  mean plan-code alignment:    {mean_align:.3f}")
        print(f"  mean O→P disagreement:       {1 - mean_align:.3f}")

    drifted = [r for r in rows if r.get("drift_onset_turn") is not None]
    if drifted:
        print("\n=== DRIFT ONSET ===")
        print(f"  samples that drifted:           {len(drifted)}/{len(rows)} "
              f"({100 * len(drifted) / len(rows):.1f}%)")
        print(f"  mean onset turn:                "
              f"{statistics.mean(r['drift_onset_turn'] for r in drifted):.2f}")
        toks = [r["drift_onset_tokens"] for r in drifted
                if r.get("drift_onset_tokens") is not None]
        if toks:
            print(f"  mean onset tokens:              {statistics.mean(toks):.0f}")

    print(f"\nResults written to {OUT_DIR}/")
    print("Next: python drift_plots.py")


if __name__ == "__main__":
    main()