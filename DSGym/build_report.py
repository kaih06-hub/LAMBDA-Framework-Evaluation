"""
build_report.py
---------------
Reads:  ./evaluation_results/qrdata_lambda_inproc_judged.json
Writes: ./evaluation_results/qrdata_lambda_inproc_report.md
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path("./evaluation_results")
RUN_NAME    = "qrdata_lambda_inproc"
INPUT_PATH  = RESULTS_DIR / f"{RUN_NAME}_judged.json"
OUTPUT_PATH = RESULTS_DIR / f"{RUN_NAME}_report.md"


def log(msg: str) -> None:
    print(f"[report] {msg}", flush=True)


# ---------------------------------------------------------------------------
def is_correct(r: Dict[str, Any]) -> bool:
    em = r.get("metrics", {}).get("exact_match", {})
    try:
        return float(em.get("score", 0)) >= 1.0
    except (TypeError, ValueError):
        return False


def gemini_verdict(r: Dict[str, Any]) -> str:
    j = r.get("gemini_judgment")
    if not j:
        return "not judged"
    return j.get("verdict", "unclear")


def clean_response(text: str) -> str:
    """Readable plain text from LAMBDA's HTML-heavy output."""
    if not text:
        return "_(no response)_"
    text = html.unescape(text)

    # Unwrap <details><pre>...</pre></details> into a labelled block
    def unwrap_details(m: re.Match) -> str:
        inner = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        return f"\n**Execution output:**\n```\n{inner}\n```\n" if inner else ""

    text = re.sub(
        r"<details[^>]*>.*?<pre>(.*?)</pre>.*?</details>",
        unwrap_details, text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<button[^>]*>.*?</button>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        log(f"ERROR: {INPUT_PATH} not found — run gemini_judge.py first.")
        return

    with INPUT_PATH.open() as f:
        results: List[Dict[str, Any]] = json.load(f)

    n = len(results)
    strict  = sum(1 for r in results if is_correct(r))
    rescued = sum(
        1 for r in results
        if not is_correct(r) and gemini_verdict(r) == "correct"
    )
    lenient = strict + rescued

    out: List[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    out.append("# LAMBDA on DSGym — Evaluation Report")
    out.append("")
    out.append("| Metric | Count | Rate |")
    out.append("|---|---|---|")
    out.append(f"| Total samples | {n} | |")
    out.append(f"| Strict correct (DSGym exact_match) | {strict} | {strict/n:.0%} |")
    out.append(f"| Rescued by Gemini | {rescued} | {rescued/n:.0%} |")
    out.append(f"| **Lenient correct** | **{lenient}** | **{lenient/n:.0%}** |")
    out.append(f"| Confirmed wrong | {n - lenient} | {(n-lenient)/n:.0%} |")
    out.append("")
    out.append("---")
    out.append("")

    # ── One block per sample ─────────────────────────────────────────────────
    for i, r in enumerate(results):
        sid      = r.get("sample_id", f"sample_{i}")
        question = r.get("query", "")
        gt       = r.get("ground_truth", "")
        raw      = r.get("raw_result") or r.get("raw_response") or ""
        j        = r.get("gemini_judgment") or {}
        verdict  = j.get("verdict", "not judged")
        confidence = j.get("confidence", "")
        reasoning  = j.get("reasoning", "")
        extracted  = j.get("extracted_answer", "")

        # verdict icon
        if is_correct(r):
            icon = "✅"
        elif verdict == "correct":
            icon = "⚠️"
        elif verdict == "incorrect":
            icon = "❌"
        else:
            icon = "🔍"

        out.append(f"## {icon} Sample {i+1}")
        out.append(f"**ID:** `{sid}`")
        out.append("")

        # 1. Task
        out.append("### Task")
        out.append(f"**Question:** {question}")
        out.append("")
        out.append(f"**Ground truth:** `{gt}`")
        out.append("")

        # 2. LAMBDA's full response
        out.append("### LAMBDA's Response")
        out.append("")
        cleaned = clean_response(raw)
        out.append(cleaned)
        out.append("")

        # 3. Gemini verdict
        out.append("### Gemini Verdict")
        if not j:
            out.append("_Not judged (DSGym marked this correct)._")
        else:
            out.append(f"**Verdict:** `{verdict}` ({confidence} confidence)")
            if extracted:
                out.append(f"**Answer Gemini found in response:** `{extracted}`")
            out.append(f"**Reasoning:** {reasoning}")
        out.append("")
        out.append("---")
        out.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(out), encoding="utf-8")
    log(f"Wrote {OUTPUT_PATH}")
    log(f"Strict {strict}/{n} | Lenient {lenient}/{n}")


if __name__ == "__main__":
    main()