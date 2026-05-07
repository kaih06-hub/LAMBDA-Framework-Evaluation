# Task-Intent Drift in LAMBDA — Research Project

> **Research question:** Does the orchestrator→programmer agent boundary in LAMBDA introduce measurable task-intent drift on the QRData benchmark, and how much of LAMBDA's apparent failure rate is recoverable by a permissive second judge?

---

## Project Overview

This project instruments [LAMBDA](https://arxiv.org/abs/2401.17626) — a multi-agent data-science LLM system — and evaluates it on the [QRData](https://arxiv.org/abs/2402.01843) split of the [DSGym](https://github.com/DSGym) benchmark. It introduces per-turn drift instrumentation, a strict/lenient dual-grading scheme, and a suite of visualisations to characterise where and how multi-agent intent transmission breaks down.

**Key findings (n = 50):**
- Strict accuracy: **70%** (35/50) — depressed entirely by output-formatting mismatch
- Lenient accuracy: **100%** (50/50) — after LLaMA-3 second-judge re-rating
- **84%** of samples required a nudge to produce an answer
- **46%** of samples classified as `early_exit_recovered` drift
- Apparent 43.8% verbal–execution disagreement rate found to be a measurement artifact on inspection

---

## Repository Structure

```
.
├── main.py                        # Entry point — runs LAMBDA on DSGym samples
├── lambda_dsgym_wrapper.py        # Instrumented LAMBDA↔DSGym adapter (drift capture)
├── gemini_judge.py                # Second-judge re-rater (LLaMA-3 / Gemini)
├── lambda_drift_analysis.ipynb    # Full analysis notebook (all figures + tables)
│
├── evaluation_results/            # Created by main.py and gemini_judge.py
│   ├── qrdata_lambda_inproc_results.json    # Raw LAMBDA outputs + drift metadata
│   └── qrdata_lambda_inproc_judged.json     # With judge_result field added
│
├── drift_analysis/                # Created by the notebook
│   ├── figures/                   # All publication-ready PNGs
│   │   ├── fig_category_breakdown.png
│   │   ├── fig_drift_trajectory.png
│   │   ├── fig_o_p_disagreement.png
│   │   ├── fig_recovery_by_reason.png
│   │   ├── fig_verbal_vs_execution.png
│   │   └── fig_drift_accuracy_curves.png
│   ├── per_sample.csv             # One row per sample, all drift fields
│   ├── drift_trajectory.csv       # One row per agent turn
│   ├── session_timeline.csv       # Panel A data (sample index vs accuracy)
│   ├── drift_rows.csv             # Per-sample drift signals
│   ├── category_counts.csv        # Taxonomy breakdown
│   └── recovery_rate.csv          # Nudge recovery by trigger reason
│
└── lambda_drift_report.tex        # LaTeX report (Overleaf-ready)
```

---

## Pipeline

The full pipeline is **three steps** in sequence:

```
main.py  →  gemini_judge.py  →  lambda_drift_analysis.ipynb
```

### Step 1 — Run LAMBDA on DSGym

```bash
python -u main.py
```

Runs LAMBDA in-process on 50 QRData samples. Writes:
- `evaluation_results/qrdata_lambda_inproc_results.json` — raw results with `metadata.drift` re-attached

**What this does internally:**
1. Boots LAMBDA without its Gradio UI
2. Clears session state between samples
3. Sends each question + data file reference to LAMBDA
4. Captures per-turn instrumentation (anchoring, tokens, code flags) into `metadata.drift`
5. Issues targeted nudges (re-injecting the original question) if no answer is detected
6. Returns the extracted scalar (or full HTML fallback) as DSGym's `solution` field

### Step 2 — Re-judge strict failures

```bash
# Using local LLaMA (default, no API key needed)
export MODEL_PROVIDER=llama
python gemini_judge.py

# Using Gemini (requires API key)
export MODEL_PROVIDER=gemini
export GEMINI_API_KEY=<your_key>
python gemini_judge.py
```

Reads the results JSON, identifies the 15 strict failures, and re-judges each with LLaMA-3 (via Ollama) or Gemini using a semantic-equivalence prompt with 3% relative tolerance. Writes:
- `evaluation_results/qrdata_lambda_inproc_judged.json` — with `judge_result` field on each failure

**LLaMA-3 setup (Ollama):**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
ollama serve   # runs on http://127.0.0.1:11434
```

### Step 3 — Run the analysis notebook

Open `lambda_drift_analysis.ipynb` in Jupyter. The notebook auto-detects `evaluation_results/` by walking up the directory tree from wherever the notebook is run.

Run all cells top-to-bottom. All figures display inline and are saved to `drift_analysis/figures/`.

---

## Instrumentation Details

### What `lambda_dsgym_wrapper.py` captures per turn

| Field | Description |
|---|---|
| `turn_index` | Absolute turn number within the sample |
| `phase` | `orchestrator` or `workflow` |
| `elapsed_sec` | Wall-clock seconds since sample start |
| `tokens_in_turn` | Tokens in this turn (tiktoken `cl100k_base`) |
| `cumulative_tokens` | Running total |
| `task_anchoring` | Fraction of question keywords present in this turn's text |
| `has_code` | Whether a code block was detected |
| `has_execution_output` | Whether `<pre>` output was present |
| `has_inspector_signals` | Whether inspector-style language was detected |

### Drift signals computed per sample

| Field | Description |
|---|---|
| `plan_items` | Numbered/bulleted steps from orchestrator turn 0 |
| `plan_code_alignment` | Fraction of plan items covered by programmer code actions |
| `o_p_disagreement` | `1 − plan_code_alignment` |
| `execution_scalar` | Last bare number in `<pre>` code output |
| `verbal_scalar` | Number nearest to answer phrase in prose |
| `verbal_execution_disagree` | Whether the two scalars differ beyond `rtol=0.02` |
| `drift_onset_turn` | First turn where anchoring drops below `0.6 × orchestrator anchoring` |
| `nudge_triggered` | Whether a targeted nudge was sent |
| `nudge_events` | List of `{reason, turn, mode, recovered}` dicts |

---

## Drift Taxonomy

Each sample is classified into one of six mutually exclusive categories:

| Category | Definition |
|---|---|
| `clean` | Lenient-correct on first pass, no nudge needed |
| `early_exit_recovered` | LAMBDA stopped early, nudge recovered the answer |
| `early_exit_failed` | LAMBDA stopped early, nudge failed |
| `verbal_execution_divergence` | Prose answer ≠ code output (by scalar comparison) |
| `crash` | Workflow raised an unhandled exception |
| `other_failure` | Wrong answer with no identifiable early-exit or crash signal |

---

## Configuration

Key hyperparameters (all in `lambda_dsgym_wrapper.py`):

```python
DRIFT_RATIO = 0.6          # anchoring threshold for drift onset (0.6 × orch. anchoring)
MAX_TURNS   = 3            # max programmer turns before giving up
TIMEOUT_SEC = 300          # per-sample wall-clock timeout
NUDGE_MODE  = "targeted"   # "targeted" re-injects question | "generic" does not
```

Grading tolerance (in `gemini_judge.py` prompt and scalar agreement):
- Relative tolerance: `rtol = 0.02` (2%)
- Percentage encoding: `0.20 ≈ 20%` is handled automatically

---

## Dependencies

```bash
# Core
pip install dsgym           # DSGym evaluation harness
pip install tiktoken        # Token counting
pip install pandas numpy matplotlib pillow

# For the notebook
pip install jupyter ipython

# Second judge — local (no API cost)
pip install requests        # Ollama HTTP client (built-in to gemini_judge.py)
# + install Ollama and pull llama3 (see Step 2 above)

# Second judge — Gemini (optional)
pip install google-genai
```

---

## Reproducing the Results

```bash
# 1. Clone and set up
git clone <repo>
cd <repo>
pip install -r requirements.txt

# 2. Install and start Ollama with LLaMA-3
ollama pull llama3
ollama serve &

# 3. Run the full pipeline
python -u main.py
python gemini_judge.py
jupyter notebook lambda_drift_analysis.ipynb
```

Expected outputs after running all three steps:
- `evaluation_results/qrdata_lambda_inproc_judged.json` — 50 samples, 35 strict-correct, 15 rescued
- `drift_analysis/figures/` — 6 PNG figures
- `drift_analysis/*.csv` — 6 tidy CSV exports

---

## Citation

If you use this instrumentation or the dual-grading scheme, please cite the underlying systems:

**DSGym:**
```
Fan Nie, Junlin Wang, Harper Hua, Federico Bianchi, Yongchan Kwon, Zhenting Qi,
Owen Queen, Shang Zhu, James Zou.
DSGym: A Holistic Framework for Evaluating and Training Data Science Agents.
Stanford University, TogetherAI, Duke University, Harvard University.
```

**LAMBDA:**
```
Maojun Sun, Ruijian Han, Binyan Jiang, Houduo Qi, Defeng Sun,
Yancheng Yuan, Jian Huang.
LAMBDA: A Large Model Based Data Agent.
Department of Applied Mathematics and Department of Data Science and
Artificial Intelligence, The Hong Kong Polytechnic University.
```

---

## Notes

- **Plan extraction is regex-based.** In the current run, LAMBDA's orchestrator turn-0 output did not contain extractable numbered plans, so `plan_code_alignment` and `o_p_disagreement` are `NaN` for all 50 samples. The measurement infrastructure is retained for runs where the orchestrator emits explicit plans.
- **Verbal scalar extraction is fragile.** The 43.8% verbal–execution disagreement rate observed in this run is largely a measurement artifact: the prose-side extractor captures the wrong number (e.g., confidence level instead of the point estimate). The execution scalar matched ground truth in all 21 apparent disagreements. Do not interpret the raw disagreement rate as genuine LAMBDA failure without the manual audit table.
- **Single trial per sample.** No repeat trials were run; the cross-trial consistency section of the notebook requires `N_TRIALS ≥ 2` set in `main.py`.
