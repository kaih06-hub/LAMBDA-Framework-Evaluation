"""
gemini_judge.py
---------------
Re-judges DSGym failures using Gemini or local LLaMA (Ollama).

Usage:
    # Gemini
    export MODEL_PROVIDER=gemini
    export GEMINI_API_KEY=<your_key>

    # Local LLaMA (Ollama)
    export MODEL_PROVIDER=llama
    python gemini_judge.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
RESULTS_DIR = Path("./evaluation_results")
RUN_NAME = "qrdata_lambda_inproc"
INPUT_PATH = RESULTS_DIR / f"{RUN_NAME}_results.json"
OUTPUT_PATH = RESULTS_DIR / f"{RUN_NAME}_judged.json"

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "llama")

GEMINI_MODEL = "gemini-2.5-flash-lite"
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3")
LLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

BETWEEN_CALLS_DELAY_SEC = 15
MAX_RETRIES = 3
RETRY_BASE_DELAY_SEC = 35

RESPONSE_PREVIEW_CHARS = 6000


def log(msg: str) -> None:
    print(f"[judge] {msg}", flush=True)


# ---------------------------------------------------------------------------
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    try:
        from google import genai
    except ImportError:
        log("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def get_client():
    if MODEL_PROVIDER == "gemini":
        return get_gemini_client()
    elif MODEL_PROVIDER == "llama":
        return None  # no client needed for HTTP
    else:
        log(f"ERROR: Unknown MODEL_PROVIDER={MODEL_PROVIDER}")
        sys.exit(1)


def generate_response(client, prompt: str) -> str:
    if MODEL_PROVIDER == "gemini":
        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return result.text or ""

    elif MODEL_PROVIDER == "llama":
        import requests

        url = f"{LLAMA_BASE_URL}/chat/completions"

        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }

        resp = requests.post(url, json=payload)

        if resp.status_code != 200:
            raise Exception(f"Ollama error: {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unsupported provider: {MODEL_PROVIDER}")


# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are a careful grader for a data-analysis benchmark.

A model was asked a question, ran Python code against a dataset, and produced
a response that may include code, execution output, tables, and prose.

A strict string-matching evaluator says the response did NOT match the
expected answer. Your job is to decide whether the model's response actually
contains the correct answer — possibly buried in output, formatted as a
percentage, or phrased differently.

Equivalence rules:
- Numbers within ~3% relative tolerance are equivalent
- Categorical answers match if the meaning is the same
- If the model only shows code WITHOUT execution output, it has not answered

QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{ground_truth}

MODEL RESPONSE (may be truncated):
{response}

Reply with a JSON object on ONE line — no prose before or after:
{{"verdict": "correct"|"incorrect"|"unclear", "confidence": "high"|"medium"|"low", "extracted_answer": "<what the model said>", "reasoning": "<one sentence>"}}
"""


def build_prompt(question: str, ground_truth: str, response: str) -> str:
    if len(response) > RESPONSE_PREVIEW_CHARS:
        response = response[:RESPONSE_PREVIEW_CHARS] + "\n...[truncated]..."
    return JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        response=response,
    )


def parse_reply(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return _error_verdict(f"Could not parse JSON: {text[:200]}")
        else:
            return _error_verdict(f"No JSON found: {text[:200]}")
    return {
        "verdict": str(data.get("verdict", "unclear")).lower(),
        "confidence": str(data.get("confidence", "low")).lower(),
        "extracted_answer": str(data.get("extracted_answer", "")),
        "reasoning": str(data.get("reasoning", "")),
    }


def _error_verdict(reason: str) -> Dict[str, Any]:
    return {
        "verdict": "unclear",
        "confidence": "low",
        "extracted_answer": "",
        "reasoning": reason
    }


def judge_one(client, question: str, ground_truth: str, response: str,
              sample_idx: int, total: int) -> Dict[str, Any]:
    prompt = build_prompt(question, ground_truth, response)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = generate_response(client, prompt)
            return parse_reply(text)
        except Exception as e:
            err_str = str(e)
            log(f"  ERROR: {err_str}")
            delay = 5 * attempt
            log(f"  attempt {attempt}/{MAX_RETRIES} failed: waiting {delay}s...")
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    return _error_verdict(f"All {MAX_RETRIES} attempts failed: {err_str}")


# ---------------------------------------------------------------------------
def is_dsgym_correct(result: Dict[str, Any]) -> bool:
    em = result.get("metrics", {}).get("exact_match", {})
    if isinstance(em, dict):
        try:
            return float(em.get("score", 0)) >= 1.0
        except (TypeError, ValueError):
            return False
    else:
        try:
            return float(em) >= 1.0
        except (TypeError, ValueError):
            return False
    return False


def main() -> None:
    if not INPUT_PATH.exists():
        log(f"ERROR: {INPUT_PATH} not found — run main.py first.")
        sys.exit(1)

    with INPUT_PATH.open() as f:
        results: List[Dict[str, Any]] = json.load(f)

    failures = [r for r in results if not is_dsgym_correct(r)]
    log(f"Loaded {len(results)} samples — {len(failures)} failed DSGym strict matching")

    client = get_client()
    log(f"Using {MODEL_PROVIDER.upper()} model")

    for i, r in enumerate(results, 1):
        if is_dsgym_correct(r):
            r["judge_result"] = None
            continue

        verdict = judge_one(
            client,
            r.get("query", ""),
            str(r.get("ground_truth") or ""),
            str(r.get("raw_result") or r.get("raw_response") or r.get("prediction") or ""),
            i,
            len(results)
        )

        r["judge_result"] = verdict
        log(f"[{i}/{len(results)}] → {verdict['verdict']} ({verdict['confidence']})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))

    log(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()