"""
Single-stage DSGym evaluation of LAMBDA, in-process (no Gradio).

Run with:
    python -u main.py

Writes results to ./evaluation_results/qrdata_lambda_inproc_results.json,
which gemini_judge.py reads in the post-processing step.
"""
import os
import sys
import json
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"


def log(msg: str) -> None:
    print(msg, flush=True)


log("[main] starting")

from dsgym.datasets import DatasetRegistry
log("[main] DatasetRegistry imported")

from dsgym.eval import Evaluator, EvaluationConfig
log("[main] Evaluator imported")

from lambda_dsgym_wrapper import LambdaDSGymAgent
log("[main] wrapper imported")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME = "qrdata"
LIMIT        = 50
OUTPUT_DIR   = "./evaluation_results"
RUN_NAME     = f"{DATASET_NAME}_lambda_inproc"

# ---------------------------------------------------------------------------
log("[main] constructing agent")
agent = LambdaDSGymAgent()
log("[main] agent constructed")

log(f"[main] loading dataset {DATASET_NAME}")
dataset = DatasetRegistry.load(DATASET_NAME)
samples = dataset.load(limit=LIMIT)
log(f"[main] loaded {len(samples)} sample(s)")

evaluator = Evaluator(
    protocol="multi_turn",
    dataset=dataset,
    parallel_workers=1,
)

config = EvaluationConfig(
    model_name="lambda",
    backend_type="lambda_inproc",
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
)

log("[main] running evaluator")
results = evaluator.evaluate(
    agent=agent,
    tasks=samples,
    config=config,
    save_results=True,
)

results_path = Path(f"{OUTPUT_DIR}/{RUN_NAME}_results.json")
with open(results_path) as f:
    saved = json.load(f)

drift_by_idx = {i: agent._drift_cache[i]["drift"] 
                for i in range(len(agent._drift_cache))}

for i, sample in enumerate(saved):
    sample["metadata"] = {"drift": drift_by_idx.get(i, {})}

with open(results_path, "w") as f:
    json.dump(saved, f, indent=2, default=str)

print("[main] drift metadata re-attached")

m = results.get("metrics", {})
log(
    f"[main] DSGym strict score: "
    f"{m.get('exact_match_mean', 0):.2f} "
    f"({int((m.get('exact_match_mean') or 0) * m.get('total_samples', 0))}"
    f"/{m.get('total_samples', 0)})"
)
log(f"[main] Results saved under: {OUTPUT_DIR}/{RUN_NAME}_*.json")
log("[main] Next step: run `python gemini_judge.py` to re-judge failures with Gemini")