"""
run_lambda.py
-------------
Runs the LAMBDA agentic workflow N times, saves each run's output to
results/run_XX.md, and logs overall progress to results/run.log.

Usage:
    python run_lambda.py              # 50 runs (default)
    python run_lambda.py --runs 10   # custom count
    python run_lambda.py --start 23  # resume from run 23
"""
from gradio_client import Client
import httpx
import argparse
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these paths to match your environment
# ---------------------------------------------------------------------------
LAMBDA_DIR  = "/Users/carter/Desktop/Prof_Zheng_Project/LAMBDA"
SERVER_URL  = "http://127.0.0.1:8000"
SERVER_BOOT_WAIT = 50   # Increased for stability
RUN_COOLDOWN     = 5
COMPLETION_TIMEOUT = 300  # Increased because DSGym loading takes time

PROMPT = """
1. Load the dataset and perform any necessary preprocessing.
2. Perform exploratory data analysis to understand key patterns in the data.
3. Split the data into training and test sets with a proportion of 8:2 (do not fix the random_state).
4. Select relevant features for modeling.
5. Train a logistic regression model and evaluate it by accuracy on the test set.
6. Train an SVM model and evaluate it by accuracy on the test set.
7. Train an MLP model and evaluate it by accuracy on the test set.
8. Train a decision tree model and evaluate it by accuracy on the test set.
9. Train a random forest model and evaluate it by accuracy on the test set.
10. Use GridSearchCV to find the best hyperparameters for the MLP model (five parameter groups).
11. Save all trained models.
12. Generate a report summarizing the analysis.
"""

COMPLETION_SIGNALS = [
    "Executing result:",
    "display_link:",
    "Test set accuracy",
    "Accuracy on test set",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "run.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
log_lines: list[str] = []


def _stream_logs(proc: subprocess.Popen) -> None:
    """Background thread: read server stdout into log_lines."""
    for line in iter(proc.stdout.readline, b""):
        log_lines.append(line.decode().rstrip())


def start_server() -> subprocess.Popen:
    log.info("Starting LAMBDA server...")
    venv_python = "/Users/carter/Desktop/Prof_Zheng_Project/DSGym/.venv/bin/python"
    proc = subprocess.Popen(
        [venv_python, "lambda_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=LAMBDA_DIR,
    )
    t = threading.Thread(target=_stream_logs, args=(proc,), daemon=True)
    t.start()
    time.sleep(SERVER_BOOT_WAIT)
    log.info("Server ready.")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    log.info("Shutting down LAMBDA server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.info("Server stopped.")


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------
def build_client():
    client = None
    for attempt in range(5):
        try:
            log.info(f"Connecting to server (Attempt {attempt+1}/5)...")
            client = Client(SERVER_URL)
            break
        except Exception:
            log.info("Server booting... waiting 5s")
            time.sleep(5)
    
    if not client:
        raise ConnectionError("Could not connect to LAMBDA server.")

    log.info("LAMBDA Client connected. Using DSGym internal datasets.")
    return client


def wait_for_completion(start_index: int, timeout: int = COMPLETION_TIMEOUT) -> bool:
    """Poll log_lines until a completion signal appears or timeout is reached."""
    for elapsed in range(timeout):
        run_logs = log_lines[start_index:]
        if any(sig in line for line in run_logs for sig in COMPLETION_SIGNALS):
            time.sleep(2)  # let remaining log lines flush
            return True
        time.sleep(1)
    log.warning("Timeout after %ds — moving on.", timeout)
    return False


def save_run_output(run_number: int, start_index: int, timed_out: bool) -> None:
    """Write the server logs for this run to a markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_logs = log_lines[start_index:]

    status = "TIMED OUT" if timed_out else "✅ Complete"
    md = (
        f"# LAMBDA Analysis Report — Run {run_number}\n"
        f"**Generated:** {timestamp}  \n"
        f"**Status:** {status}\n\n"
        "---\n\n"
        "```\n"
        + "\n".join(run_logs)
        + "\n```\n"
    )

    out_path = RESULTS_DIR / f"run_{run_number:02d}.md"
    out_path.write_text(md, encoding="utf-8")
    log.info("Run %d output saved → %s", run_number, out_path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_all(total_runs: int, start_from: int) -> None:
    proc = start_server()

    try:
        client = build_client()

        for i in range(start_from - 1, total_runs):
            run_number = i + 1
            log.info("=" * 60)
            log.info("RUN %d of %d", run_number, total_runs)
            log.info("=" * 60)

            log_start = len(log_lines)

            # Stage 1 — orchestrator plans the pipeline
            job1 = client.submit(
                message=PROMPT,
                chat_history=[],
                code=None,
                api_name="/chat_streaming",
            )
            updated_chat = job1.result()[1]

            # Stage 2 — programmer/inspector execute
            job2 = client.submit(
                chat_history_display=updated_chat,
                code=None,
                api_name="/stream_workflow",
            )
            for _ in job2:
                pass  # drain the stream

            log.info("Waiting for analysis to complete...")
            timed_out = not wait_for_completion(log_start)

            save_run_output(run_number, log_start, timed_out)

            if i < total_runs - 1:
                log.info("Cooling down for %ds before next run...", RUN_COOLDOWN)
                time.sleep(RUN_COOLDOWN)

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
    finally:
        stop_server(proc)
        log.info("All done. Results in: %s/", RESULTS_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LAMBDA agent N times.")
    parser.add_argument("--runs",  type=int, default=3, help="Total number of runs (default: 50)")
    parser.add_argument("--start", type=int, default=1,  help="Resume from this run number (default: 1)")
    args = parser.parse_args()

    if args.start > args.runs:
        parser.error("--start cannot be greater than --runs")

    run_all(total_runs=args.runs, start_from=args.start)