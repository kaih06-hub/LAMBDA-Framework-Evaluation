"""
Headless LAMBDA wrapper for DSGym, instrumented for task-intent drift research.

What the wrapper now captures (in metadata.drift):
  Trial-level
    trial_id, nudge_mode

  Sample-level
    orchestrator_intent       turn-0 plan from the orchestrator
    plan_items                numbered/bulleted items extracted from the plan
    code_actions              functions, modules, and names from programmer code
    plan_code_alignment       fraction of plan items covered by code (0..1)
    o_p_disagreement          1 - plan_code_alignment

    nudge_triggered, nudge_count, nudge_events, crashed, crash_message,
    answer_detected_at_turn

    execution_output, verbal_text         split halves of the final response
    execution_scalar, verbal_scalar       extracted numeric answers
    verbal_execution_disagree             True/False/None

  Per-turn drift trajectory (agent_turns list)
    Each entry: {turn_index, phase, elapsed_sec, tokens_in_turn,
                 cumulative_tokens, has_code, has_execution_output,
                 has_inspector_signals, task_anchoring, text_preview}

  Drift onset (computed from agent_turns)
    drift_onset_turn, drift_onset_tokens, drift_onset_sec
    re_anchor_turn, re_anchor_tokens (if recovery happened)

Configuration (set once before evaluator.evaluate):
  agent.set_trial_config(trial_id=0, nudge_mode="targeted")

Nudge modes:
  - "targeted": re-inject the specific question (default)
  - "generic":  retry without question text (ablation control)
  - "none":     single-turn, no retry (baseline)
"""
from __future__ import annotations

import os
import re
import html as _html
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dsgym.agents.base_agent import BaseAgent

# ---------------------------------------------------------------------------
LAMBDA_DIR = Path("/Users/carter/Desktop/Prof_Zheng_Project/LAMBDA")
LAMBDA_CONFIG = "config.yaml"

if str(LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(LAMBDA_DIR))

PER_SAMPLE_TIMEOUT = 300
MAX_RETRY_TURNS = 3

# Drift onset threshold: a turn is considered "drifted" when its task-anchoring
# score drops below DRIFT_RATIO * initial_anchoring (i.e. the orchestrator's
# anchoring at turn 0). Tunable.
DRIFT_RATIO = 0.6


# ---------------------------------------------------------------------------
# Token counting (tiktoken if available, otherwise whitespace fallback)
# ---------------------------------------------------------------------------
try:
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # pragma: no cover
    def _count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))


# ---------------------------------------------------------------------------
# Lexical helpers used for plan extraction and task-anchoring
# ---------------------------------------------------------------------------
_STOP: Set[str] = set(
    "the a an and or but in on at to for of with by is are was were be been "
    "being has have had do does did this that these those it its as if then "
    "than from we you they i he she them his her their our your my me us "
    "not no so too very can could should would may might will shall just "
    "also any all some each both either neither into onto over under again "
    "more most less few many several other another such which who whom whose "
    "what when where why how here there now please".split()
)

_INSPECTOR_SIGNALS = re.compile(
    r"(?:successfully|looks?\s+(?:correct|good|right)|seems?\s+(?:fine|correct|right)"
    r"|completed?|verified|the\s+(?:result|code|output)|let\s+me|i(?:'ll|\s+will)\s+"
    r"|next(?:,)?\s+(?:i|we|you)|to\s+verify|appears?\s+(?:to|correct)"
    r"|error|exception|failed|fix|issue|problem|incorrect|wrong)",
    re.IGNORECASE,
)


def _log(msg: str) -> None:
    print(f"[wrapper] {msg}", file=sys.stderr, flush=True)


@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class _FileShim:
    def __init__(self, path: str):
        self.name = path


# ===========================================================================
class LambdaDSGymAgent(BaseAgent):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(backend="lambda_inproc", model="lambda")
        cfg = config_path or LAMBDA_CONFIG
        _log(f"Initializing LAMBDA (config={cfg})")
        with _chdir(LAMBDA_DIR):
            from LAMBDA import LAMBDA as LambdaCls
            self._lambda = LambdaCls(config_path=cfg)
        _log(f"LAMBDA ready. Session: {self._lambda.session_cache_path}")

        self._trial_id: int = 0
        self._nudge_mode: str = "targeted"
        self._drift_cache: List[Dict] = []

    # -----------------------------------------------------------------------
    def set_trial_config(self, trial_id: int = 0, nudge_mode: str = "targeted") -> None:
        if nudge_mode not in {"targeted", "generic", "none"}:
            raise ValueError(f"Invalid nudge_mode: {nudge_mode}")
        self._trial_id = trial_id
        self._nudge_mode = nudge_mode
        _log(f"Trial config: trial_id={trial_id}, nudge_mode={nudge_mode}")

    # =======================================================================
    # Main entry point
    # =======================================================================
    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        start = time.time()

        drift: Dict[str, Any] = {
            "trial_id": self._trial_id,
            "nudge_mode": self._nudge_mode,
            # Plan + alignment
            "orchestrator_intent": None,
            "plan_items": [],
            "code_actions": {"functions": [], "modules": [], "names": []},
            "plan_code_alignment": None,
            "o_p_disagreement": None,
            # Nudge / failure flags
            "nudge_triggered": False,
            "nudge_count": 0,
            "nudge_events": [],
            "crashed": False,
            "crash_message": None,
            "answer_detected_at_turn": None,
            # Verbal-vs-execution
            "execution_output": None,
            "verbal_text": None,
            "execution_scalar": None,
            "verbal_scalar": None,
            "verbal_execution_disagree": None,
            # Per-turn trajectory
            "agent_turns": [],
            # Drift onset / recovery (filled at end)
            "drift_onset_turn": None,
            "drift_onset_tokens": None,
            "drift_onset_sec": None,
            "re_anchor_turn": None,
            "re_anchor_tokens": None,
        }

        try:
            with _chdir(LAMBDA_DIR):
                self._lambda.clear_all("", [])

                files = self._resolve_sample_files(sample)
                for f in files:
                    _log(f"Uploading {f.name}")
                    self._lambda.add_file(_FileShim(str(f)))

                prompt = self._build_prompt(sample, [f.name for f in files])
                _log(f"Prompt: {len(prompt)} chars")

                question = self._extract_question(sample)
                context = (sample.get("extra_info", {}) or {}).get("context", "")
                task_kw = self._task_keywords(question, context)

                # ===================================================
                # Stage 1: orchestrator
                # ===================================================
                t_phase = time.time()
                _, chat = self._lambda.chat_streaming(prompt, [])
                orch_text = self._last_assistant_text(chat)

                drift["orchestrator_intent"] = orch_text
                drift["plan_items"] = self._extract_plan_items(orch_text)

                orch_tokens = _count_tokens(orch_text)
                drift["agent_turns"].append({
                    "turn_index": 0,
                    "phase": "orchestrator",
                    "elapsed_sec": round(time.time() - start, 2),
                    "phase_sec": round(time.time() - t_phase, 2),
                    "tokens_in_turn": orch_tokens,
                    "cumulative_tokens": orch_tokens,
                    "has_code": bool(re.search(r"```|<pre>", orch_text)),
                    "has_execution_output": "<pre>" in orch_text,
                    "has_inspector_signals": bool(_INSPECTOR_SIGNALS.search(
                        re.sub(r"<[^>]+>", " ", orch_text or "")
                    )),
                    "task_anchoring": self._task_anchoring(orch_text, task_kw),
                    "text_preview": (orch_text or "")[:1000],
                })

                # ===================================================
                # Stage 2: programmer + inspector loop, with optional nudges
                # ===================================================
                final_chat = chat
                response = ""
                turns_used = 0
                cum_tokens = orch_tokens
                max_turns = 1 if self._nudge_mode == "none" else MAX_RETRY_TURNS

                for turn in range(1, max_turns + 1):
                    if time.time() >= start + PER_SAMPLE_TIMEOUT:
                        _log("Timeout reached before retry.")
                        break

                    _log(f"stream_workflow turn {turn}/{max_turns}...")
                    t_phase = time.time()
                    final_chat, crashed = self._drain_workflow(
                        final_chat, deadline=start + PER_SAMPLE_TIMEOUT
                    )
                    turns_used = turn

                    if crashed:
                        drift["crashed"] = True
                        drift["crash_message"] = "stream_workflow raised exception"

                    response = self._last_assistant_text(final_chat)

                    # Capture this turn
                    turn_tokens = _count_tokens(response)
                    cum_tokens += turn_tokens
                    text_no_html = re.sub(r"<[^>]+>", " ", response or "")
                    drift["agent_turns"].append({
                        "turn_index": turn,
                        "phase": "workflow",
                        "elapsed_sec": round(time.time() - start, 2),
                        "phase_sec": round(time.time() - t_phase, 2),
                        "tokens_in_turn": turn_tokens,
                        "cumulative_tokens": cum_tokens,
                        "has_code": bool(re.search(r"```|def\s+\w+|import\s+\w+", response)),
                        "has_execution_output": "<pre>" in response,
                        "has_inspector_signals": bool(_INSPECTOR_SIGNALS.search(text_no_html)),
                        "task_anchoring": self._task_anchoring(response, task_kw),
                        "text_preview": (response or "")[:1000],
                    })

                    if self._has_computed_answer(response):
                        drift["answer_detected_at_turn"] = turn
                        _log(f"Answer detected after turn {turn}.")
                        for ev in drift["nudge_events"]:
                            if ev.get("recovered") is None:
                                ev["recovered"] = True
                        break

                    if turn < max_turns:
                        nudge_text = self._build_nudge(question)
                        if nudge_text is None:
                            _log(f"nudge_mode={self._nudge_mode}, skipping nudge")
                            break
                        reason = self._nudge_reason(response)
                        drift["nudge_triggered"] = True
                        drift["nudge_count"] += 1
                        drift["nudge_events"].append({
                            "turn": turn,
                            "reason": reason,
                            "mode": self._nudge_mode,
                            "recovered": None,
                        })
                        _log(f"Sending {self._nudge_mode} nudge (turn {turn}, reason={reason})")
                        _, final_chat = self._lambda.chat_streaming(nudge_text, final_chat)

                for ev in drift["nudge_events"]:
                    if ev.get("recovered") is None:
                        ev["recovered"] = False

                # ===================================================
                # Post-processing: extract code, alignment, drift onset
                # ===================================================
                full_program_text = "\n\n".join(
                    t["text_preview"] for t in drift["agent_turns"]
                    if t["phase"] == "workflow"
                )
                drift["code_actions"] = self._extract_code_actions(full_program_text)
                alignment = self._plan_code_alignment(
                    drift["plan_items"], drift["code_actions"]
                )
                drift["plan_code_alignment"] = alignment
                drift["o_p_disagreement"] = (
                    None if alignment is None else round(1.0 - alignment, 3)
                )

                # Drift onset: first turn where anchoring drops below threshold
                turns = drift["agent_turns"]
                if len(turns) >= 2:
                    initial = turns[0]["task_anchoring"] or 0.0
                    threshold = DRIFT_RATIO * initial if initial > 0 else 0.0
                    for t in turns[1:]:
                        if (t["task_anchoring"] or 0.0) < threshold:
                            drift["drift_onset_turn"] = t["turn_index"]
                            drift["drift_onset_tokens"] = t["cumulative_tokens"]
                            drift["drift_onset_sec"] = t["elapsed_sec"]
                            break
                    # Recovery: a later turn climbs back above threshold
                    if drift["drift_onset_turn"] is not None:
                        for t in turns:
                            if (t["turn_index"] > drift["drift_onset_turn"]
                                    and (t["task_anchoring"] or 0.0) >= threshold):
                                drift["re_anchor_turn"] = t["turn_index"]
                                drift["re_anchor_tokens"] = t["cumulative_tokens"]
                                break

                # Verbal vs execution
                execution_output, verbal_text = self._split_execution_verbal(response)
                drift["execution_output"] = execution_output[:5000] if execution_output else ""
                drift["verbal_text"] = verbal_text[:5000] if verbal_text else ""
                drift["execution_scalar"] = self._extract_scalar(execution_output, prefer="last")
                drift["verbal_scalar"] = self._extract_scalar(verbal_text, prefer="answer_phrase")
                if (drift["execution_scalar"] is not None
                        and drift["verbal_scalar"] is not None):
                    drift["verbal_execution_disagree"] = not self._scalars_agree(
                        drift["execution_scalar"], drift["verbal_scalar"]
                    )

                elapsed = time.time() - start
                _log(
                    f"Done in {elapsed:.1f}s, {turns_used} turn(s), "
                    f"plan_items={len(drift['plan_items'])}, "
                    f"o_p_disagree={drift['o_p_disagreement']}, "
                    f"drift_onset_turn={drift['drift_onset_turn']}, "
                    f"verbal_exec_disagree={drift['verbal_execution_disagree']}"
                )

                extracted = drift["execution_scalar"] or drift["verbal_scalar"]
                solution = str(round(extracted, 2)) if extracted is not None else response
            
                self._drift_cache.append({
                    "sample_id": sample.get("id") or sample.get("sample_id"),
                    "drift": drift
                })
                
                return {
                    "solution": solution,
                    "success": bool(response),
                    "turns": turns_used,
                    "error": None,
                    "metadata": {
                        "model": self.model,
                        "backend": self.backend,
                        "elapsed_sec": elapsed,
                        "drift": drift,
                    },
                    "conversation": self._chat_to_messages(final_chat),
                    "raw_result": response,
                }

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"solve_task failed: {type(e).__name__}: {e}\n{tb}")
            drift["crashed"] = True
            drift["crash_message"] = f"{type(e).__name__}: {e}"
            return {
                "solution": "",
                "success": False,
                "turns": 0,
                "error": f"{type(e).__name__}: {e}",
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "traceback": tb,
                    "drift": drift,
                },
                "conversation": [],
                "raw_result": None,
            }

    # =======================================================================
    # Workflow execution
    # =======================================================================
    def _drain_workflow(self, chat, deadline: float) -> Tuple[Any, bool]:
        gen = self._lambda.conv.stream_workflow(chat, code=None)
        last = chat
        crashed = False
        try:
            for step in gen:
                last = step if step is not None else last
                if time.time() > deadline:
                    _log(f"workflow exceeded {PER_SAMPLE_TIMEOUT}s; stopping")
                    try:
                        gen.close()
                    except Exception:
                        pass
                    break
        except Exception as e:
            _log(f"stream_workflow raised: {type(e).__name__}: {e}")
            crashed = True
        return last, crashed

    # =======================================================================
    # Nudge construction
    # =======================================================================
    def _build_nudge(self, question: str) -> Optional[str]:
        if self._nudge_mode == "none":
            return None
        if self._nudge_mode == "generic":
            return (
                "Please re-read the task and try again. Write and execute "
                "Python code that produces the requested value."
            )
        return (
            f'You have not yet fully answered this question: "{question}"\n'
            "Please re-read the question carefully, then write and execute "
            "Python code that directly computes the requested value."
        )

    @staticmethod
    def _nudge_reason(response: str) -> str:
        if not response:
            return "empty_response"
        text = _html.unescape(re.sub(r"<[^>]+>", " ", response))
        if len(text.strip()) < 250:
            return "too_short"
        if re.search(
            r"(?:next(?:,)?\s+you\s+can|the\s+next\s+step\s+is"
            r"|let'?s\s+(?:now|proceed|run))",
            text[-500:], re.IGNORECASE
        ):
            return "ends_with_next_steps"
        return "no_scalar_or_answer_phrase"

    # =======================================================================
    # Answer detection
    # =======================================================================
    @staticmethod
    def _has_computed_answer(response: str) -> bool:
        if not response:
            return False
        text = _html.unescape(re.sub(r"<[^>]+>", " ", response))
        if len(text.strip()) < 250:
            return False
        if re.search(r"np\.(?:float|int)(?:32|64)?", response):
            return True
        for block in re.findall(r"<pre>(.*?)</pre>", response, re.DOTALL | re.IGNORECASE):
            block_text = _html.unescape(re.sub(r"<[^>]+>", "", block)).strip()
            lines = [l.strip() for l in block_text.splitlines() if l.strip()]
            if lines and re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", lines[-1]):
                return True
        nouns = (
            r"proportion|percentage|percent|median|mean|average|count|number|iqr"
            r"|answer|result|value|total|correlation|coefficient|p[-\s]?value"
            r"|probability|ratio|range|difference|sum|min(?:imum)?|max(?:imum)?"
            r"|standard\s+deviation|variance|slope|intercept|estimate|rate"
        )
        if re.search(
            rf"(?:the\s+)?(?:{nouns})\s*(?:is|of|=|:|was|are)\s*[+-]?\d",
            text, re.IGNORECASE
        ):
            return True
        tail = text[-800:]
        if re.search(r"(?:answer|final|conclusion)[^.]{0,80}?[+-]?\d+(?:\.\d+)?",
                     tail, re.IGNORECASE):
            return True
        if re.search(r"FINAL_ANSWER", text, re.IGNORECASE):
            return True
        if re.search(
            r"(?:next(?:,)?\s+you\s+can|the\s+next\s+step\s+is"
            r"|let'?s\s+(?:now|proceed|run))",
            text[-500:], re.IGNORECASE
        ):
            return False
        return False

    # =======================================================================
    # Plan + code action extraction (for O→P disagreement)
    # =======================================================================
    @staticmethod
    def _extract_plan_items(text: str) -> List[str]:
        if not text:
            return []
        clean = _html.unescape(re.sub(r"<[^>]+>", " ", text))
        items: List[str] = []
        # Numbered: "1. Foo", "1) Foo", "Step 1: Foo"
        for m in re.finditer(
            r"(?:^|\n)\s*(?:step\s+)?\d+\s*[.):-]\s*(.+?)(?=\n|$)",
            clean, re.IGNORECASE,
        ):
            item = m.group(1).strip()
            if 5 <= len(item) <= 300:
                items.append(item)
        if items:
            return items
        # Bulleted
        for m in re.finditer(r"(?:^|\n)\s*[-*•]\s+(.+?)(?=\n|$)", clean):
            item = m.group(1).strip()
            if 5 <= len(item) <= 300:
                items.append(item)
        if items:
            return items
        # Action-verb sentences as fallback
        action_verbs = (
            r"load|read|import|compute|calculate|filter|group|merge|plot|train|"
            r"split|encode|fit|predict|evaluate|measure|count|sort|select|drop|"
            r"answer|determine|find|identify|return|print"
        )
        for m in re.finditer(
            rf"\b(?:{action_verbs})\b[^.\n]{{3,200}}\.", clean, re.IGNORECASE,
        ):
            items.append(m.group(0).strip())
        return items[:30]

    @staticmethod
    def _extract_code_actions(text: str) -> Dict[str, List[str]]:
        actions = {"functions": set(), "modules": set(), "names": set()}
        if not text:
            return {k: [] for k in actions}
        # Code blocks (```python ... ``` and <pre>...</pre>)
        blocks: List[str] = re.findall(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
        blocks += [
            _html.unescape(re.sub(r"<[^>]+>", "", b))
            for b in re.findall(r"<pre>(.*?)</pre>", text, re.DOTALL | re.IGNORECASE)
        ]
        code = "\n".join(blocks)
        for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code):
            actions["functions"].add(m.group(1))
        for m in re.finditer(r"\b([a-z_][a-zA-Z0-9_]*)\.[a-zA-Z_]", code):
            actions["modules"].add(m.group(1))
        for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=(?!=)", code):
            actions["names"].add(m.group(1))
        return {k: sorted(v) for k, v in actions.items()}

    @staticmethod
    def _plan_code_alignment(plan: List[str],
                             code_actions: Dict[str, List[str]]) -> Optional[float]:
        if not plan:
            return None
        terms: Set[str] = set()
        for vals in code_actions.values():
            terms.update(v.lower() for v in vals)
        if not terms:
            return 0.0
        aligned = 0
        for item in plan:
            words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", item.lower())
            meaningful = [w for w in words if w not in _STOP]
            covered = any(
                w in terms or any(w in t or t in w for t in terms)
                for w in meaningful
            )
            if covered:
                aligned += 1
        return round(aligned / len(plan), 3)

    # =======================================================================
    # Task-anchoring (per-turn drift signal)
    # =======================================================================
    @staticmethod
    def _task_keywords(question: str, context: str = "") -> Set[str]:
        words = re.findall(
            r"[a-zA-Z][a-zA-Z0-9_]{2,}",
            f"{question or ''} {context or ''}".lower(),
        )
        return {w for w in words if w not in _STOP}

    @staticmethod
    def _task_anchoring(text: str, keywords: Set[str]) -> float:
        if not text or not keywords:
            return 0.0
        clean = _html.unescape(re.sub(r"<[^>]+>", " ", text)).lower()
        # Token-set jaccard-like: how many keywords appear?
        hits = sum(1 for k in keywords if k in clean)
        return round(hits / max(1, len(keywords)), 3)

    # =======================================================================
    # Verbal vs execution split + scalar extraction
    # =======================================================================
    @staticmethod
    def _split_execution_verbal(response: str) -> Tuple[str, str]:
        if not response:
            return "", ""
        pre_blocks = re.findall(r"<pre>(.*?)</pre>", response, re.DOTALL | re.IGNORECASE)
        execution_output = "\n".join(
            _html.unescape(re.sub(r"<[^>]+>", "", b)).strip() for b in pre_blocks
        ).strip()
        verbal = re.sub(r"<pre>.*?</pre>", " ", response,
                        flags=re.DOTALL | re.IGNORECASE)
        verbal = _html.unescape(re.sub(r"<[^>]+>", " ", verbal))
        verbal = re.sub(r"\s+", " ", verbal).strip()
        return execution_output, verbal

    @staticmethod
    def _extract_scalar(text: str, prefer: str = "answer_phrase") -> Optional[float]:
        if not text:
            return None
        if prefer == "last":
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for line in reversed(lines):
                m = re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", line)
                if m:
                    try:
                        return float(m.group(0))
                    except ValueError:
                        pass
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text)
            if nums:
                try:
                    return float(nums[-1])
                except ValueError:
                    return None
            return None
        for pat in [
            r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            r"(?:the\s+result\s+is|result\s*[:=])\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            r"(?:approximately|about|roughly)\s+([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        nouns = (
            r"proportion|percentage|median|mean|average|correlation|coefficient"
            r"|p[-\s]?value|probability|ratio|standard\s+deviation|variance"
            r"|slope|intercept|count|number|total|estimate|rate"
        )
        m = re.search(
            rf"(?:{nouns})[^.\d]{{0,40}}?([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            text, re.IGNORECASE,
        )
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                return None
        return None

    @staticmethod
    def _scalars_agree(a: float, b: float, rtol: float = 0.02, atol: float = 1e-6) -> bool:
        if a == b:
            return True
        if abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1e-9):
            return True
        for scale in (100.0, 0.01):
            if abs(a * scale - b) <= atol + rtol * max(abs(a * scale), abs(b), 1e-9):
                return True
        return False

    # =======================================================================
    # Sample helpers
    # =======================================================================
    @staticmethod
    def _extract_question(sample: Dict[str, Any]) -> str:
        extra = sample.get("extra_info", {}) or {}
        question = (extra.get("question") or "").strip()
        if question:
            return question
        prompt_field = sample.get("prompt")
        if isinstance(prompt_field, list):
            for msg in reversed(prompt_field):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "").strip()
        return ""

    @staticmethod
    def _resolve_sample_files(sample: Dict[str, Any]) -> List[Path]:
        extra = sample.get("extra_info", {}) or {}
        data_files = extra.get("data_files")
        if not data_files:
            return []
        if isinstance(data_files, dict):
            paths = (
                data_files.get("absolute")
                or data_files.get("virtual")
                or data_files.get("relative")
                or []
            )
        elif isinstance(data_files, list):
            paths = data_files
        else:
            paths = [data_files]
        return [Path(p) for p in paths if Path(p).exists()]

    @staticmethod
    def _build_prompt(sample: Dict[str, Any], filenames: List[str]) -> str:
        extra = sample.get("extra_info", {}) or {}
        question = (extra.get("question") or "").strip()
        context = (extra.get("context") or "").strip()
        if not question:
            prompt_field = sample.get("prompt")
            if isinstance(prompt_field, list):
                for msg in reversed(prompt_field):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg.get("content", "").strip()
                        break
        parts: List[str] = []
        if context:
            parts.append(f"Background: {context}")
        if filenames:
            parts.append(
                "The following data file(s) have been uploaded to your working "
                "directory: " + ", ".join(filenames)
                + ". Load them with pandas using the absolute path shown in "
                "the upload message above."
            )
        parts.append(f"Task: {question}")
        parts.append(
            "Please write and execute Python code to answer the question, "
            "then clearly state your final answer."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _last_assistant_text(chat) -> str:
        if not chat:
            return ""
        try:
            last_pair = chat[-1]
            if isinstance(last_pair, (list, tuple)) and len(last_pair) >= 2:
                return str(last_pair[1] or "")
        except Exception:
            pass
        return str(chat)

    @staticmethod
    def _chat_to_messages(chat) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if not isinstance(chat, list):
            return messages
        for item in chat:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user, assistant = item[0], item[1]
                if user is not None:
                    messages.append({"role": "user", "content": str(user)})
                if assistant is not None:
                    messages.append({"role": "assistant", "content": str(assistant)})
            elif isinstance(item, dict) and "role" in item:
                messages.append(item)
        return messages