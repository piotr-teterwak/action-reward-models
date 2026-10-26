"""Frontier (GPT-5.x) catts_vision action selector for OpenWebRL rollouts.

Web-flavor port of the browser_agents catts_vision arbiter (the recipe behind
the OM2W selection-ARM numbers), applied over OpenWebRL's own agent: the policy
samples N candidates per step (temp override), a frontier judge picks one from
the CURRENT screenshot + task + action history + candidates, explain-then-JSON
so the live decision carries its own rationale (for the trajectory explorer).

CLUSTER_NO_DOM semantics: no dedup/merging — all N raw samples are shown as
independent candidates every step. Any failure -> candidate 0.

Env knobs (read here and in generate_browser._maybe_arbitered_inference):
    OWRL_ARBITER=1            enable
    OWRL_ARBITER_N            candidates/step (default 5)
    OWRL_ARBITER_TEMP         candidate sampling temperature (default 0.7)
    OWRL_ARBITER_MODEL        judge model (default gpt-5.5)
    OWRL_ARBITER_LOG_DIR      per-task sidecar JSONL dir (optional)
Judge auth: OPENAI_API_KEY (already exported by the SCC launcher).
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from openai import AsyncOpenAI

_SELECTION_RE = re.compile(r'\{\s*"selection"\s*:\s*(\d+)\s*\}')
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_POINT_RE = re.compile(r'("point_2d"\s*:\s*\[)\s*(\d+)\s*,\s*(\d+)\s*(\])')


def _image_size(image_b64: str):
    try:
        import base64 as _b64, io as _io
        from PIL import Image as _Image
        s = image_b64.split("base64,", 1)[1] if "base64," in str(image_b64) else image_b64
        with _Image.open(_io.BytesIO(_b64.b64decode(s))) as im:
            return im.width, im.height
    except Exception:
        return None


def to_pixel_display(action: str, w: int, h: int) -> str:
    """Rewrite [0,1000]-normalized point_2d values to PIXEL coords of the
    attached screenshot — DISPLAY ONLY (frontier judges are pixel-native;
    the executed action keeps the policy's normalized values)."""
    def repl(m):
        x = round(int(m.group(2)) / 1000 * w)
        y = round(int(m.group(3)) / 1000 * h)
        return f'{m.group(1)}{x}, {y}{m.group(4)}'
    return _POINT_RE.sub(repl, action)

SYSTEM_PROMPT = (
    "You are a web automation arbiter. Pick the candidate action most likely "
    "to advance the user's task.\n\n"
    "First explain your choice in 2-4 sentences, then on a new line output "
    "exactly: {\"selection\": <integer N matching a candidate number>}"
)

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    """OWRL_ARBITER_URL lets a locally-served judge (e.g. a trained selection
    ARM on vLLM) replace the OpenAI API; OWRL_ARBITER_API_KEY overrides the key
    (local servers take anything)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=os.environ.get("OWRL_ARBITER_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OWRL_ARBITER_API_KEY")
                    or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    return _client


def split_think(response_text: str) -> Tuple[str, str]:
    """model response -> (thought, action_text). Action = everything after
    </think> (verbatim — it is what the env executes)."""
    t = response_text or ""
    m = _THINK_RE.search(t)
    if m:
        thought = " ".join(m.group(1).split())[:2000]
        action = (t[m.end():] or "").strip()
    elif "</think>" in t:
        # chat template prefills the opening <think>: response starts mid-think
        before, after = t.split("</think>", 1)
        thought = " ".join(before.replace("<think>", "").split())[:2000]
        action = after.strip()
    else:
        thought, action = "", t.strip()
    return thought, action or "(empty response)"


def extract_context(prompt_text: str) -> Tuple[str, str, List[str]]:
    """(intent, current_url, prior_action_lines) from the policy's own prompt."""
    intents = re.findall(r"<user_request>\s*(.*?)\s*</user_request>", prompt_text, re.S)
    intent = intents[0] if intents else ""
    urls = re.findall(r"Tab \d+ \(active\):\s*(\S+)", prompt_text)
    url = urls[-1] if urls else ""
    hist = []
    for m in re.finditer(r"</think>\s*(.*?)<\|im_end\|>", prompt_text, re.S):
        hist.append(" ".join(m.group(1).split())[:300])
    return intent, url, hist


def build_messages(intent: str, url: str, history: List[str],
                   candidates: List[Tuple[str, str]], image_b64: str) -> list:
    trajectory_block = ""
    if history:
        trajectory_block = ("#### Action History ####\n"
                           + "\n".join(f"- {h}" for h in history[-5:]) + "\n\n")
    # Frontier judges ground in raw pixels of the submitted image; Qwen-family
    # judges are [0,1000]-native. OWRL_ARBITER_COORDS=pixel (default) converts
    # the DISPLAYED coords; "normalized" keeps the policy's values verbatim.
    pixel_mode = os.environ.get("OWRL_ARBITER_COORDS", "pixel") == "pixel"
    dims = _image_size(image_b64) if pixel_mode else None
    cand_lines = []
    for i, (thought, action) in enumerate(candidates, 1):
        shown = to_pixel_display(action, *dims) if dims else action
        cand_lines.append(f"  {i}. {shown}")
        if thought:
            cand_lines.append(f"     Thought: {thought}")
    before = (f"{trajectory_block}"
              f"#### Task ####\n{intent}\n\n"
              f"#### Current URL ####\n{url}\n\n")
    after = (
        "#### Candidate Actions ####\n" + "\n".join(cand_lines) + "\n\n"
        "#### Instructions ####\n"
        "Select the best action by number. Consider:\n"
        "1. Which action makes the most progress toward the task goal?\n"
        "2. Which action targets a relevant, visible element on the current page?\n"
        "3. Avoid repeating actions that have already failed.\n"
        "4. Multiple candidates may target similar regions; pick the one most likely to advance the task.\n"
        + ("5. Candidate point_2d values are PIXEL coordinates of the attached screenshot; locate each directly (no markers are drawn).\n"
           if os.environ.get("OWRL_ARBITER_COORDS", "pixel") == "pixel" else
           "5. Locate each candidate's coordinates by mapping the normalized [0,1000] values onto the screenshot directly (no markers are drawn).\n")
        + 
        "6. Re-read the task and identify EVERY constraint (e.g., distances like 'near zip X', categories like 'bread recipe' (not 'bread-based'), filters like 'under 45 min', counts, types, ratings). Before picking ANY candidate (whether terminate or non-terminate), verify the candidate's likely outcome would satisfy each constraint. If the visible state can't satisfy a constraint, prefer an action that gathers more information (scroll, change filter, click into details) over committing to a near-match. Do not accept loose interpretations.\n\n"
        "First, explain your choice in 2-4 sentences: compare the candidates and say "
        "which best advances the task and why (note any that are malformed or would "
        "violate a task constraint). Then, on a new line, output exactly:\n"
        "  {\"selection\": <integer N matching a candidate number>}"
    )
    img = image_b64 if str(image_b64).startswith("data:") else f"data:image/png;base64,{image_b64}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": before},
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "text", "text": after},
        ]},
    ]


async def select(prompt_text: str, image_b64: str,
                 candidates: List[Tuple[str, str]]) -> Tuple[int, str, dict]:
    """-> (winner_index, reasoning_text, record). Failure -> candidate 0."""
    model = os.environ.get("OWRL_ARBITER_MODEL", "gpt-5.5")
    intent, url, hist = extract_context(prompt_text)

    # Scalar-RM mode: score each candidate with the value-head server, argmax.
    scalar_url = os.environ.get("OWRL_SCALAR_URL")
    if scalar_url:
        import httpx
        record = {"model": "scalar-rm", "intent": intent, "url": url,
                  "history_tail": hist[-5:]}
        try:
            async with httpx.AsyncClient(timeout=300) as cli:
                r = await cli.post(scalar_url.rstrip("/") + "/score", json={
                    "intent": intent, "url": url, "history": hist[-8:],
                    "image_b64": image_b64,
                    "candidates": [{"thought": th or "", "action": a}
                                   for th, a in candidates],
                })
                r.raise_for_status()
                scores = r.json()["scores"]
            idx = max(range(len(scores)), key=lambda i: scores[i])
            record["scores"] = scores
            record["selected"] = idx
            reasoning = "scalar scores: " + ", ".join(f"{x:.3f}" for x in scores)
            return idx, reasoning, record
        except Exception as e:
            record["error"] = str(e)
            record["selected"] = 0
            return 0, "", record

    messages = build_messages(intent, url, hist, candidates, image_b64)
    record = {"model": model, "intent": intent, "url": url,
              "history_tail": hist[-5:],
              "prompt": {"system": messages[0]["content"],
                         "before": messages[1]["content"][0]["text"],
                         "after": messages[1]["content"][2]["text"]}}
    try:
        resp = await _get_client().chat.completions.create(
            model=model, messages=messages, max_completion_tokens=2048)
        text = resp.choices[0].message.content or ""
        record["response"] = text
        ms = list(_SELECTION_RE.finditer(text))
        idx = int(ms[-1].group(1)) - 1 if ms else 0
        if not (0 <= idx < len(candidates)):
            idx = 0
        reasoning = (text[: ms[-1].start()] if ms else text).strip()
        record["selected"] = idx
        return idx, reasoning, record
    except Exception as e:  # any failure -> candidate 0, run continues
        record["error"] = str(e)
        record["selected"] = 0
        return 0, "", record


def log_sidecar(task_id: str, step: int, candidates, selected: int,
                reasoning: str, record: dict) -> None:
    log_dir = os.environ.get("OWRL_ARBITER_LOG_DIR")
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id))
        with open(os.path.join(log_dir, f"{safe}.jsonl"), "a") as f:
            f.write(json.dumps({
                "task_id": task_id, "step": step,
                "candidates": [{"thought": t, "action": a} for t, a in candidates],
                "selected": selected, "reasoning": reasoning, "record": record,
            }) + "\n")
    except Exception:
        pass
