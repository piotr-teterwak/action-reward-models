"""Vision arbiter for n=5 action selection (extracted from eval/webarbiter.py).

Minimal delete-only extraction of the ONE live path used by the action-reward-model
reproductions: single-shot 5-way `catts_vision` selection with the v2 prompt builder
under the production env config
    CATTS_VISION_PROMPT_V2=1 CATTS_VISION_COLORED=1 CATTS_VISION_MAX_TOKENS=16384
    VISION_NO_SOM=1 VISION_ABLATE_DOM=1 VISION_ABLATE_VOTES=1
    NORMALIZE_COORDS=1 CLUSTER_NO_DOM=1 VISION_NO_COT=1
(no ARBITER_PAIRWISE, no PRM_SCORE_MODE, no lookahead, no SoM rendering, no
per-image markers). Env-var reads are kept so behavior matches the source; paths
whose code was removed raise loudly if their env flag is set anyway.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Env vars that fully determine the arbiter prompt/behavior. Snapshotted into
# every prompt_record so an old run is reconstructable even if defaults change.
_ARBITER_PROVENANCE_ENV = (
    "CATTS_VISION_PROMPT_V2", "CATTS_VISION_COLORED", "CATTS_VISION_MAX_TOKENS",
    "VISION_NO_SOM", "VISION_ABLATE_IMAGE", "VISION_ABLATE_DOM",
    "VISION_ABLATE_VOTES", "VISION_ABLATE_CAND_THOUGHTS", "VISION_ABLATE_HISTORY",
    "VISION_ABLATE_TRAJ_THOUGHTS", "CLUSTER_NO_DOM", "NORMALIZE_COORDS",
    "PRM_SCORE_MODE", "PRM_INDEPENDENT", "PRM_RUBRIC", "VISION_NO_COT",
    "VISION_NO_COT_THINK", "VISION_WITH_CONSTRAINTS", "VISION_CONSTRAINTS_NO_THOUGHT",
    "VISION_THINK_BUDGET_CHARS", "VISION_DISABLE_THINK",
    "ARBITER_NO_THINK", "LOOKAHEAD_ENABLE",
)


def _arbiter_env_snapshot() -> dict:
    return {k: os.environ.get(k) for k in _ARBITER_PROVENANCE_ENV
            if os.environ.get(k) is not None}


def _extract_prompt_parts(messages):
    """Split a chat `messages` list into (system_text, user_text, image_bytes).

    user_text is the concatenation of the user text blocks with a literal
    `<<IMAGE>>` marker inserted where the image block sits, so the saved text
    preserves the exact wording AND where the screenshot was positioned. Only
    the FIRST image is returned as bytes (the main screenshot)."""
    import base64 as _b64
    system_text, parts, image_bytes = "", [], None
    for m in messages:
        content = m.get("content")
        if m.get("role") == "system":
            system_text = content if isinstance(content, str) else ""
            continue
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for b in content:
                if b.get("type") == "text":
                    parts.append(b["text"])
                elif b.get("type") == "image_url":
                    parts.append("<<IMAGE>>")
                    if image_bytes is None:
                        url = b.get("image_url", {}).get("url", "")
                        if url.startswith("data:") and "base64," in url:
                            try:
                                image_bytes = _b64.b64decode(url.split("base64,", 1)[1])
                            except Exception:
                                pass
    return system_text, "\n".join(parts), image_bytes


def _build_request_record(messages, api_kwargs, candidate_index=None,
                          candidate_action=None) -> dict:
    """Capture one chat-completion request verbatim for later reconstruction.
    Stashes the raw image under the private `_image_bytes` key for the caller to
    write to disk; the caller must pop it before serializing to JSON."""
    system_text, user_text, image_bytes = _extract_prompt_parts(messages)
    rec = {
        "candidate_index": candidate_index,
        "candidate_action": candidate_action,
        "system": system_text,
        "user_text": user_text,
        "response_format": api_kwargs.get("response_format"),
        "sampling": {k: api_kwargs[k] for k in
                     ("temperature", "max_tokens", "max_completion_tokens")
                     if k in api_kwargs},
    }
    if image_bytes is not None:
        rec["image"] = {
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "nbytes": len(image_bytes),
        }
        rec["_image_bytes"] = image_bytes
    else:
        rec["image"] = None
    return rec


@dataclass
class ArbiterCandidate:
    """A candidate action with both MolmoWeb and WebArbiter representations."""
    molmo_action: str        # Original: click(622, 470)
    arbiter_action: str      # Translated: click('42')
    thought: str             # MolmoWeb's reasoning
    element_info: str = ""   # What was at the click target
    raw_output: str = ""     # Full MolmoWeb output
    element_id: Optional[int] = None  # DOM element identity from elementFromPoint


def _parse_click_xy(molmo_action: str) -> Optional[Tuple[int, int]]:
    """Pull (x, y) pixel coords out of a MolmoWeb click/type action."""
    m = re.match(r"click\(\s*(\d+)\s*,\s*(\d+)\s*\)", molmo_action)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"type\(\s*(\d+)\s*,\s*(\d+)\s*,", molmo_action)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_catts_vision_response(response: str, num_candidates: int) -> Tuple[Optional[int], Optional[str]]:
    """Parse {\"thought\": ..., \"selection\": N} from VLM output. Returns
    (zero-based-index, thought) or (None, None) on parse failure."""
    s = (response or "").strip()
    # Try direct JSON parse, then JSON-substring parse.
    candidates_text = [s]
    if "{" in s and "}" in s:
        candidates_text.append(s[s.find("{"): s.rfind("}") + 1])
    for c in candidates_text:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict): continue
        sel = obj.get("selection")
        try: sel = int(sel)
        except Exception: continue
        if 1 <= sel <= num_candidates:
            return sel - 1, obj.get("thought", "")
    # Salvage truncated answer-first JSON (e.g. finish=length on `{"selection": N, "thought": ...`).
    m = re.search(r'"selection"\s*:\s*(\d+)', s)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_candidates:
            return idx, None
    # Last-ditch: scrape any digit from the tail
    for n in re.findall(r"\b(\d+)\b", s[-80:])[::-1]:
        idx = int(n) - 1
        if 0 <= idx < num_candidates:
            return idx, None
    return None, None


# VISION_NO_COT variant: bare selection only, no generated reasoning fields.
CATTS_VISION_V2_SYSTEM_PROMPT_NOCOT = (
    "You are a web automation arbiter. Pick the candidate action most likely "
    "to advance the user's task.\n\n"
    "Reply with a single JSON object: {\"selection\": <integer N matching a candidate number>}\n"
    "Do not output anything else."
)

# VISION_NO_COT_THINK variant: model thinks freely inside <think>...</think>,
# THEN emits the bare {"selection": N}. Decoder uses guided_regex so the
# JSON tail is grammar-enforced (no parse failures); the thinking middle is
# unconstrained so Qwen3.5's native CoT can run.
CATTS_VISION_V2_SYSTEM_PROMPT_NOCOT_THINK = (
    "You are a web automation arbiter. Pick the candidate action most likely "
    "to advance the user's task.\n\n"
    "First, reason briefly inside <think> ... </think> tags about which "
    "candidate best advances the task (consider task constraints, candidate "
    "actions vs visible UI, and recent action history). Then output one "
    "JSON object: {\"selection\": <integer N matching a candidate number>}.\n"
    "Output exactly this structure and nothing else after the JSON."
)


def build_catts_vision_prompt_v2(
    intent: str,
    trajectory: List[dict],
    current_url: str,
    cluster_list: List[dict],
    image_bytes: bytes,
) -> List[dict]:
    """Build the v2 vision arbiter prompt.

    `cluster_list` is a list of dicts with keys {rep, vote_count, cluster_key},
    one per DOM-resolved cluster of MolmoWeb candidates. The order is the
    order shown to the model — caller is responsible for any shuffling.

    Ablation env vars (each "1" = REMOVE that component):
        VISION_ABLATE_IMAGE          drop the marked screenshot
        VISION_ABLATE_VOTES          drop "— N vote(s)" suffix
        VISION_ABLATE_DOM            drop "-> 'button Search'" DOM-label suffix
        VISION_ABLATE_CAND_THOUGHTS  drop the per-candidate "Thought: …" line
        VISION_ABLATE_HISTORY        drop the entire #### Action History #### block
        VISION_ABLATE_TRAJ_THOUGHTS  keep history but strip "(thought: …)" per step
    """
    import base64 as _b64

    ablate_image          = os.environ.get("VISION_ABLATE_IMAGE") == "1"
    ablate_votes          = os.environ.get("VISION_ABLATE_VOTES") == "1"
    ablate_dom            = os.environ.get("VISION_ABLATE_DOM") == "1"
    ablate_cand_thoughts  = os.environ.get("VISION_ABLATE_CAND_THOUGHTS") == "1"
    ablate_history        = os.environ.get("VISION_ABLATE_HISTORY") == "1"
    ablate_traj_thoughts  = os.environ.get("VISION_ABLATE_TRAJ_THOUGHTS") == "1"
    no_som_markers        = os.environ.get("VISION_NO_SOM") == "1"
    normalize_coords      = os.environ.get("NORMALIZE_COORDS") == "1"
    no_cot                = os.environ.get("VISION_NO_COT") == "1"
    no_cot_think          = os.environ.get("VISION_NO_COT_THINK") == "1"
    if not (no_cot or no_cot_think):
        # Repro build: the with-CoT decision prompt (constraints+thought+selection)
        # was removed — only the no-CoT paths used by the reproduced runs remain.
        raise RuntimeError("Repro build requires VISION_NO_COT=1 (or VISION_NO_COT_THINK=1); "
                           "the with-CoT arbiter prompt variant was removed.")

    # Click-coord normalizer: pixel (1280×720) → Qwen-native [0,1000].
    _click_re = re.compile(r'click\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)')
    def _norm(s):
        if not normalize_coords:
            return s
        def repl(m):
            x = round(float(m.group(1)) / 1280 * 1000)
            y = round(float(m.group(2)) / 720 * 1000)
            return f'click({x}, {y})'
        return _click_re.sub(repl, s)

    # Trajectory block (action history) — only shown if not ablated.
    trajectory_block = ""
    if not ablate_history and trajectory:
        # Default: last-5 steps (myopic). VISION_FULL_HISTORY=1 → show the entire
        # trajectory so the arbiter sees the whole rollout, not just a 5-step window.
        hist = trajectory if os.environ.get("VISION_FULL_HISTORY") == "1" else trajectory[-5:]
        traj_lines = []
        for step in hist:
            action = _norm(step.get("action", "") or "")
            thought = step.get("thought", "") or ""
            if ablate_traj_thoughts:
                traj_lines.append(f"- {action}")
            else:
                traj_lines.append(f"- {action}  (thought: {thought})")
        trajectory_block = "#### Action History ####\n" + "\n".join(traj_lines) + "\n\n"

    # Candidate lines: one per cluster representative.
    cand_lines = []
    for i, cl in enumerate(cluster_list, 1):
        c = cl["rep"]
        count = cl["vote_count"]
        is_click = _parse_click_xy(c.molmo_action) is not None
        if is_click and not ablate_image and not no_som_markers:
            color_name = "marker"  # SoM rendering removed; unreachable without VISION_NO_SOM=1
            marker_note = f"  [{color_name} marker]"
        elif not is_click and not no_som_markers:
            marker_note = "  [no marker — non-click action]"
        else:
            marker_note = ""
        line = f"  {i}. {_norm(c.molmo_action)}"
        if not ablate_dom:
            line += f"  -> {c.element_info or '(no element info)'}"
        if not ablate_votes:
            line += f"  — {count} vote(s)"
        line += marker_note
        cand_lines.append(line)
        if not ablate_cand_thoughts and (c.thought or "").strip():
            cand_lines.append(f"     Thought: {c.thought}")
    candidates_str = "\n".join(cand_lines)

    # Image is placed RIGHT BEFORE the candidate-actions block so the marker
    # references in each candidate line ("[RED marker]") are adjacent in the
    # token stream to the visual region they reference.
    text_before_image = (
        f"{trajectory_block}"
        f"#### Task ####\n{intent}\n\n"
        f"#### Current URL ####\n{current_url}\n\n"
    )
    if not no_som_markers:
        marker_instruction = "Use the colored markers on the screenshot to locate each click candidate; non-click actions have no marker."
    elif normalize_coords:
        marker_instruction = "Locate each click(x, y) by mapping the normalized [0,1000] coordinates onto the screenshot directly (no markers are drawn)."
    else:
        marker_instruction = "Locate each click(x, y) by mapping the pixel coordinates onto the screenshot directly (no markers are drawn)."
    vote_instruction = (
        "Prefer actions with higher vote counts unless a minority action is clearly better."
        if not ablate_votes
        else "Multiple candidates may target similar regions; pick the one most likely to advance the task."
    )
    text_after_image = (
        f"#### Candidate Actions ####\n{candidates_str}\n\n"
        f"#### Instructions ####\n"
        f"Select the best action by number. Consider:\n"
        f"1. Which action makes the most progress toward the task goal?\n"
        f"2. Which action targets a relevant, visible element on the current page?\n"
        f"3. Avoid repeating actions that have already failed.\n"
        f"4. {vote_instruction}\n"
        f"5. {marker_instruction}\n"
        f"6. Re-read the task and identify EVERY constraint (e.g., distances like 'near zip X', categories like 'bread recipe' (not 'bread-based'), filters like 'under 45 min', counts, types, ratings). Before picking ANY candidate (whether terminate or non-terminate), verify the candidate's likely outcome would satisfy each constraint. If the visible state can't satisfy a constraint, prefer an action that gathers more information (scroll, change filter, click into details) over committing to a near-match. Do not accept loose interpretations (e.g., 'sandwich made with bread' is NOT a 'bread recipe'; a pet 534 mi away is NOT 'near' the requested zip).\n\n"
        + (
            "Reason briefly inside <think> ... </think> (keep it concise — a few sentences) "
            "about which candidate best advances the task. Then emit the JSON object on its own:\n"
            "  {\"selection\": <integer N matching a candidate number>}\n"
            "Output exactly this structure: <think>...</think> followed by the JSON, and nothing else after the JSON."
            if no_cot_think else
            "Reply with a single JSON object:\n"
            "  {\"selection\": <integer N matching a candidate number>}\n"
            "Do not output anything else."
        )
    )

    if ablate_image:
        user_content_blocks = [{"type": "text", "text": text_before_image + text_after_image}]
    else:
        b64 = _b64.b64encode(image_bytes).decode("ascii")
        user_content_blocks = [
            {"type": "text", "text": text_before_image},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text_after_image},
        ]

    user_content = (user_content_blocks[0]["text"]
                    if len(user_content_blocks) == 1
                       and user_content_blocks[0].get("type") == "text"
                    else user_content_blocks)
    if no_cot_think:
        system_prompt = CATTS_VISION_V2_SYSTEM_PROMPT_NOCOT_THINK
    else:
        system_prompt = CATTS_VISION_V2_SYSTEM_PROMPT_NOCOT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def catts_vision_select_v2(
    client,
    model: str,
    intent: str,
    screenshot_bytes: bytes,
    trajectory: List[dict],
    current_url: str,
    cluster_list: List[dict],
    viewport_w: int = 1280,
    viewport_h: int = 720,
    prompt_record: Optional[dict] = None,
) -> Tuple[ArbiterCandidate, str, bool]:
    """v2 vision arbiter: clustered candidates + ablation-aware prompt.

    `prompt_record` (optional out-param): if a dict is passed, it is filled with
    the EXACT request(s) sent (system+user prompt, model, endpoint, schema,
    sampling, env snapshot, image hash) so the run is reconstructable later even
    if this code changes. Per-request raw image bytes are stashed under the
    private `_image_bytes` key for the caller to write to disk + then pop.
    """
    prm_mode = os.environ.get("PRM_SCORE_MODE") == "1"
    prm_independent = os.environ.get("PRM_INDEPENDENT") == "1"
    prm_rubric = os.environ.get("PRM_RUBRIC") == "1"
    pairwise = os.environ.get("ARBITER_PAIRWISE") == "1"
    # Repro build: only the single-shot decision path was extracted. Fail loudly
    # rather than silently running a different scoring mode.
    if prm_mode or pairwise or os.environ.get("CATTS_VISION_PER_IMAGE") == "1":
        raise RuntimeError(
            "This reproduction build only supports the single-shot decision path "
            "(unset PRM_SCORE_MODE / ARBITER_PAIRWISE / CATTS_VISION_PER_IMAGE)."
        )
    if prompt_record is not None:
        prompt_record.update({
            "arbiter_fn": "catts_vision_select_v2",
            "prompt_builder": "build_catts_vision_prompt_v2",
            "model": model,
            "base_url": str(getattr(client, "base_url", "") or ""),
            "env": _arbiter_env_snapshot(),
            "scoring": ("pairwise" if pairwise
                        else "prm_rubric" if (prm_mode and prm_rubric)
                        else "prm_independent" if (prm_mode and prm_independent)
                        else "prm_allatonce" if prm_mode else "decision"),
            "n_clusters": len(cluster_list),
            "requests": [],
        })

    if len(cluster_list) < 2:
        rep = cluster_list[0]["rep"] if cluster_list else None
        if prompt_record is not None:
            prompt_record["short_circuit"] = "fewer_than_2_clusters_no_model_call"
        return rep, "v2: need >=2 clusters", False

    ablate_image = os.environ.get("VISION_ABLATE_IMAGE") == "1"
    no_som = os.environ.get("VISION_NO_SOM") == "1"

    # Render one marker per CLUSTER (not per raw candidate). With VISION_NO_SOM
    # we pass the raw unmarked screenshot (no overlay) so the model must
    # ground click coords visually on its own.
    if ablate_image:
        marked_png = b""
    elif no_som:
        marked_png = screenshot_bytes
    else:
        # SoM overlay rendering was removed in this repro build (the production
        # config runs VISION_NO_SOM=1). Fail loudly instead of silently diverging.
        raise RuntimeError("SoM marker rendering removed in repro build; "
                           "set VISION_NO_SOM=1 (or VISION_ABLATE_IMAGE=1).")

    messages = build_catts_vision_prompt_v2(
        intent, trajectory, current_url, cluster_list, marked_png,
    )

    is_reasoning = any(x in model for x in ("o4", "o3", "o1", "gpt-5"))
    is_frontier = is_reasoning or any(x in model for x in (
        "gpt-4", "claude", "qwen3-vl-235b", "qwen3.5-397b", "qwen3-max",
    ))

    api_kwargs = {"model": model, "messages": messages}
    if is_reasoning:
        api_kwargs["max_completion_tokens"] = 2048
        if os.environ.get("ARBITER_NO_THINK") == "1":
            api_kwargs["reasoning_effort"] = "none"  # suppress hidden reasoning
    else:
        api_kwargs["max_tokens"] = int(os.environ.get("CATTS_VISION_MAX_TOKENS", 4096))
        api_kwargs["temperature"] = 0
        # Qwen3.5's chat template defaults to opening <think>; with strict json_schema
        # the model is forced to emit JSON inside an unclosed think block (suboptimal
        # conditioning). VISION_DISABLE_THINK=1 swaps the template to emit
        # <think>\n\n</think>\n\n (empty think block) so JSON lands in post-think mode.
        if os.environ.get("VISION_DISABLE_THINK") == "1":
            api_kwargs.setdefault("extra_body", {})
            api_kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
    n_cands = len(cluster_list)
    if not is_frontier:
        if os.environ.get("VISION_NO_COT_THINK") == "1":
            # Native <think>...</think> reasoning + grammar-enforced JSON tail.
            # Qwen3.5's chat template auto-prepends <think>, so the response itself
            # starts INSIDE the think block — the regex must not require the opening
            # <think> tag. It must cap chars inside the think block (thinking budget)
            # and then mandate </think> + the JSON selection.
            #
            # Char budget ≈ tokens * 3.5 (English). VISION_THINK_BUDGET_CHARS caps
            # the thought; default 6000 (~1700 tokens).
            allowed = "|".join(str(i) for i in range(1, n_cands + 1))
            budget = int(os.environ.get("VISION_THINK_BUDGET_CHARS", 6000))
            api_kwargs.pop("response_format", None)
            api_kwargs.setdefault("extra_body", {})
            api_kwargs["extra_body"]["guided_regex"] = (
                r"[\s\S]{0," + str(budget) + r"}</think>\s*"
                r'\{"selection":\s*(?:' + allowed + r")\}"
            )
            # vLLM 0.19 requires backend choice be explicit when guided_regex is non-trivial;
            # xgrammar handles {min,max} quantifiers and alternations cleanly.
            api_kwargs["extra_body"].setdefault("guided_decoding_backend", "xgrammar")
            api_kwargs["max_tokens"] = int(os.environ.get("CATTS_VISION_MAX_TOKENS", 16384))
        elif os.environ.get("VISION_NO_COT") == "1":
            # bare selection, no generated reasoning (CoT ablation)
            api_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vision_selection_nocot",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selection": {"type": "integer", "minimum": 1, "maximum": n_cands},
                        },
                        "required": ["selection"],
                    },
                },
            }
        # (neither flag set -> build_catts_vision_prompt_v2 already raised above)

    req_rec = _build_request_record(messages, api_kwargs) if prompt_record is not None else None
    if req_rec is not None:
        prompt_record["requests"].append(req_rec)
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: client.chat.completions.create(**api_kwargs)
        )
        text = resp.choices[0].message.content or ""
        if req_rec is not None:
            req_rec["response"] = text
            req_rec["ok"] = True
        idx, _thought = parse_catts_vision_response(text, n_cands)
        if idx is None:
            logger.warning(f"  CATTS-VISION-V2 response unparseable: {text[:120]!r}")
            return cluster_list[0]["rep"], text, False
        winner = cluster_list[idx]["rep"]
        logger.info(f"  CATTS-VISION-V2 selected: {winner.molmo_action} ({winner.arbiter_action})")
        return winner, text, True
    except Exception as e:
        logger.warning(f"CATTS-VISION-V2 selection failed: {e}")
        if req_rec is not None:
            req_rec["response"] = str(e)
            req_rec["ok"] = False
        return cluster_list[0]["rep"], str(e), False
