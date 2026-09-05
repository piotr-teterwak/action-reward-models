#!/usr/bin/env python3
"""Collect catts-arbiter teacher examples (input prompt + GPT-5 response) for SFT.

Walks completed catts_entropy_v2 / dual_v3 trajectories, finds vote_log steps
where the catts arbiter actually fired (not skipped via entropy gate / single
candidate / etc.), and writes a JSONL of:
    { "messages": [<system>, <user>], "completion": <GPT-5 reasoning + selection> }
suitable for chat-template-based SFT of any small model.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/projectnb/ivc-ml/piotrt/browser_agents/browser-environment")
sys.path.insert(0, str(ROOT))

from eval.webarbiter import build_catts_prompt, ArbiterCandidate, prune_axtree

DEFAULT_SOURCE_DIRS = [
    ROOT / "output/mind2web_bench/catts_entropy_v2",
    ROOT / "output/mind2web_bench/catts_entropy_v2_remaining",
    ROOT / "output/mind2web_bench/catts_entropy_v2_remaining2",
    ROOT / "output/mind2web_bench/catts_entropy_v2_remaining3",
    ROOT / "output/mind2web_bench/dual_v3",
]

SELECTION_RE = re.compile(r"<Selection>\s*(\d+)\s*</Selection>", re.IGNORECASE)


def reconstruct_trajectory(action_history, thoughts):
    """Rebuild the catts trajectory list from saved action_history + thoughts."""
    trajectory = []
    for i, action in enumerate(action_history):
        thought = thoughts[i] if i < len(thoughts) else ""
        trajectory.append({"thought": thought or "", "action": action})
    return trajectory


def extract_arbiter_candidates(saved_candidates):
    """Recover ArbiterCandidate objects from the dict form saved in vote_log."""
    out = []
    for c in saved_candidates:
        out.append(ArbiterCandidate(
            molmo_action=c.get("molmo", ""),
            arbiter_action=c.get("arbiter", ""),
            thought=c.get("thought", "") or "",
            element_info=c.get("element_info", "") or "",
            element_id=c.get("element_id"),
        ))
    return out


def harvest(source_dir: Path, target_n: int, collected, seen_keys, max_axtree_chars: int = 0):
    """Walk one rollout dir and append catts examples until we hit target_n."""
    if not source_dir.is_dir():
        return
    for tid in sorted(os.listdir(source_dir)):
        if len(collected) >= target_n:
            return
        rj = source_dir / tid / "result.json"
        if not rj.is_file():
            continue
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        task = d.get("task", "")
        action_history = d.get("action_history") or []
        thoughts = d.get("thoughts") or []
        if not task:
            continue
        for step_idx, vl in enumerate(d.get("vote_log") or []):
            if len(collected) >= target_n:
                return
            if not isinstance(vl, dict):
                continue
            if vl.get("arbiter_skip"):
                continue          # entropy gate or other skip path
            arb = vl.get("arbiter")
            if not isinstance(arb, dict):
                continue
            saved_cands = arb.get("candidates") or []
            axtree = arb.get("axtree") or ""
            if not axtree or len(saved_cands) < 2:
                continue
            if max_axtree_chars and len(axtree) > max_axtree_chars:
                cand_actions = [c.get("arbiter") or c.get("molmo") or "" for c in saved_cands]
                axtree = prune_axtree(axtree, cand_actions, max_axtree_chars)
            match_log = arb.get("match_log") or []
            if not match_log:
                continue
            # catts mode logs a single match_log entry with mode="catts*" and the
            # full GPT-5 response. Pull the response.
            response = ""
            for m in match_log:
                r = m.get("response", "")
                if isinstance(r, str) and SELECTION_RE.search(r):
                    response = r
                    break
            if not response:
                continue
            cands = extract_arbiter_candidates(saved_cands)
            if len(cands) < 2:
                continue
            trajectory = reconstruct_trajectory(action_history[:step_idx], thoughts[:step_idx])
            # vote_counts from the saved distribution
            distribution = vl.get("distribution") or []
            vote_counts = {entry["action"]: entry["count"] for entry in distribution}
            for c in cands:
                # mirror agents.py: also map arbiter_action -> count if molmo had it
                if c.arbiter_action not in vote_counts and c.molmo_action in vote_counts:
                    vote_counts[c.arbiter_action] = vote_counts[c.molmo_action]

            try:
                messages = build_catts_prompt(
                    intent=task,
                    axtree=axtree,
                    trajectory=trajectory,
                    start_url=d.get("url") or "",
                    current_url=d.get("url") or "",
                    candidates=cands,
                    vote_counts=vote_counts,
                )
            except Exception:
                continue

            key = (tid, step_idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            collected.append({
                "task_id": tid,
                "step": step_idx,
                "messages": messages,        # [system, user]
                "completion": response,      # full GPT-5 reasoning + <Selection>N</Selection>
                "selection": int(SELECTION_RE.search(response).group(1)),
                "n_candidates": len(cands),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="target number of examples")
    ap.add_argument("--out", default="output/distill_catts_v1/teacher_pairs.jsonl")
    ap.add_argument("--max_axtree_chars", type=int, default=0,
                    help="If >0, prune long axtrees to this char budget (priority: candidate-bid lines, then interactive nodes). 0 = no pruning.")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collected = []
    seen_keys = set()
    for src in DEFAULT_SOURCE_DIRS:
        harvest(src, args.n, collected, seen_keys, max_axtree_chars=args.max_axtree_chars)
        if len(collected) >= args.n:
            break

    with open(out_path, "w") as f:
        for ex in collected:
            f.write(json.dumps(ex) + "\n")

    print(f"wrote {len(collected)} examples to {out_path}")
    if collected:
        # Quick sanity stats
        n_cands = [ex["n_candidates"] for ex in collected]
        chars = [len(ex["completion"]) for ex in collected]
        print(f"  n_candidates: min={min(n_cands)}, max={max(n_cands)}")
        print(f"  completion chars: min={min(chars)}, median={sorted(chars)[len(chars)//2]}, max={max(chars)}")
        sels = [ex["selection"] for ex in collected]
        from collections import Counter
        print(f"  selection distribution: {dict(Counter(sels))}")


if __name__ == "__main__":
    main()
