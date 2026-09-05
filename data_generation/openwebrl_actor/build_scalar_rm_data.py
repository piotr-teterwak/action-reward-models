#!/usr/bin/env python3
"""Scalar (value-head) RM dataset: selection sets -> pairwise preference data.

Each teacher-labeled candidate set (GPT-5.5 picked index k of n distinct
candidates) yields (chosen = candidate k, rejected = each other distinct
candidate) pairs, in LLaMAFactory's rm-stage ShareGPT format:
    {"messages": [system, user(with <image>)], "chosen": {...}, "rejected": {...},
     "images": [...]}
The user prompt frames ONE candidate for scoring (same catts context, single
action instead of 5) so the RM scores singleton (state, action) pairs.
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, "/projectnb/ivc-ml/piotrt/CUA_evals/OpenWebRL")
os.environ.setdefault("OWRL_ARBITER_COORDS", "normalized")
from openwebrl.frontier_arbiter import extract_context, split_think  # noqa: E402

SYSTEM = (
    "You are an expert web-agent action evaluator. Given the task, the current "
    "page state and screenshot, and ONE proposed next action, judge how well "
    "the action advances the task."
)


def action_prompt(intent, url, hist, thought, action):
    hist_s = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(hist[-8:])) or "  (none)"
    th = f"\nAgent reasoning: {thought.strip()[:600]}" if thought else ""
    return (
        f"Task: {intent}\nCurrent URL: {url}\nRecent actions:\n{hist_s}\n"
        f"Current page screenshot: <image>\n\nProposed action:\n{action.strip()[:800]}{th}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pairs-per-set", type=int, default=2,
                    help="cap pairs per set to bound dataset size/diversity mix")
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(args.seed)

    states = {json.loads(l)["state_id"]: json.loads(l) for l in open(args.states)}
    cands = {json.loads(l)["state_id"]: json.loads(l) for l in open(args.candidates)}

    train, val = [], []
    n_skip = 0
    for l in open(args.labels):
        lab = json.loads(l)
        sid = lab.get("state_id")
        base = (sid or "").split("#")[0]
        if lab.get("selection") is None or base not in states or sid not in cands:
            n_skip += 1
            continue
        st, cd = states[base], cands[sid]
        pairs_ta = [split_think(c) for c in cd["candidates"]]
        k = lab["selection"]
        if not (0 <= k < len(pairs_ta)):
            n_skip += 1
            continue
        chosen_th, chosen_a = pairs_ta[k]
        # distinct losers only (identical-action "losers" are not preferences)
        losers = [(th, a) for i, (th, a) in enumerate(pairs_ta)
                  if i != k and a.strip() != chosen_a.strip()]
        if not losers:
            n_skip += 1
            continue
        rng.shuffle(losers)
        losers = losers[: args.max_pairs_per_set]

        intent, url, hist = extract_context(st["prompt_text"])
        src_img = st["images"][-1]
        dst = os.path.join(img_dir, os.path.basename(src_img))
        if not os.path.exists(dst):
            shutil.copyfile(src_img, dst)
        img_rel = os.path.join("images", os.path.basename(dst))

        for (lth, la) in losers:
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": action_prompt(intent, url, hist, chosen_th, chosen_a)},
                ],
                "chosen": {"role": "assistant", "content": "good action"},
                "rejected": {"role": "assistant", "content": "bad action"},
                "images": [img_rel],
            }
            # rm stage compares score(prompt+chosen) vs score(prompt+rejected);
            # for action scoring the ACTION must live in the compared branch:
            sample["messages"] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": action_prompt(intent, url, hist, None, "(see candidate below)")},
            ]
            sample["chosen"] = {"role": "assistant",
                                "content": f"Action: {chosen_a.strip()[:800]}\nReasoning: {(chosen_th or '')[:400]}"}
            sample["rejected"] = {"role": "assistant",
                                  "content": f"Action: {la.strip()[:800]}\nReasoning: {(lth or '')[:400]}"}
            (val if rng.random() < args.holdout else train).append(sample)

    name = "owrl_scalar_rm"
    for split, rows in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"{name}_{split}.json"), "w") as f:
            json.dump(rows, f)
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w") as f:
        json.dump({
            name: {
                "file_name": f"{name}_train.json",
                "ranking": True,
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "chosen": "chosen",
                            "rejected": "rejected", "images": "images"},
                "tags": {"role_tag": "role", "content_tag": "content",
                         "user_tag": "user", "assistant_tag": "assistant",
                         "system_tag": "system"},
            }
        }, f, indent=1)
    print(f"train {len(train)}, val {len(val)}, skipped {n_skip} -> {args.out_dir}")


if __name__ == "__main__":
    main()
