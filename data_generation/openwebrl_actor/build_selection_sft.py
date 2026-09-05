#!/usr/bin/env python3
"""Stage D: states + candidates + teacher labels -> LLaMAFactory SFT dataset
for the selection ARM.

Rebuilds each catts prompt with the SAME builder as inference (normalized
coords) and writes ShareGPT-format samples:
    system: catts system prompt
    user:   <image> + prompt text (image tag where the screenshot sits)
    assistant: target — no-CoT: '{"selection": N}'
               (--cot: teacher reasoning + newline + the JSON)
plus images/ copies and dataset_info.json, matching the layout their
sft/prepare_openai_for_llamafactory.py produces.

Usage:
  OWRL_ARBITER_COORDS=normalized python build_selection_sft.py \
      --states states.jsonl --candidates candidates.jsonl --labels labels.jsonl \
      --out-dir sft_selection [--cot] [--holdout 0.02]
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, "/projectnb/ivc-ml/piotrt/CUA_evals/OpenWebRL")
os.environ.setdefault("OWRL_ARBITER_COORDS", "normalized")
from openwebrl.frontier_arbiter import (  # noqa: E402
    SYSTEM_PROMPT, build_messages, extract_context, split_think)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    states = {json.loads(l)["state_id"]: json.loads(l) for l in open(args.states)}
    cands = {json.loads(l)["state_id"]: json.loads(l) for l in open(args.candidates)}
    rng = random.Random(args.seed)
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
        pairs = [split_think(c) for c in cd["candidates"]]
        if not (0 <= lab["selection"] < len(pairs)):
            n_skip += 1
            continue
        intent, url, hist = extract_context(st["prompt_text"])
        # build WITHOUT a real image (placeholder): we only need the text parts
        msgs = build_messages(intent, url, hist, pairs, "PLACEHOLDER")
        before = msgs[1]["content"][0]["text"]
        after = msgs[1]["content"][2]["text"]
        src_img = st["images"][-1]
        dst_img = os.path.join(img_dir, os.path.basename(src_img))
        if not os.path.exists(dst_img):
            shutil.copyfile(src_img, dst_img)
        target = f'{{"selection": {lab["selection"] + 1}}}'
        if args.cot and lab.get("reasoning"):
            target = lab["reasoning"].strip() + "\n" + target
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": before + "<image>" + after},
                {"role": "assistant", "content": target},
            ],
            "images": [os.path.join("images", os.path.basename(dst_img))],
        }
        (val if rng.random() < args.holdout else train).append(sample)

    name = "owrl_selection_cot" if args.cot else "owrl_selection"
    for split, rows in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"{name}_{split}.json"), "w") as f:
            json.dump(rows, f)
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w") as f:
        json.dump({
            name: {
                "file_name": f"{name}_train.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {"role_tag": "role", "content_tag": "content",
                         "user_tag": "user", "assistant_tag": "assistant",
                         "system_tag": "system"},
            }
        }, f, indent=1)
    print(f"train {len(train)}, val {len(val)}, skipped {n_skip} -> {args.out_dir}")


if __name__ == "__main__":
    main()
