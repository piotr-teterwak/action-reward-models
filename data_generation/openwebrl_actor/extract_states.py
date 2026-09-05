#!/usr/bin/env python3
"""Stage A: released SFT trajectories -> per-turn STATES for ARM training.

Runs OpenWebRL's own Stage-1 converter output (canonical OpenAI episodes with
base64 screenshots) through the SAME chat-template render the runtime uses
(tokenizer.apply_chat_template with tools), producing one record per assistant
turn:
    {"state_id", "episode_id", "turn", "prompt_text", "images": [file, ...],
     "demo_action": <the demonstrated assistant response, for reference>}
Images are written as PNG files (shared across turns of an episode) so the
JSONL stays small and the sampler can attach them as data-URIs.

Usage:
  python extract_states.py --canonical canonical_episodes.jsonl \
      --out states.jsonl --img-dir state_images [--limit-episodes N]

(If --canonical doesn't exist yet, run their sft/convert_to_openai_messages.py
on the downloaded trajectories first.)
"""
import argparse
import base64
import hashlib
import json
import os

from transformers import AutoTokenizer

MODEL = os.environ.get("OWRL_ARM_POLICY", "OpenWebRL/OpenWebRL-4B-SFT")


def render_prompt(tokenizer, messages, tools):
    """The exact call the OpenWebRL runtime makes at inference: template the
    conversation up to (and including) the assistant generation header."""
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )


def save_images(msgs, img_dir):
    """Replace base64 image_url blocks with file refs; return file list in
    order of appearance. Deduped by content hash."""
    files = []
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for block in c:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if "base64," not in url:
                    continue
                raw = base64.b64decode(url.split("base64,", 1)[1])
                h = hashlib.sha1(raw).hexdigest()[:16]
                path = os.path.join(img_dir, f"{h}.png")
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(raw)
                files.append(path)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True,
                    help="Stage-1 output: one canonical OpenAI episode per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--max-images-per-state", type=int, default=1,
                    help="keep only the LAST k screenshots (runtime context_num_screenshots=1)")
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    n_states = n_eps = 0
    with open(args.out, "w") as out:
        for line in open(args.canonical):
            ep = json.loads(line)
            msgs = ep.get("messages") or []
            tools = ep.get("tools")
            ep_id = ep.get("episode_id") or ep.get("task_id") or f"ep{n_eps}"
            n_eps += 1
            if args.limit_episodes and n_eps > args.limit_episodes:
                break
            # walk assistant turns; state k = conversation strictly before it
            turn = 0
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                # PRUNE older screenshots from a deep copy of the context so
                # the rendered template contains exactly the placeholders for
                # the images we keep (runtime keeps context_num_screenshots=1)
                import copy
                context = copy.deepcopy(msgs[:i])
                img_positions = []
                for mi, mm in enumerate(context):
                    cc = mm.get("content")
                    if isinstance(cc, list):
                        for bi, b in enumerate(cc):
                            if isinstance(b, dict) and b.get("type") == "image_url":
                                img_positions.append((mi, bi))
                keep = set(img_positions[-args.max_images_per_state:])
                for mi, bi in reversed(img_positions):
                    if (mi, bi) not in keep:
                        context[mi]["content"].pop(bi)
                imgs = save_images(context, args.img_dir)
                if not imgs:
                    turn += 1
                    continue
                prompt_text = render_prompt(tok, context, tools)
                demo = m.get("content")
                if isinstance(demo, list):
                    demo = " ".join(b.get("text", "") for b in demo if isinstance(b, dict))
                out.write(json.dumps({
                    "state_id": f"{ep_id}__t{turn}",
                    "episode_id": ep_id, "turn": turn,
                    "prompt_text": prompt_text,
                    "images": imgs,
                    "demo_action": (demo or "")[:2000],
                }) + "\n")
                n_states += 1
                turn += 1
    print(f"episodes: {n_eps - 1 if args.limit_episodes else n_eps}, states: {n_states} -> {args.out}")


if __name__ == "__main__":
    main()
