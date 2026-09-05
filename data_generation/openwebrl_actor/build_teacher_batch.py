#!/usr/bin/env python3
"""Stage C1: states + candidates -> OpenAI Batch API request file(s) for
GPT-5.5 teacher selection labels.

Each request is the catts_vision selection prompt (built by the SAME
frontier_arbiter.build_messages the live arbiter uses; normalized coords for
the Qwen student — set OWRL_ARBITER_COORDS=normalized), explain-then-JSON so we
can train both CoT and no-CoT variants from one label set.

Writes chunked JSONL files (<= --chunk-size requests, images inline) ready for
`client.batches.create`, plus a manifest mapping custom_id -> state_id.

Usage:
  OWRL_ARBITER_COORDS=normalized python build_teacher_batch.py \
      --states states.jsonl --candidates candidates.jsonl \
      --out-dir teacher_batches --model gpt-5.5 [--limit N]
Then submit with submit_teacher_batches.py.
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, "/projectnb/ivc-ml/piotrt/CUA_evals/OpenWebRL")
os.environ.setdefault("OWRL_ARBITER_COORDS", "normalized")
from openwebrl.frontier_arbiter import build_messages, extract_context, split_think  # noqa: E402


def to_datauri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0, help="stop after N unique sets")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    states = {json.loads(l)["state_id"]: json.loads(l) for l in open(args.states)}
    seen_sets = set()  # (base_state, frozenset(actions)) dedup across draws
    chunk_idx = n_in_chunk = n_total = 0
    manifest = {}
    chunk_f = None

    def new_chunk():
        nonlocal chunk_f, chunk_idx, n_in_chunk
        if chunk_f:
            chunk_f.close()
        chunk_idx += 1
        n_in_chunk = 0
        chunk_f = open(os.path.join(args.out_dir, f"batch_{chunk_idx:04d}.jsonl"), "w")

    new_chunk()
    for l in open(args.candidates):
        rec = json.loads(l)
        sid = rec.get("base_state_id") or rec["state_id"]
        st = states.get(sid)
        if not st:
            continue
        cands = [split_think(c) for c in rec["candidates"]]
        # skip states with a single distinct action — nothing to select
        if len({a for _, a in cands}) < 2:
            continue
        # dedup identical candidate SETS across draws of the same state
        key = (sid, frozenset(a for _, a in cands))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        if args.cap and n_total >= args.cap:
            break
        intent, url, hist = extract_context(st["prompt_text"])
        img = to_datauri(st["images"][-1])
        messages = build_messages(intent, url, hist, cands, img)
        cid = f"arm_{n_total:06d}"
        manifest[cid] = rec["state_id"]
        chunk_f.write(json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": args.model, "messages": messages,
                     "max_completion_tokens": 2048},
        }) + "\n")
        n_total += 1
        n_in_chunk += 1
        if n_in_chunk >= args.chunk_size:
            new_chunk()
        if args.limit and n_total >= args.limit:
            break
    chunk_f.close()
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"))
    print(f"{n_total} teacher requests in {chunk_idx} chunks -> {args.out_dir}")


if __name__ == "__main__":
    main()
