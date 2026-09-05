#!/usr/bin/env python3
"""Stage B: n=5 candidates per state from the SFT policy via the vLLM shim.

Resumable (skips state_ids already in --out). Temperature is a CLI arg — set
from the diversity-probe verdict.

Usage (shim already serving the policy on --url):
  python sample_candidates.py --states states.jsonl --out candidates.jsonl \
      --url http://127.0.0.1:9000/generate --n 5 --temperature 0.7 --workers 8
"""
import argparse
import base64
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

_lock = threading.Lock()


def to_datauri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def claim(args, state_id):
    """True if we won the claim (or claiming is off)."""
    if not args.claim_dir:
        return True
    try:
        os.close(os.open(os.path.join(args.claim_dir, state_id),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def sample_state(state, args, out):
    if not claim(args, state["state_id"]):
        return
    """--draws K: K independent candidate-sets of n per state. All samples for
    a (state, temperature) come back from ONE big-n request (vLLM prefix
    caching makes the shared prompt nearly free), then are partitioned into
    sets client-side (iid draws, so grouping is arbitrary)."""
    imgs = [to_datauri(p) for p in state["images"]]
    # temperature plan: mostly the deployment temp, a slice at 1.0 for wider
    # candidate coverage (draws 4k+1 of every 4 when --mix-temp)
    plan = {}
    for d in range(args.draws):
        t = 1.0 if (args.mix_temp and d % 4 == 3) else args.temperature
        plan[t] = plan.get(t, 0) + 1
    texts_by_temp = {}
    for t, k in plan.items():
        need = k * args.n
        pool = []
        while len(pool) < need:
            chunk = min(20, need - len(pool))  # chunk big-n requests
            r = requests.post(args.url, json={
                "text": state["prompt_text"],
                "sampling_params": {"temperature": t, "top_p": 0.9, "n": chunk,
                                    "max_new_tokens": args.max_new_tokens},
                "return_logprob": True,
                "image_data": imgs,
            }, timeout=(30, 900))
            r.raise_for_status()
            tx = r.json()["text"]
            pool.extend(tx if isinstance(tx, list) else [tx])
        texts_by_temp[t] = pool
        print(f"  {state['state_id']} t={t}: {len(pool)} samples", flush=True)
    recs = []
    d = 0
    for t, k in plan.items():
        pool = texts_by_temp[t]
        for j in range(k):
            cands = pool[j * args.n:(j + 1) * args.n]
            if len(cands) < args.n:
                continue
            recs.append({"state_id": f"{state['state_id']}#d{d}",
                         "base_state_id": state["state_id"],
                         "temperature": t,
                         "images": state["images"], "candidates": cands})
            d += 1
    with _lock:
        for rec in recs:
            out.write(json.dumps(rec) + "\n")
        out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:9000/generate")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--mix-temp", action="store_true")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/N contiguous state slice")
    ap.add_argument("--reverse", action="store_true",
                    help="process the slice tail-first (meet-in-the-middle helper)")
    ap.add_argument("--claim-dir", default="",
                    help="atomic per-state claim files; lets any number of jobs share one state list")
    ap.add_argument("--rotate-frac", default="",
                    help="i/K: rotate the state list so job i starts at a distinct offset (cache locality)")
    args = ap.parse_args()

    import glob as _glob
    done = set()
    for f in _glob.glob(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                     "candidates_shard*.jsonl")):
        for l in open(f):
            try:
                done.add(json.loads(l).get("base_state_id") or json.loads(l)["state_id"])
            except Exception:
                pass
    all_states = [json.loads(l) for l in open(args.states)]
    i, nsh = (int(x) for x in args.shard.split("/"))
    per = (len(all_states) + nsh - 1) // nsh
    my = all_states[i * per:(i + 1) * per]
    if args.reverse:
        my = list(reversed(my))
    states = [s for s in my if s["state_id"] not in done]
    if args.claim_dir:
        os.makedirs(args.claim_dir, exist_ok=True)
    if args.rotate_frac:
        i, k = (int(x) for x in args.rotate_frac.split("/"))
        off = i * len(states) // k
        states = states[off:] + states[:off]
    if args.limit:
        states = states[:args.limit]
    print(f"{len(states)} states to sample ({len(done)} already done)")
    failed = 0
    with open(args.out, "a") as out, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(sample_state, s, args, out) for s in states]
        for i, f in enumerate(futs):
            try:
                f.result()
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"state failed: {e}")
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(states)}", flush=True)
    print(f"done ({failed} failures — rerun to retry)")


if __name__ == "__main__":
    main()
