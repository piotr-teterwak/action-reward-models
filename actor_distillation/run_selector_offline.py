"""
Run the deployed 4B-nc arbiter (Qwen3.5-4B, no-CoT, pure-vision/no-DOM/no-votes/
normalized) as a SELECTOR over the frozen offline candidates, by calling the real
catts_vision_select_v2 with the exact nc env config. Records which original
candidate it picks per state, so we can score its PRM-regret vs slot-1/majority/oracle.

Needs a local vLLM serving Qwen3.5-4B (the qsub starts it). Chunkable via SEL_CHUNK/SEL_NCHUNKS.
Out: $SEL_OUT  ({state_id, selected_orig_idx, ok, raw})
"""
import os, json, asyncio, random
# SHARED pure-vision/no-DOM/no-votes/normalized config. The MODE flags
# (VISION_NO_COT, VISION_WITH_CONSTRAINTS, VISION_CONSTRAINTS_NO_THOUGHT,
# VISION_DISABLE_THINK, ARBITER_PAIRWISE, ...) come from the CALLER's env so
# one driver runs any ablation.
for k, v in {
    "CATTS_VISION_PROMPT_V2": "1", "CATTS_VISION_COLORED": "1", "CATTS_VISION_MAX_TOKENS": "16384",
    "VISION_NO_SOM": "1", "VISION_ABLATE_DOM": "1", "VISION_ABLATE_VOTES": "1",
    "NORMALIZE_COORDS": "1", "CLUSTER_NO_DOM": "1",
    # this repo's selection_prompt.py keeps only the no-CoT prompt variant
    # (the with-CoT branch was removed in the repro extraction), so default it
    # on — production left it to the caller's env (onpolicy_select.qsub).
    "VISION_NO_COT": "1",
}.items():
    os.environ.setdefault(k, v)

from openai import OpenAI
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "inference"))
from selection_prompt import catts_vision_select_v2, ArbiterCandidate  # repo-canonical builder

ROOT = os.environ.get("SEL_ROOT", "output/prm_offline_pilot/test_full")
STATES = {s["state_id"]: s for s in (json.loads(l) for l in open(f"{ROOT}/states.jsonl"))}
CANDS = [json.loads(l) for l in open(f"{ROOT}/candidates.jsonl")]
CHUNK = int(os.environ.get("SEL_CHUNK", "0")); NCHUNKS = int(os.environ.get("SEL_NCHUNKS", "1"))
if NCHUNKS > 1:
    CANDS = CANDS[CHUNK::NCHUNKS]
OUT = os.environ["SEL_OUT"]
MODEL = os.environ.get("ARBITER_MODEL", "q4b-arbiter")
_url = os.environ["ARBITER_URL"]
# mirror agents.py: openrouter endpoint -> OPENROUTER_API_KEY; local vLLM -> dummy
_key = (os.environ.get("OPENROUTER_API_KEY") or "dummy") if "openrouter" in _url.lower() else "x"
client = OpenAI(base_url=_url, api_key=_key)
LIMIT = int(os.environ.get("SEL_LIMIT", "0"))   # 0 = all (smoke-test cap)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["state_id"])
        except: pass


async def select_one(c, sem, fp, lock):
    sid = c["state_id"]
    async with sem:
        st = STATES[sid]
        sb = open(st["screenshot"], "rb").read()
        traj = [{"action": h.get("coord") or h.get("action_str") or "", "thought": h.get("thought", "")}
                for h in st["history"]]
        reps = []
        for i, k in enumerate(c["candidates"]):
            ac = ArbiterCandidate(molmo_action=k["action_str"], arbiter_action="", thought=k.get("thought", ""))
            ac._orig = i
            reps.append(ac)
        order = list(range(len(reps)))
        random.Random(sid).shuffle(order)   # deterministic per-state shuffle
        cluster_list = [{"rep": reps[j], "vote_count": 1, "cluster_key": ("nodom", pos)}
                        for pos, j in enumerate(order)]
        try:
            winner, text, ok = await catts_vision_select_v2(
                client, MODEL, st["task"], sb, traj, st["url"], cluster_list)
            sel = getattr(winner, "_orig", None) if winner is not None else None
        except Exception as e:
            sel, ok, text = None, False, f"ERR {e}"
        async with lock:
            fp.write(json.dumps({"state_id": sid, "selected_orig_idx": sel,
                                 "ok": bool(ok), "raw": (text or "")[:160]}) + "\n")
            fp.flush()


async def main():
    todo = [c for c in CANDS if c["state_id"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"chunk {CHUNK}/{NCHUNKS}: {len(todo)} states to select (skipping {len(done)} done)", flush=True)
    sem = asyncio.Semaphore(int(os.environ.get("SEL_CONCURRENCY", "16")))
    lock = asyncio.Lock()
    with open(OUT, "a") as fp:
        prog = {"n": 0}
        async def runner(c):
            await select_one(c, sem, fp, lock); prog["n"] += 1
            if prog["n"] % 50 == 0: print(f"  {prog['n']}/{len(todo)}", flush=True)
        await asyncio.gather(*(runner(c) for c in todo))
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
