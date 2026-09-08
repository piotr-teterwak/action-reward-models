"""
On-policy distillation GO/NO-GO pre-check (no training).

How often does the arbiter's selected action differ from the actor's plurality
(most-sampled) action across the 5 on-policy samples? That disagreement IS the
headroom: if they usually agree, the actor already does what the arbiter picks
and there's nothing to distill; if they often disagree, the loop has room to help.

Run: python scripts/onpolicy_precheck.py --candidates <stageA.jsonl> --selections <stageB.jsonl>
"""
import json, argparse, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--selections", required=True)
    a = ap.parse_args()
    sel = {json.loads(l)["state_id"]: json.loads(l).get("selected_orig_idx")
           for l in open(a.selections)}
    n = changed = sel1 = 0   # changed vs plurality; sel1 = picked the (1-indexed) first sample
    distinct = []
    for l in open(a.candidates):
        c = json.loads(l); sid = c["state_id"]; p = sel.get(sid)
        if p is None:
            continue
        cands = c["candidates"]
        if p < 0 or p >= len(cands):
            continue
        n += 1
        acts = [k["action_str"] for k in cands]
        plurality = collections.Counter(acts).most_common(1)[0][0]
        if acts[p] != plurality:
            changed += 1
        if p == 0:
            sel1 += 1
        distinct.append(len(set(acts)))
    print(f"states: {n}")
    print(f"  arbiter picked OFF the plurality action: {changed} ({100*changed/max(n,1):.1f}%)  <- headroom")
    print(f"  arbiter picked the 1st sample:           {sel1} ({100*sel1/max(n,1):.1f}%)")
    print(f"  mean distinct actions among the 5:        {sum(distinct)/max(len(distinct),1):.2f}")
    print("  RULE OF THUMB: <~10% off-plurality -> little to distill; >~25% -> worth running the loop.")


if __name__ == "__main__":
    main()
