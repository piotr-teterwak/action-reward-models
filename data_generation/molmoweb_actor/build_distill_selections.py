"""
Self-distillation selection (option A): pick the best-of-5 candidate per state by
GPT-5.5 PRM score, to feed build_onpolicy_sft.py (stage C) -> train_actor_sft (stage D).

This replaces the *trained-arbiter* selection (stage B, selected_orig_idx) with the
PRM-score argmax computed directly from the reward-model training data, so the actor
self-distills the action GPT-5.5 judged best -- offline, no online rollouts, no drift.

Selection rule per state:
  - drop the state if the best candidate's PRM score < THRESH (don't distill a
    best-of-bad action),
  - among candidates tied at the max score, pick the one whose action_str is most
    frequently sampled (the robust consensus), tiebreak on lowest index.

Out row matches the stage-B schema build_onpolicy_sft.py expects:
  {state_id, selected_orig_idx, ok, top_score}

Env: THRESH (default 0.7).
Usage: python scripts/build_distill_selections.py --scored <scored.jsonl> --out <sel.jsonl>
"""
import os, json, argparse, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="output/prm_offline_pilot/scored_train_full.jsonl")
    ap.add_argument("--out", default="output/onpolicy/distill_a/selections.jsonl")
    ap.add_argument("--thresh", type=float, default=float(os.environ.get("THRESH", "0.7")))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    n = kept = below = 0
    with open(a.out, "w") as f:
        for l in open(a.scored):
            c = json.loads(l); cands = c["candidates"]; n += 1
            scs = [k.get("prm_score", -1) for k in cands]
            if not scs or min(scs) < 0:
                continue
            top = max(scs)
            if top < a.thresh:
                below += 1
                continue
            # candidates tied at the max score -> prefer the most-sampled action_str
            freq = collections.Counter(k["action_str"] for k in cands)
            best_idx = max((i for i in range(len(cands)) if scs[i] == top),
                           key=lambda i: (freq[cands[i]["action_str"]], -i))
            f.write(json.dumps({"state_id": c["state_id"], "selected_orig_idx": best_idx,
                                "ok": True, "top_score": top}) + "\n")
            kept += 1
    print(f"states {n} | dropped (top<{a.thresh}): {below} | wrote {kept} selections -> {a.out}")


if __name__ == "__main__":
    main()
