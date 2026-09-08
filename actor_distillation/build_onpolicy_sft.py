"""
On-policy selection-distillation, STAGE C: build actor SFT data.

Joins on-policy candidates (stage A: 5 samples/state from the CURRENT actor) with
the arbiter's selection (stage B: selected_orig_idx) and emits, per state, the
actor prompt + the SELECTED action as the SFT target. The actor is then trained
(stage D) to GENERATE that action greedily -> folds best-of-5+arbiter into n=1.

Out row: {state_id, screenshot, prompt, target}   target = selected candidate's raw molmo JSON

--only-changed keeps only states where the selected action differs from the actor's
plurality (most-sampled) action -- i.e. where selection actually moved something,
so we don't just reinforce what the actor already does.
"""
import os, json, argparse, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="stage-A candidates jsonl (merged)")
    ap.add_argument("--selections", required=True, help="stage-B selector output jsonl (selected_orig_idx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-changed", action="store_true",
                    help="keep only states where selected action != actor plurality action")
    a = ap.parse_args()

    sel = {}
    for l in open(a.selections):
        x = json.loads(l)
        if x.get("selected_orig_idx") is not None and x.get("ok", True):
            sel[x["state_id"]] = int(x["selected_orig_idx"])

    n = kept = changed = 0
    with open(a.out, "w") as f:
        for l in open(a.candidates):
            c = json.loads(l); sid = c["state_id"]
            if sid not in sel:
                continue
            n += 1
            cands = c["candidates"]; p = sel[sid]
            if p < 0 or p >= len(cands):
                continue
            plurality = collections.Counter(k["action_str"] for k in cands).most_common(1)[0][0]
            is_changed = cands[p]["action_str"] != plurality
            if is_changed:
                changed += 1
            if a.only_changed and not is_changed:
                continue
            f.write(json.dumps({"state_id": sid, "screenshot": c["screenshot"],
                                "prompt": c["prompt"], "target": cands[p]["raw"]}) + "\n")
            kept += 1
    print(f"states with a selection: {n} | selection moved off plurality: {changed} ({100*changed/max(n,1):.1f}%)")
    print(f"wrote {kept} actor-SFT examples -> {a.out}  ({'only-changed' if a.only_changed else 'all'})")


if __name__ == "__main__":
    main()
