"""
Build POINTWISE reward-model data: one (state, single-action) prompt per candidate,
with that candidate's GPT-5.5 PRM score as the scalar label. Mirrors
build_nocot_soft_data.py but emits, per state, the 5 single-candidate prompts +
their scores (the unit a Bradley-Terry / regression reward head trains on).

Out row: {state_id, screenshot, cands: [{system, user_pre, user_post, score} x5]}

The training objective (Bradley-Terry pairwise vs Huber regression) is a CLI flag
on train_reward.py, NOT here -- the pointwise data is identical for both.
"""
import os, json, random, re, argparse
for k, v in {"CATTS_VISION_PROMPT_V2": "1", "CATTS_VISION_COLORED": "1", "VISION_NO_SOM": "1",
             "VISION_ABLATE_DOM": "1", "VISION_ABLATE_VOTES": "1", "NORMALIZE_COORDS": "1",
             "CLUSTER_NO_DOM": "1", "VISION_NO_COT": "1"}.items():
    os.environ.setdefault(k, v)
from eval.webarbiter import build_catts_vision_prompt_v2, ArbiterCandidate
import io
from PIL import Image

_PH = io.BytesIO(); Image.new("RGB", (8, 8)).save(_PH, format="PNG"); _PH = _PH.getvalue()


def split_prompt(messages):
    system = messages[0]["content"]; uc = messages[-1]["content"]
    if isinstance(uc, str):
        return system, uc, ""
    pre, post, seen = [], [], False
    for b in uc:
        if b.get("type") == "text":
            (post if seen else pre).append(b["text"])
        else:
            seen = True
    return system, "".join(pre), "".join(post)


def _real_choice(cands):
    def bk(a):
        if a.startswith("click("):
            m = re.findall(r"-?\d+", a); return ("c", int(int(m[0]) / 80), int(int(m[1]) / 80)) if len(m) >= 2 else ("c",)
        return (a.split("(")[0],)
    return len(set(bk(k["action_str"]) for k in cands)) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="output/prm_offline_pilot/scored_train_full.jsonl")
    ap.add_argument("--states", default="output/prm_offline_pilot/train_full/states.jsonl")
    ap.add_argument("--out", default="output/prm_offline_pilot/train_full/reward_data.jsonl")
    ap.add_argument("--filter", default="realchoice", choices=["realchoice", "score", "none"])
    a = ap.parse_args()

    ST = {s["state_id"]: s for s in (json.loads(l) for l in open(a.states))}
    SC = [json.loads(l) for l in open(a.scored)]
    n = 0
    with open(a.out, "w") as f:
        for c in SC:
            sid = c["state_id"]; st = ST.get(sid)
            if st is None:
                continue
            cands = c["candidates"]; scs = [k["prm_score"] for k in cands]
            if min(scs) < 0:
                continue
            if a.filter == "realchoice" and not _real_choice(cands):
                continue
            if a.filter == "score" and len(set(round(x, 3) for x in scs)) == 1:
                continue
            traj = [{"action": h.get("coord") or h.get("action_str") or "", "thought": h.get("thought", "")}
                    for h in st["history"]]
            out_cands = []
            for k in cands:
                rep = ArbiterCandidate(molmo_action=k["action_str"], arbiter_action="", thought=k.get("thought", ""))
                cluster_list = [{"rep": rep, "vote_count": 1, "cluster_key": ("nodom", 0)}]  # SINGLE candidate
                msgs = build_catts_vision_prompt_v2(c["task"], traj, c["url"], cluster_list, _PH)
                system, user_pre, user_post = split_prompt(msgs)
                out_cands.append({"system": system, "user_pre": user_pre, "user_post": user_post,
                                  "score": float(k["prm_score"]), "action_str": k["action_str"]})
            f.write(json.dumps({"state_id": sid, "screenshot": c["screenshot"], "cands": out_cands}) + "\n")
            n += 1
    print(f"built {n} pointwise reward states ({len(out_cands) if n else 0} cands each) -> {a.out}")


if __name__ == "__main__":
    main()
