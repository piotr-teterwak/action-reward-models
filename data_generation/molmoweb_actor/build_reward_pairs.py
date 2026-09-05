"""
Sample a fixed number of Bradley-Terry PAIRS from pointwise reward_data, spread
evenly across trajectories (round-robin), for compute parity with the selector run.
Each output row has exactly 2 candidates (winner first, loser second) = one pair;
train_reward.py then forms a single -logsigmoid(r_w - r_l) per row.

  N pairs / 2 forwards-per-pair  ==  the selector run's forwards-per-epoch.

Selection: per state, take its best pair (max-score vs min-score, gap > --gap).
Round-robin over trajectories: round k takes the k-th highest-gap state of each
trajectory, so pairs spread across trajectories before doubling up on any one.
"""
import os, json, argparse, collections, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-data", default="output/prm_offline_pilot/train_full/reward_data.jsonl")
    ap.add_argument("--states", default="output/prm_offline_pilot/train_full/states_doubled.jsonl")
    ap.add_argument("--out", default="output/prm_offline_pilot/train_full/reward_pairs_15k.jsonl")
    ap.add_argument("--n", type=int, default=15500)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--sample", choices=["extreme", "random"], default="extreme",
                    help="extreme = max-score vs min-score per state; random = a random pair per state")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    sid2traj = {json.loads(l)["state_id"]: json.loads(l)["sample_id"] for l in open(a.states)}
    # per trajectory: list of (gap, winner_cand, loser_cand) best-pairs
    by_traj = collections.defaultdict(list)
    n_states = 0
    for l in open(a.reward_data):
        r = json.loads(l); cands = r["cands"]
        if a.sample == "random":
            i, j = rng.sample(range(len(cands)), 2)
            hi, lo = (i, j) if cands[i]["score"] >= cands[j]["score"] else (j, i)
        else:
            hi = max(range(len(cands)), key=lambda i: cands[i]["score"])
            lo = min(range(len(cands)), key=lambda i: cands[i]["score"])
        gap = cands[hi]["score"] - cands[lo]["score"]
        if gap <= a.gap:
            continue
        n_states += 1
        traj = sid2traj.get(r["state_id"], r["state_id"])
        by_traj[traj].append((gap, {"state_id": r["state_id"], "screenshot": r["screenshot"],
                                    "cands": [cands[hi], cands[lo]]}))   # winner first
    for t in by_traj:
        by_traj[t].sort(key=lambda x: -x[0])   # highest-gap (clearest) pairs first within a trajectory

    # round-robin: round k takes the k-th best state of each trajectory until we hit n
    out = []; k = 0
    trajs = list(by_traj)
    while len(out) < a.n:
        added = 0
        for t in trajs:
            if k < len(by_traj[t]):
                out.append(by_traj[t][k][1]); added += 1
                if len(out) >= a.n:
                    break
        if added == 0:
            break   # exhausted all states
        k += 1
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    per = collections.Counter(sid2traj.get(r["state_id"], r["state_id"]) for r in out)
    print(f"states with a valid pair: {n_states} across {len(by_traj)} trajectories")
    print(f"wrote {len(out)} pairs -> {a.out}  (rounds used: {k}, max per-traj: {max(per.values())}, "
          f"trajectories covered: {len(per)}, mean pairs/traj: {len(out)/len(per):.2f})")


if __name__ == "__main__":
    main()
