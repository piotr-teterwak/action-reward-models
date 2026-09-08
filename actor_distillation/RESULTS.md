# Actor self-distillation — results (OM2W, MolmoWeb-4B actor, Jun–Jul 2026)

Two variants of stage B were run to completion. One failed, one worked, and
the contrast between them is the main lesson of this directory.

## TL;DR

| variant | stage B selector | outcome (greedy n=1 vs base) |
|---|---|---|
| online streaming (`onpolicy_online.py`, 8k states) | trained 4B selection ARM, temp-1.0 candidates | **−4.2pp** (26.8% vs 31.1%, worse on all 3 seeds) |
| offline "option A" (`build_distill_selections.py`) | GPT-5.5 PRM-score argmax, 0.7 quality floor | **+8.7pp** vs baseline (95% CI 2.4–15.0, sig.); **+7.2pp** vs random-SFT control (paired t=3.27, p≈0.001, n=190) |

Reference band for what "success" means: best-of-5 **at inference** with the
deployed 31k all-linear selector was **+7.2pp** over n=1 (37.0% vs 29.8%),
and the BT reward head +4.1–4.3pp. So the offline distillation folded
essentially the *entire* selection gain into a single greedy sample.

## The failed run: fully-online, 8k states (June)

Final checkpoint (step 334 ≈ 8k on-policy states), greedy n=1, no judge,
Exact-UV judged, early-term filtered, joint intersection n=236:

```
untrained MolmoWeb (floor)        31.4 / 32.6 / 29.2   avg 31.1%
+ online self-distillation        30.1 / 24.2 / 26.3   avg 26.8%   (Δ −4.2pp)
```

Why it lost — regression analysis over the 15 tasks the base actor solved
(≥2/3 seeds) and the distilled LoRA failed (0/3):

```
trajectory length:        untrained 14.1    LoRA 23.4
terminated (send_msg):    untrained 100%    LoRA 27%
hit the 30-step cap:      untrained 7%      LoRA 67%
scroll fraction:          untrained 24%     LoRA 61%
```

The distilled actor learned to **loop** — scrolling instead of committing,
never sending the final answer. Held-out loss confirms memorization rather
than generalization: on-policy train loss ~0.31 vs held-out loss on
selector-picked actions 0.5192 (base: 0.5245 — a −0.005 nudge, 393/1000
states lower). Diagnosis: temp-1.0 candidates + a judge with no quality
floor means the "winner" of 5 bad candidates still gets distilled, and
mid-trajectory dithering (scroll/noop) is systematically safer-looking to a
local judge than a commitment action.

## The successful run: offline PRM-argmax, "option A" (July)

Stage B replaced the live ARM with the stored GPT-5.5 PRM-score argmax
(`../data_generation/molmoweb_actor/build_distill_selections.py`), whose two
guardrails address exactly the failure above: **drop the state if the best
candidate scores < 0.7** (never distill best-of-bad), and break ties toward
the actor's most-sampled action (consensus prior).

Intersection-judged results (`distill_a_actor`, greedy n=1):

```
vs baseline:            +8.7pp  (n=196, 95% CI 2.4–15.0, significant)
vs baseline, early-term:+9.2pp  (n=239, significant)
```

**The control that makes this causal**: `distill_rand` — same SFT recipe,
same states, but the target is a RANDOM candidate instead of the PRM pick —
lands at baseline (+0.5–1.0pp, noise). Selected-vs-random on paired tasks:
**+7.2pp, t=3.27, p≈0.001 (n=190)**. It's the *selection*, not merely SFT on
the actor's own outputs.

## Provenance / where the numbers live

- Online-run eval: `output/mind2web_bench/online8k_step334_n1_run{1,2,3}` vs
  `baseline_v{13,14,15}`; comparison script `cmp_actor_vs_untrained_uv.py`
  (beware: its printed labels still say "BT-half" — it was cloned from
  `cmp_bthalf_vs_n1_uv.py` with only the run list swapped).
- Regression/overfit analysis: `output/onpolicy/run8k/{_regressions,_final2}.txt`.
- Offline-run eval: `output/mind2web_bench/distill_{a,8b,rand}_actor_run*`,
  aggregated in the 6.24 dashboard `results_table.json`.
- Selection-at-inference reference: `output/onpolicy/run8k/{DBL3,FULLHALF}.txt`
  (all "BT-half / doubled / all-linear" files there are best-of-5 *selector*
  arms over the untrained actor — not distilled actors).
- All under `/projectnb/ivc-ml/piotrt/browser_agents/browser-environment`;
  dashboard tabs "6.10/6.16 → Integrating into training" and the 6.24
  results table on weekly-dashboard.

## Takeaways for the next attempt

1. **Quality-floor the teacher signal.** Argmax-of-5 without a threshold
   distills garbage on hard states; the 0.7 PRM floor is the single change
   most responsible for the sign flip.
2. **Watch for loop collapse**, not just loss: trajectory length, terminate
   rate, and scroll fraction of the distilled actor are the early-warning
   metrics.
3. **Always run the random-SFT control** — without `distill_rand` the +8.7pp
   would be unattributable.
4. Online/streaming needs an anti-collapse mechanism (KL anchor, low epochs)
   before it's worth retrying; offline round-based distillation is the
   proven recipe.
