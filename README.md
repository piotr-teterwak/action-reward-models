# Action Reward Models for Web Agents

Minimal, self-contained repo for the **action reward model (ARM) study**:
generate per-step candidate-action data from a web-agent policy, train two
kinds of reward models on teacher labels, and use them to pick actions at
inference time. Everything here was extracted from two production pipelines
("eras") and trimmed to the essential path.

**Written to be read by an AI assistant picking this up cold** — file paths,
gotchas, and provenance are spelled out. Author: Piotr Teterwak
(piotr.teterwak@gmail.com). Models & data: https://huggingface.co/PTeterwak

## The idea, in one paragraph

At every step, a web agent's policy samples **n=5 candidate actions**
(temp 0.7) instead of 1. A **reward model picks which one to execute**. Two RM
types, trained from the *same* teacher-selection labels: a **selection ARM**
(generative: sees all 5 candidates + screenshot, replies `{"selection": N}`)
and a **Bradley–Terry scalar RM** (a value head scores each candidate
independently; argmax executes). Comparative judging beats absolute scoring,
and both beat n=1, across two actor families and even on desktop (OSWorld):

| actor / bench | n=1 | scalar BT | selection ARM | teacher (GPT-5.5) |
|---|---|---|---|---|
| MolmoWeb-4B / OM2W (gpt-5.2 judge) | 25.1% | 33.6% | **37.1%** | — |
| OpenWebRL-4B-SFT / OM2W (o4-mini judge) | 33.8% | 46.3% | **51.1%** | 47.1% |
| Qwen3.5-4B / OSWorld (369 desktop tasks) | 10.0% | — | — | **21.0%** (GPT-5.5 as ARM) |

(The trained ARM *beats its own GPT-5.5 teacher* on OM2W. Judge models differ
across rows — never compare absolute numbers across rows without a re-judge.)

## Assets on Hugging Face

**Models** (all judge-base **Qwen3.5-4B**, LoRA unless noted):

| repo | type | actor era |
|---|---|---|
| `PTeterwak/OpenWebRL-4B-SelectionARM` | selection (merged full model) | OpenWebRL |
| `PTeterwak/OpenWebRL-4B-ScalarRM-LoRA` | BT scalar (adapter + `value_head.safetensors`) | OpenWebRL |
| `PTeterwak/om2w-action-rm-selection-4b` | selection (merged) | MolmoWeb |
| `PTeterwak/om2w-action-rm-scalar-4b-lora` | BT scalar (adapter + `value_head.pt`) | MolmoWeb |

**Data**: `PTeterwak/action-reward-models-data` — states, candidate sets,
teacher labels, and the built training sets for the OpenWebRL era (incl.
screenshots), plus the MolmoWeb-era BT pair files. Layout documented in the
dataset card.

## Pipeline (what the scripts do, in order)

### Stage 1 — data generation (`data_generation/`)

**OpenWebRL actor** (`openwebrl_actor/`, Aug 2026 era — the cleaner one):
1. `extract_states.py` — pull (prompt_text, screenshot) states from SFT
   trajectories. Output: `states_full.jsonl` + `state_images/`.
2. `sample_candidates.py` — for each state, sample **n=5 candidates at
   temp 0.7** from the actor (served via vLLM/sglang). Re-run each state
   multiple times with `#draw` suffixed state_ids to scale the set (we did
   ~3k states → 49.5k sets). Designed as a claim-file fleet: any number of
   concurrent jobs share one work list via `O_CREAT|O_EXCL` claim files.
3. `build_teacher_batch.py` — package each 5-candidate set into an OpenAI
   **Batch API** request with the selection prompt; the teacher (GPT-5.5)
   returns `{"selection": N}` + reasoning. (~$150 for 40k labels; 99.8% parse.)
4. `build_selection_sft.py` — labels → ShareGPT-format SFT set for
   LLaMA-Factory (image + 5 candidates → `{"selection": N}` target).
   2% of *label draws* held out by seed-42 (`rng.random() < 0.02`) — every
   eval script reproduces this exact split; don't change the seed.
5. `build_scalar_rm_data.py` — the same labels → Bradley–Terry pairs
   (teacher's pick vs each *distinct* loser, ≤2 pairs/set → 76.7k pairs),
   LLaMA-Factory `ranking: true` format with chosen/rejected branches.

**MolmoWeb actor** (`molmoweb_actor/`, May–Jun 2026 era): same shape, older
plumbing. `build_catts_distill_data.py` / `build_distill_selections.py` build
selection training data from arbiter runs over MolmoWeb-sampled candidates;
`build_reward_data.py` + `build_reward_pairs.py` build pointwise scores and
compute-parity BT pairs (that era also ablated the *label source*: BT trained
on counterfactual-PRM labels vs on selection labels — see `bt_*` variants).

### Stage 2 — training (`training/`)

**LLaMA-Factory route** (`llamafactory/` — used for the OpenWebRL era; the
easiest to reproduce):
- `arm_lora.yaml` — selection ARM: stage `sft`, all-linear LoRA r=32 on the
  actor's own base, 2 epochs (loss 2.67 → 0.05, ~22h on 1×80GB). Then
  `arm_merge.yaml` merges to a full checkpoint for vLLM serving.
- `scalar_rm_lora.yaml` — BT scalar: stage **`rm`** (LLaMA-Factory trains a
  value head with -logsigmoid(r_chosen − r_rejected)), LoRA r=32, 1 epoch.
  Held-out pairwise accuracy 75.8%. **Gotchas:** eval needs `eval_dataset:`
  (not `dataset:`), batch size 1, 80GB GPU (`scalar_rm_eval.yaml` shows the
  working config); pip may resolve a CUDA-ABI-mismatched torchaudio — pin to
  your torch's CUDA.

**Custom BT trainer** (`custom_bt/` — the MolmoWeb era's route):
`train_reward.py` with `OBJECTIVE=bt` (pairs jsonl in, LoRA + value head
out); `run_train_reward*.qsub` show the exact env/args used, and
`chain_bt_run.sh` the smoke-then-full launch pattern.

### Stage 3 — inference (`inference/`)

- `selection_infer.py` — **the 60-second demo.** Serve the selection ARM with
  vLLM, pass task + screenshot + candidates json, get `{"selection": N}`.
- `scalar_infer.py` — loads base + LoRA + value head directly (no server),
  scores each candidate, argmax.
- `scalar_server.py` — production-grade batched `/score` endpoint (the one
  the eval harnesses call), PRM'-format prompts for the MolmoWeb-era model
  (`templates/prm2_templates.json` is byte-exact to its training data).
- `selection_prompt.py` — the **canonical selection prompt builder**
  (catts_vision v2: pure vision, no DOM/SoM/votes/CoT, single shot). Both
  selection ARMs were trained on prompts from this builder with:
  `CATTS_VISION_PROMPT_V2=1 CATTS_VISION_COLORED=1 VISION_NO_SOM=1
  VISION_ABLATE_DOM=1 VISION_ABLATE_VOTES=1 NORMALIZE_COORDS=1
  CLUSTER_NO_DOM=1 VISION_NO_COT=1`. **Prompt drift is the #1 way to get
  garbage numbers** — use this builder, don't approximate it.

## Cross-cutting gotchas (earned the hard way)

- **Coordinates are normalized [0,1000]** in candidate actions for both ARMs.
  Feeding pixel coords silently degrades selection quality.
- **Selection index is 1-based** in the `{"selection": N}` contract; parse
  failures fall back to candidate 1 (do the same, it matters for parity).
- **Candidate diversity is the fuel**: sample candidates at temp 0.7. Greedy
  candidates collapse to near-duplicates and selection becomes a no-op
  (~72% of steps have ≥2 genuinely distinct candidates at 0.7).
- **The dedup question**: we do NOT dedup candidates before selection
  (CLUSTER_NO_DOM=1 presents all 5). Strict-index metrics under-credit judges
  when duplicates exist — use action-level agreement for analysis.
- **Judges differ across eras** (gpt-5.2 / o4-mini / GPT-4.1 give spreads of
  5–9 points on identical trajectories). Any new comparison table should
  re-judge everything with one judge.
- The scalar value head rides OUTSIDE the adapter weights — always ship
  `value_head.{safetensors,pt}` next to the LoRA (both HF repos do).

## Environments

Three pinned requirements files, split by concern (versions taken from the
actual working SCC envs, Sep 2026): `requirements-inference.txt` (torch 2.11 /
transformers 5.14 / vllm 0.26 — transformers must be >=5.x for qwen3_5),
`requirements-training.txt` (LLaMA-Factory from source; beware CUDA-mismatched
torchaudio), `requirements-datagen.txt` (CPU-side). Training and inference
were run from SEPARATE envs — the MolmoWeb actor pins transformers 4.57.x
while the Qwen3.5 RMs need >=5.x, so plan on two envs if you run both.

**Serving checklist** (vLLM JIT-compiles kernels at startup; every one of
these was independently fatal in a bare batch shell, in this order):
1. `peft` + `safetensors` installed in the serving env (scalar path).
2. `CUDA_HOME` set to a real CUDA >=12.8 install and `$CUDA_HOME/bin` on
   PATH (**absolute paths** — HPC `module load` can silently no-op in
   non-interactive shells; don't trust it).
3. The conda/venv `bin` FIRST on PATH (vLLM's JIT needs `ninja` from it).
4. `LD_LIBRARY_PATH` containing `$CUDA_HOME/lib64` (JIT-built kernels dlopen
   `libcudart.so.12` at runtime).
Both demo paths were verified end-to-end with real weights (scalar: HF
adapter + value head scored 5 candidates; selection: vLLM-served ARM returned
`{"selection": N}`) under exactly this environment.

## Repo layout

```
data_generation/
  openwebrl_actor/   extract_states.py sample_candidates.py build_teacher_batch.py
                     build_selection_sft.py build_scalar_rm_data.py
  molmoweb_actor/    build_catts_distill_data.py build_distill_selections.py
                     build_reward_data.py build_reward_pairs.py
training/
  llamafactory/      arm_lora.yaml arm_merge.yaml scalar_rm_lora.yaml scalar_rm_eval.yaml
  custom_bt/         train_reward.py run_train_reward*.qsub chain_bt_run.sh
requirements-{inference,training,datagen}.txt
inference/
  selection_infer.py scalar_infer.py scalar_server.py selection_prompt.py
  templates/prm2_templates.json
```

## Provenance

Extracted from: `CUA_evals/owrl_arm` (OpenWebRL era, Aug 2026),
`browser_agents/browser-environment` + `browser_agents/repro_v2_tree`
(MolmoWeb era, May–Jul 2026; the repro tree also contains a fully
self-contained OM2W reproduction harness with pinned tasks). Dashboard with
all result tabs: https://weekly-dashboard-inky.vercel.app (tabs 5.20–9.1).
