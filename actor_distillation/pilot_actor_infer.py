"""
Pilot Stage 2: MolmoWeb-4B actor inference, n=5 candidates per offline state.

Reuses the REAL MolmoWebAgent._build_prompt and _parse_molmo_output so the
prompt fed to the model is byte-identical to the live harness. The generate
step is a faithful inline copy of MolmoWebAgent._generate_sync (n>1 branch),
avoiding the data_generation.* import that isn't on path in this checkout.

Run on a GPU node (qsub). CPU/login node has no GPU.

Out: output/prm_offline_pilot/candidates.jsonl  (one line per state, 5 candidates)
"""
import os, io, json, re
import torch
from PIL import Image
from eval.agents import MolmoWebAgent

ROOT = "output/prm_offline_pilot"
STATES = os.environ.get("PILOT_STATES", f"{ROOT}/states.jsonl")
N = 5
TEMP = float(os.environ.get("PILOT_TEMP", "0.7"))
TOP_P = float(os.environ.get("PILOT_TOP_P", "0.8"))
# default (0.7, 0.8) -> candidates.jsonl ; else encode both temp and top_p (when != 0.8)
if abs(TEMP - 0.7) < 1e-9 and abs(TOP_P - 0.8) < 1e-9:
    SUFFIX = ""
else:
    SUFFIX = f"_t{TEMP:g}" + ("" if abs(TOP_P - 0.8) < 1e-9 else f"_p{TOP_P:g}")
OUT = os.environ.get("PILOT_OUT", f"{ROOT}/candidates{SUFFIX}.jsonl")


def clean_action(h):
    """History action string in MolmoWeb's vocabulary: prefer reconstructed
    coordinate for clicks, else the dataset action_str with bid stripped."""
    if h.get("coord"):
        return h["coord"]
    a = h.get("action_str") or h.get("action_description") or ""
    a = re.sub(r"bid='[^']*',?\s*", "", a)   # drop browsergym bid noise
    return a.strip()


def main():
    states = [json.loads(l) for l in open(STATES)]
    CHUNK = int(os.environ.get("PILOT_CHUNK", "0"))
    NCHUNKS = int(os.environ.get("PILOT_NCHUNKS", "1"))
    if NCHUNKS > 1:
        states = states[CHUNK::NCHUNKS]   # strided slice -> balanced chunks
        print(f"chunk {CHUNK}/{NCHUNKS}: {len(states)} states", flush=True)
    else:
        print(f"loaded {len(states)} states", flush=True)

    print(f"TEMP={TEMP} TOP_P={TOP_P} -> {OUT}", flush=True)
    agent = MolmoWebAgent(num_samples=N, temperature=TEMP, top_p=TOP_P)
    agent._ensure_loaded()
    proc, model = agent._processor, agent._model

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["state_id"])
            except: pass

    with open(OUT, "a") as fp:
        for st in states:
            if st["state_id"] in done:
                continue
            img = Image.open(st["screenshot"]).convert("RGB")
            hist = [(clean_action(h), h.get("thought", "")) for h in st["history"]]
            # REAL prompt builder -> guarantees parity with live harness
            prompt = agent._build_prompt(st["task"], st["url"], st["step_idx"], hist)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": img},
            ]}]
            inputs = proc.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}
            input_len = inputs["input_ids"].size(1)
            batched = {k: v.repeat(N, *([1] * (v.dim() - 1))) for k, v in inputs.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(
                    **batched, max_new_tokens=1024,
                    do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=0,
                )
            cands = []
            for i in range(N):
                raw = proc.decode(output[i, input_len:], skip_special_tokens=True)
                action_str = agent._parse_molmo_output(raw)
                try:
                    thought = json.loads(raw.strip()).get("thought", "")
                except Exception:
                    thought = ""
                cands.append(dict(cand_idx=i + 1, raw=raw, action_str=action_str, thought=thought))
            rec = dict(state_id=st["state_id"], sample_id=st["sample_id"], task=st["task"],
                       step_idx=st["step_idx"], url=st["url"], screenshot=st["screenshot"],
                       prompt=prompt, candidates=cands)
            fp.write(json.dumps(rec) + "\n"); fp.flush()
            print(f"  state {st['state_id']:02d}: " +
                  " | ".join(c["action_str"][:24] for c in cands), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
