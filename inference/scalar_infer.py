#!/usr/bin/env python3
"""Minimal scalar (Bradley–Terry value-head) RM demo: score each candidate
action independently, pick the argmax. No server needed — loads the model
directly (1 GPU, ~12GB bf16).

    python scalar_infer.py --image screenshot.png \
        --task "Find running shoes under $50" --url https://shop.example.com \
        --candidates cands.json \
        --adapter PTeterwak/OpenWebRL-4B-ScalarRM-LoRA

Adapters (both use base Qwen/Qwen3.5-4B + a value_head file in the repo):
  PTeterwak/OpenWebRL-4B-ScalarRM-LoRA   (OpenWebRL-actor era, value_head.safetensors)
  PTeterwak/om2w-action-rm-scalar-4b-lora (MolmoWeb-actor era, value_head.pt;
      that era scored with the PRM' prompt format — see inference/scalar_server.py
      + templates/prm2_templates.json for the byte-exact production path)

This demo uses the OpenWebRL-era prompt format. For production-grade serving
(batched 5-way /score endpoint) see scalar_server.py in this directory.
"""
import argparse, json, os

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "OpenWebRL/OpenWebRL-4B-SFT"

SYSTEM = ("You are an expert web-agent action evaluator. Given the task, the current "
          "page state and screenshot, and ONE proposed next action, judge how well "
          "the action advances the task.")


def load_value_head(adapter):
    from huggingface_hub import hf_hub_download
    for fn, loader in (("value_head.safetensors", "st"), ("value_head.pt", "pt")):
        try:
            path = adapter + "/" + fn if os.path.isdir(adapter) else hf_hub_download(adapter, fn)
        except Exception:
            continue
        if not os.path.exists(path):
            continue
        if loader == "st":
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu")
    raise FileNotFoundError("no value_head.{safetensors,pt} in adapter repo")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--url", default="")
    p.add_argument("--candidates", required=True)
    p.add_argument("--adapter", default="PTeterwak/OpenWebRL-4B-ScalarRM-LoRA")
    p.add_argument("--base", default=BASE)
    a = p.parse_args()

    cands = json.load(open(a.candidates))
    from PIL import Image
    img = Image.open(a.image).convert("RGB")

    processor = AutoProcessor.from_pretrained(a.base, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    vh = load_value_head(a.adapter)
    w = vh["v_head.summary.weight"].to(device="cuda", dtype=torch.bfloat16)
    b = vh.get("v_head.summary.bias")
    b = b.to(device="cuda", dtype=torch.bfloat16) if b is not None else None

    scores = []
    for c in cands:
        user = (f"Task: {a.task}\nCurrent URL: {a.url}\nRecent actions:\n  (none)\n"
                f"Current page screenshot: ")
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user},
                                          {"type": "image"},
                                          {"type": "text", "text": "\n\nProposed action:\n(see candidate below)"}]},
            {"role": "assistant", "content":
                f"Action: {c['action'].strip()[:800]}\nReasoning: {(c.get('thought') or '')[:400]}"},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1]
        s = (h.to(w.dtype) @ w.T).squeeze()
        if b is not None:
            s = s + b.squeeze()
        scores.append(float(s))

    pick = max(range(len(scores)), key=lambda i: scores[i])
    print(json.dumps({"scores": scores, "argmax": pick,
                      "action": cands[pick]["action"]}, indent=2))


if __name__ == "__main__":
    main()
