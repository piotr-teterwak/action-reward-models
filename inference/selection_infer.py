#!/usr/bin/env python3
"""Minimal selection-ARM demo: given a task, screenshot, and 5 candidate
actions, ask the ARM which to execute.

The ARM is a plain chat model — serve it with anything OpenAI-compatible:
    vllm serve PTeterwak/OpenWebRL-4B-SelectionARM --port 8000 \
        --trust-remote-code --max-model-len 32768
(or PTeterwak/om2w-action-rm-selection-4b for the MolmoWeb-era model; that
one was served with --dtype half --max-model-len 65536.)

Usage:
    python selection_infer.py --url http://127.0.0.1:8000/v1 \
        --image screenshot.png --task "Find running shoes under $50" \
        --candidates cands.json
where cands.json = [{"thought": "...", "action": "..."}, ...]  (2-5 entries).

Output: the winning index + the model's raw reply. Parse contract: the model
replies {"selection": N} (1-indexed); on parse failure fall back to 1.
Click coordinates inside candidate actions are in NORMALIZED [0,1000] space
(both ARMs were trained that way — do not pass raw pixels).
"""
import argparse, json, re, sys

from openai import OpenAI

from selection_prompt import build_catts_vision_prompt_v2  # canonical builder

SEL = re.compile(r'\{\s*"selection"\s*:\s*(\d+)\s*\}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="served model name (default: first listed)")
    p.add_argument("--image", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--candidates", required=True, help="json list of {thought, action}")
    p.add_argument("--history", default=None, help="optional json list of prior action strings")
    p.add_argument("--url-page", dest="url_page", default="", help="current page URL")
    a = p.parse_args()

    cands = json.load(open(a.candidates))
    history = json.load(open(a.history)) if a.history else []
    img_bytes = open(a.image, "rb").read()

    # one singleton cluster per candidate (CLUSTER_NO_DOM semantics).
    # The builder expects candidate OBJECTS with .molmo_action / .thought:
    from types import SimpleNamespace
    clusters = [
        {"rep": SimpleNamespace(molmo_action=c["action"], thought=c.get("thought", "")),
         "vote_count": 1, "cluster_key": str(i)}
        for i, c in enumerate(cands)
    ]
    messages = build_catts_vision_prompt_v2(a.task, history, a.url_page if hasattr(a, "url_page") else "",
                                            clusters, img_bytes)

    client = OpenAI(base_url=a.url, api_key="EMPTY")
    model = a.model or client.models.list().data[0].id
    r = client.chat.completions.create(model=model, messages=messages,
                                       temperature=0.0, max_tokens=2048)
    text = r.choices[0].message.content or ""
    m = list(SEL.finditer(text))
    pick = int(m[-1].group(1)) if m else 1
    if not (1 <= pick <= len(cands)):
        pick = 1
    print(json.dumps({"selection": pick, "action": cands[pick - 1]["action"],
                      "raw_reply": text[-500:]}, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    main()
