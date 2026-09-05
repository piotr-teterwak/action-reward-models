"""
Reward-model arbiter server: makes the value-head BT reward model look like a vLLM
selector to the OM2W rollout. Exposes /v1/chat/completions; on each selection request
it splits the multi-candidate prompt into N single-candidate prompts (the reward model's
training format -- only the "#### Candidate Actions ####" block differs N vs 1), scores
each with the reward model, and returns {"selection": argmax+1}.

So the existing OM2W rollout (--arbiter-url http://host:PORT/v1) drives it unchanged.

Env: REWARD_CKPT (LoRA+value_head dir), BASE_MODEL, PORT.

Trimmed copy of scripts/reward_arbiter_server.py keeping ONLY the value-head scoring
path (GEN_MODE and the MolmoWeb backbone branches were removed; the reproduced scalar
model reward_bt_prm2_ep2 is Qwen3.5-4B based). The tiny pieces of scripts/train_reward.py
it used (processor, build_cand, RewardModel -- QWEN path only) are inlined below.

The PRM prompt templates are loaded from templates/prm2_templates.json, which was
extracted byte-for-byte from the first line of the training-format reference file the
real run used (output/prm_offline_pilot/train_full/reward_data_score_prm2.jsonl,
via PRM_FMT_REF). Setting PRM_FMT_REF to a reward-data jsonl still works and overrides
the local JSON.
"""
import os, re, json, base64, io

CAND_HDR = "#### Candidate Actions"
# a candidate line is built as f"  {i}. {action}" -> EXACTLY two leading spaces.
# (Thought continuations are 5-space indented; markdown lists inside a thought start
# at col 0.) Requiring the two-space indent avoids over-splitting on a numbered list
# embedded in a candidate's Thought (e.g. state 332's send_msg answer).
CAND_RE = re.compile(r"^  (\d+)\.\s", re.M)


def _split_user(content):
    """From a chat 'content' (str or list of blocks) -> (pre_text, post_text, image_bytes)."""
    if isinstance(content, str):
        return content, "", None
    pre, post, img, seen_img = [], [], None, False
    for b in content:
        t = b.get("type")
        if t == "text":
            (post if seen_img else pre).append(b.get("text", ""))
        elif t in ("image", "image_url"):
            seen_img = True
            u = b.get("image_url", {}).get("url", "") if t == "image_url" else b.get("image", "")
            if isinstance(u, str) and "base64," in u:
                img = base64.b64decode(u.split("base64,", 1)[1])
    return "".join(pre), "".join(post), img


def _single_candidate_posts(post):
    """Split the N-candidate post text into N posts, each with ONE candidate (renumbered 1.)."""
    i = post.find(CAND_HDR)
    if i < 0:
        return [post]                       # no candidate block found -> 1 scoring of the whole thing
    head, body = post[:i], post[i:]
    hdr_end = body.find("\n") + 1
    hdr, rest = body[:hdr_end], body[hdr_end:]
    # split rest at the next "#### " section (instructions) if present
    instr_i = rest.find("\n####")
    cands_txt, instr = (rest[:instr_i], rest[instr_i:]) if instr_i >= 0 else (rest, "")
    # break cands_txt into per-candidate chunks on lines starting with "N."
    idxs = [m.start() for m in CAND_RE.finditer(cands_txt)]
    chunks = [cands_txt[idxs[k]:(idxs[k + 1] if k + 1 < len(idxs) else len(cands_txt))] for k in range(len(idxs))]
    out = []
    for ch in chunks:
        one = re.sub(r"^  \d+\.", "  1.", ch.strip("\n"), count=1)   # renumber to 1.
        out.append(f"{head}{hdr}{one}\n{instr}")
    return out


# --- PRM-format serving (for reward models trained on reward_data_score_prm*.jsonl) ---
# The decision-format split above ends each candidate with the {"selection"} instruction
# block; a PRM-trained reward model instead expects the score block (and the "score how
# likely" system prompt). Since the value head reads the LAST-token hidden state, the
# served prompt must match training. We keep the working decision code path (split +
# return {"selection": argmax}) but rewrite each candidate's system prompt and
# instruction block to the PRM templates. Enabled by PRM_PROMPT_FORMAT=1.
INSTR_HDR = "#### Instructions ####"
_PRM_TMPL = None


def _prm_templates(ref=None):
    """(system, instruction-block) PRM templates, loaded once so they are byte-identical
    to the training data. Default: templates/prm2_templates.json next to this script
    (extracted from the first line of reward_data_score_prm2.jsonl, the PRM_FMT_REF the
    real reward_bt_prm2_ep2 rollouts used). A PRM_FMT_REF jsonl overrides it. Cached."""
    global _PRM_TMPL
    if _PRM_TMPL is None:
        ref = ref or os.environ.get("PRM_FMT_REF")
        if ref:
            c = json.loads(open(ref).readline())["cands"][0]
            up = c["user_post"]; k = up.find(INSTR_HDR)
            _PRM_TMPL = (c["system"], up[k:] if k >= 0 else "")
        else:
            _local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "templates", "prm2_templates.json")
            t = json.load(open(_local))
            _PRM_TMPL = (t["system"], t["instruction_block"])
    return _PRM_TMPL


def _to_prm_format(posts, ref=None):
    """Rewrite decision-format single-candidate posts to PRM format: keep the candidate-action
    prefix, swap the instruction block for the PRM one. Returns (prm_system, [rewritten_posts])."""
    sys_, instr = _prm_templates(ref)
    out = [(p[:p.find(INSTR_HDR)] + instr) if INSTR_HDR in p else p for p in posts]
    return sys_, out


# --- Inlined from scripts/train_reward.py (QWEN path only; molmo branches dropped) ---
SEQ = ("input_ids", "attention_mask", "mm_token_type_ids")
# image inputs the backbone forward accepts (Qwen: grid_thw)
MM_KEYS = ("pixel_values", "image_grid_thw", "mm_token_type_ids",
           "image_token_pooling", "image_grids", "image_num_crops")

processor = None   # set in main() (AutoProcessor for BASE_MODEL)


def build_cand(system, user_pre, user_post, img):
    """Prompt-only tokenization for one (state, action); pool at the last token. (Qwen path.)"""
    user = [{"type": "text", "text": user_pre}, {"type": "image", "image": img},
            {"type": "text", "text": user_post}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user}]
    enc = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                        add_generation_prompt=True)
    ex = {}
    for k, v in enc.items():
        if k == "token_type_ids":
            continue
        ex[k] = v[0] if k in SEQ else v
    ex["last_pos"] = ex["input_ids"].shape[0] - 1
    return ex


def main():
    """Load the reward model and serve. Heavy imports are deferred so the
    splitter functions above can be unit-tested without a GPU / model load."""
    global processor
    import torch, torch.nn as nn, uvicorn
    from fastapi import FastAPI, Request
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    class RewardModel(nn.Module):   # inlined from scripts/train_reward.py
        def __init__(self, backbone, hidden):
            super().__init__()
            self.backbone = backbone
            self.value_head = nn.Linear(hidden, 1)
            nn.init.normal_(self.value_head.weight, std=1e-3); nn.init.zeros_(self.value_head.bias)

        def forward(self, input_ids, attention_mask, last_pos, **mm):
            mm = {k: v for k, v in mm.items() if k in MM_KEYS}
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **mm)
            h = out.hidden_states[-1]                                  # [N,T,H]
            hl = h[torch.arange(h.size(0), device=h.device), last_pos.to(h.device)]
            return self.value_head(hl.float()).squeeze(-1)            # [N]

    BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")
    REVISION = os.environ.get("BASE_REVISION") or None
    CKPT = os.environ["REWARD_CKPT"]
    PORT = int(os.environ.get("PORT", "8995"))

    _proc_kw = {"trust_remote_code": True}
    if REVISION:
        _proc_kw["revision"] = REVISION
    processor = AutoProcessor.from_pretrained(BASE, **_proc_kw)

    print(f"[reward-arbiter] loading {CKPT} (base {BASE}, molmo=False) ...", flush=True)
    _lk = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    if REVISION:
        _lk["revision"] = REVISION
    base = AutoModelForImageTextToText.from_pretrained(BASE, **_lk)
    base.config.use_cache = False
    bb = PeftModel.from_pretrained(base, CKPT)
    hid = getattr(base.config, "hidden_size", None) or base.config.text_config.hidden_size
    model = RewardModel(bb, hid)
    model.value_head.load_state_dict(torch.load(os.path.join(CKPT, "value_head.pt"), map_location="cpu"))
    model.value_head.to(base.device).to(torch.float32); model.eval()
    print("[reward-arbiter] ready", flush=True)

    @torch.no_grad()
    def _score(system, pre, post_one, img):
        ex = build_cand(system, pre, post_one, img)
        b = {k: (v.unsqueeze(0) if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v)
             for k, v in ex.items() if k != "last_pos"}
        b = {k: (v.to(base.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        b["last_pos"] = torch.tensor([ex["last_pos"]], device=base.device)
        return model(**b).item()

    # REWARD_IO_LOG: append the per-candidate served prompt + scalar value-head score for every
    # scoring call, so the arbiter's I/O is inspectable after the fact (the rollout's arbiter_logs
    # only capture the upstream decision request, not this inner per-candidate scoring layer).
    IO_LOG = os.environ.get("REWARD_IO_LOG", "")
    if IO_LOG:
        os.makedirs(os.path.dirname(IO_LOG) or ".", exist_ok=True)
        print(f"[reward-arbiter] logging per-candidate I/O -> {IO_LOG}", flush=True)
    _STEP = [0]
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        msgs = body.get("messages", [])
        system = next((m["content"] if isinstance(m["content"], str) else "" for m in msgs if m.get("role") == "system"), "")
        user = next((m for m in msgs if m.get("role") == "user"), {"content": ""})
        pre, post, imgb = _split_user(user["content"])
        img = Image.open(io.BytesIO(imgb)).convert("RGB") if imgb else Image.new("RGB", (1280, 720))
        posts = _single_candidate_posts(post)
        if os.environ.get("PRM_PROMPT_FORMAT") == "1":   # match PRM-trained reward models
            system, posts = _to_prm_format(posts)
        scores = [_score(system, pre, p, img) for p in posts]
        if os.environ.get("ARBITER_EMPTY_CACHE") == "1":
            # cap the caching allocator's peak-hold so the arbiter fits alongside the actor on
            # ONE GPU (co-located rollout). Frees reserved-but-unused VRAM between states; the
            # tiny latency is negligible next to the per-step web + actor-generation time.
            torch.cuda.empty_cache()
        sel = int(max(range(len(scores)), key=lambda i: scores[i])) + 1   # 1-indexed
        if IO_LOG:   # one JSONL line per arbiter call = full input (prompt) + output (scalar) per candidate
            rec = {"step": _STEP[0], "system": system, "user_pre": pre, "selection": sel,
                   "candidates": [{"prompt": posts[i], "score": scores[i]} for i in range(len(posts))]}
            with open(IO_LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            _STEP[0] += 1
        content = json.dumps({"thought": "reward-model argmax", "selection": sel})
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "model": body.get("model", "reward-arbiter")}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": os.environ.get("SERVED_NAME", "reward-arbiter")}]}

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
