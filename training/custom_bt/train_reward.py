"""
Train a POINTWISE reward model: 4B VLM + LoRA + a scalar value head on the last
token, f(state, single action) -> scalar. Two objectives, selected by --objective:

  bt          Bradley-Terry pairwise (RLHF-standard): for each state, all candidate
              pairs with PRM score gap > --margin-eps contribute -logσ(r_w - r_l - m),
              m = --margin-scale * (score_w - score_l). + L2 anti-drift on r.
  regression  Huber on sigmoid(r) vs the PRM scalar (distil absolute scores).

Data: build_reward_data.py output (per state: 5 single-action prompts + scores).
Reward heads are NOT vLLM-servable -> eval/inference via HF (see eval_reward.py).

Env mirrors train_nocot_soft_lora (TRAIN_DATA/TRAIN_OUT/LORA_*/BS/GA/EPOCHS/...);
--objective also reads $OBJECTIVE so the qsub can pass it.
"""
import os, json, argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from PIL import Image

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")
REVISION = os.environ.get("BASE_REVISION") or None
# MolmoWeb (the actor backbone) as the reward model: different processor/mm-keys, no system
# role (image-first single user turn), and the embeddings/vision MUST stay frozen so Molmo2's
# in-place embed-merge doesn't break autograd (verified in smoke_molmoweb_reward.py).
IS_MOLMO = "molmo" in MODEL.lower()
DATA = os.environ.get("TRAIN_DATA", "output/prm_offline_pilot/train_full/reward_data.jsonl")
OUT = os.environ.get("TRAIN_OUT", "output/prm_offline_pilot/train_full/reward_bt")
MAXLEN = int(os.environ.get("MAXLEN", "3072"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "0"))
SMOKE = os.environ.get("SMOKE") == "1"

_proc_kw = {"trust_remote_code": True}
if REVISION:
    _proc_kw["revision"] = REVISION
if IS_MOLMO:
    _proc_kw["padding_side"] = "left"
processor = AutoProcessor.from_pretrained(MODEL, **_proc_kw)
tok = processor.tokenizer
SEQ = ("input_ids", "attention_mask", "mm_token_type_ids")
# image inputs the backbone forward accepts (Qwen: grid_thw; Molmo: pooling/grids/num_crops)
MM_KEYS = ("pixel_values", "image_grid_thw", "mm_token_type_ids",
           "image_token_pooling", "image_grids", "image_num_crops")


def molmo_conv(system, user_pre, user_post, img):
    """MolmoWeb single user turn, image FIRST, no system role (its chat template requires
    alternating user/assistant). Same text content as the Qwen prompt, concatenated."""
    text = f"{system}\n\n{user_pre}{user_post}"
    return [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": text}]}]


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


def make_example(r):
    img = Image.open(r["screenshot"]).convert("RGB")
    scores = torch.tensor([c["score"] for c in r["cands"]], dtype=torch.float32)
    if IS_MOLMO:   # defer tokenization to collate (processor batches Molmo's variable crops)
        return {"convs": [molmo_conv(c["system"], c["user_pre"], c["user_post"], img) for c in r["cands"]],
                "scores": scores}
    cands = [build_cand(c["system"], c["user_pre"], c["user_post"], img) for c in r["cands"]]
    return {"cands": cands, "scores": scores}


class DS(Dataset):
    def __init__(self, path): self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return make_example(self.rows[i])


def collate_molmo(batch):
    """Batch all B*K candidate conversations through the Molmo processor at once -- it handles
    the variable per-image crop tensors (pixel_values/image_token_pooling/...) natively, which
    is far cleaner than hand-collating them. last_pos = index of the last real token per row."""
    convs = [conv for st in batch for conv in st["convs"]]
    enc = processor.apply_chat_template(convs, tokenize=True, return_dict=True, return_tensors="pt",
                                        add_generation_prompt=True, padding=True)
    res = {k: v for k, v in enc.items() if k != "token_type_ids" and torch.is_tensor(v)}
    am = res["attention_mask"]
    res["last_pos"] = (am.shape[1] - 1 - am.flip(dims=[1]).argmax(dim=1)).long()   # robust to pad side
    res["scores"] = torch.stack([st["scores"] for st in batch])
    res["K"] = len(batch[0]["convs"])
    return res


def collate(batch):
    if IS_MOLMO:
        return collate_molmo(batch)
    flat = [c for st in batch for c in st["cands"]]               # B*K candidate prompts
    K = len(batch[0]["cands"])
    maxlen = max(c["input_ids"].shape[0] for c in flat)
    PADV = {"input_ids": tok.pad_token_id or 0, "attention_mask": 0, "mm_token_type_ids": 0}
    seq_keys = [k for k in PADV if k in flat[0]]
    out = {k: [] for k in seq_keys}; pix, grid, last = [], [], []
    for c in flat:
        for k in seq_keys:
            v = c[k]; pad = maxlen - v.shape[0]
            if pad > 0:
                v = torch.cat([v, torch.full((pad,), PADV[k], dtype=v.dtype)])
            out[k].append(v)
        if "pixel_values" in c: pix.append(c["pixel_values"])
        if "image_grid_thw" in c: grid.append(c["image_grid_thw"])
        last.append(c["last_pos"])
    res = {k: torch.stack(v) for k, v in out.items()}
    if pix: res["pixel_values"] = torch.cat(pix, dim=0)
    if grid: res["image_grid_thw"] = torch.cat(grid, dim=0)
    res["last_pos"] = torch.tensor(last, dtype=torch.long)        # [B*K]
    res["scores"] = torch.stack([st["scores"] for st in batch])   # [B,K]
    res["K"] = K
    return res


class RewardModel(nn.Module):
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


class RewardTrainer(Trainer):
    def __init__(self, *a, objective="bt", margin_eps=0.05, margin_scale=0.0, l2=1e-3, soft_k=6.0, **k):
        super().__init__(*a, **k)
        self.objective, self.margin_eps, self.margin_scale, self.l2 = objective, margin_eps, margin_scale, l2
        self.soft_k = soft_k

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        K = inputs.pop("K"); scores = inputs.pop("scores")        # [B,K]
        r = model(**inputs).view(scores.shape[0], K)              # [B,K]
        scores = scores.to(r.device)
        if self.objective == "regression":
            loss = F.huber_loss(torch.sigmoid(r), scores, delta=0.1)
        elif self.objective == "soft_bt":
            # soft Bradley-Terry: target preference prob from the (calibrated) PRM scores,
            # p(i>j) = sigmoid(K * (s_i - s_j))  -- same softmax(PRM*K) sharpness as the selector
            # (K = self.soft_k). BCE against the soft target: keeps the score MAGNITUDE, handles
            # near-ties (p~0.5), no hard threshold. ALL pairs within each state.
            terms = []
            for b in range(r.shape[0]):
                for i in range(K):
                    for j in range(i + 1, K):
                        p = torch.sigmoid(self.soft_k * (scores[b, i] - scores[b, j]))
                        lg = r[b, i] - r[b, j]
                        terms.append(-(p * F.logsigmoid(lg) + (1 - p) * F.logsigmoid(-lg)))
            loss = (torch.stack(terms).mean() if terms else r.sum() * 0.0) + self.l2 * (r ** 2).mean()
        else:  # bradley-terry (hard), all pairs within each state with a score gap
            terms = []
            for b in range(r.shape[0]):
                for i in range(K):
                    for j in range(K):
                        gap = (scores[b, i] - scores[b, j]).item()
                        if gap > self.margin_eps:                  # i is the winner
                            m = self.margin_scale * gap
                            terms.append(-F.logsigmoid(r[b, i] - r[b, j] - m))
            loss = (torch.stack(terms).mean() if terms else r.sum() * 0.0) + self.l2 * (r ** 2).mean()
        out = type("O", (), {"loss": loss})()
        return (loss, out) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.backbone.save_pretrained(output_dir)           # LoRA adapter
        torch.save(self.model.value_head.state_dict(), os.path.join(output_dir, "value_head.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Resume model weights from a checkpoint. We save the LoRA adapter + value head
        separately (see _save), so HF's default loader (which expects a full-model
        safetensors index) raises. Reload our two pieces explicitly instead; the Trainer
        restores optimizer/scheduler/global_step separately via _load_optimizer_and_scheduler.

        NOTE: we deliberately do NOT call peft.set_peft_model_state_dict. Under a DDP launch
        (torch.distributed initialized) it unconditionally calls _maybe_shard_state_dict_for_tp,
        which imports `EmbeddingParallel` from transformers.integrations.tensor_parallel -- a
        symbol absent in transformers 4.57.x (the MolmoWeb env) -> ImportError on resume. We use
        plain LoRA + DDP with NO tensor parallelism, so that sharding is a no-op; we replicate
        only the adapter-name key insertion and load the weights directly."""
        from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
        from safetensors.torch import load_file
        m = model if model is not None else self.model
        adp = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        sd = _insert_adapter_name_into_state_dict(
            load_file(adp), adapter_name="default", parameter_prefix="lora_")
        res = m.backbone.load_state_dict(sd, strict=False)
        # base-model weights are absent from the adapter file (expected missing); any UNEXPECTED
        # key means a remap mismatch -> fail loudly rather than silently resume from random LoRA.
        assert not res.unexpected_keys, f"unexpected adapter keys on resume: {res.unexpected_keys[:5]}"
        assert sd, "adapter state dict empty -- resume would reset LoRA to init"
        m.value_head.load_state_dict(
            torch.load(os.path.join(resume_from_checkpoint, "value_head.pt"), map_location="cpu"))
        print(f"[reward] resumed {len(sd)} LoRA tensors + value head from {resume_from_checkpoint}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default=os.environ.get("OBJECTIVE", "bt"), choices=["bt", "soft_bt", "regression"])
    ap.add_argument("--margin-eps", type=float, default=float(os.environ.get("MARGIN_EPS", "0.05")))
    ap.add_argument("--margin-scale", type=float, default=float(os.environ.get("MARGIN_SCALE", "0.0")))
    ap.add_argument("--l2", type=float, default=float(os.environ.get("REWARD_L2", "1e-3")))
    a = ap.parse_args()
    print(f"[reward] objective={a.objective} margin_eps={a.margin_eps} margin_scale={a.margin_scale} l2={a.l2}", flush=True)

    # DDP (torchrun) sets WORLD_SIZE>1: each rank loads the FULL model on its own GPU and
    # the Trainer wraps it in DistributedDataParallel. device_map="auto" is single-GPU only
    # (it shards the model = model-parallel, which is incompatible with DDP).
    _ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    _lr = int(os.environ.get("LOCAL_RANK", "0"))
    if _ddp:
        torch.cuda.set_device(_lr)   # pin this rank's device BEFORE any CUDA init (fused kernels)
    # DDP: load on CPU (device_map=None) and let the Trainer move to cuda:LOCAL_RANK + wrap DDP.
    # Single-GPU: device_map="auto" places the model directly.
    _load_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                    device_map=(None if _ddp else "auto"))
    if REVISION:
        _load_kw["revision"] = REVISION
    base = AutoModelForImageTextToText.from_pretrained(MODEL, **_load_kw)
    base.config.use_cache = False
    _t = os.environ.get("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj")
    cfg = dict(r=int(os.environ.get("LORA_R", "16")), lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
               lora_dropout=0.05, target_modules=("all-linear" if _t == "all-linear" else _t.split(",")),
               task_type="FEATURE_EXTRACTION")
    if os.environ.get("LORA_EXCLUDE"):
        cfg["exclude_modules"] = os.environ["LORA_EXCLUDE"]
    backbone = get_peft_model(base, LoraConfig(**cfg))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # enable_input_require_grads makes the embedding OUTPUT require grad -- for Molmo2 that tensor
    # is then modified in-place by the embed-merge (build_input_embeddings), which breaks autograd.
    # With use_reentrant=False grad checkpointing still backprops through the frozen embeddings, so
    # skip it for Molmo. (Qwen needs it for grad-checkpointing + LoRA.)
    if not IS_MOLMO:
        backbone.enable_input_require_grads()
    backbone.print_trainable_parameters()
    hidden = getattr(base.config, "hidden_size", None) or base.config.text_config.hidden_size
    model = RewardModel(backbone, hidden)

    args = TrainingArguments(
        output_dir=OUT, per_device_train_batch_size=int(os.environ.get("BS", "1")),
        gradient_accumulation_steps=1 if SMOKE else int(os.environ.get("GA", "16")),
        num_train_epochs=float(os.environ.get("EPOCHS", "2")),
        max_steps=3 if SMOKE else int(os.environ.get("MAX_STEPS", "-1")),
        learning_rate=float(os.environ.get("LR", "1e-4")),
        warmup_ratio=0.03, logging_steps=10, bf16=True, report_to=[],
        save_strategy="no" if SMOKE else ("steps" if SAVE_STEPS > 0 else "epoch"),
        save_steps=SAVE_STEPS if SAVE_STEPS > 0 else 500, remove_unused_columns=False, dataloader_num_workers=2,
        ddp_find_unused_parameters=False)   # LoRA: all trainable params used each step
    tr = RewardTrainer(model=model, args=args, train_dataset=DS(DATA), data_collator=collate,
                       objective=a.objective, margin_eps=a.margin_eps, margin_scale=a.margin_scale, l2=a.l2,
                       soft_k=float(os.environ.get("SOFT_K", "6.0")))
    # Resume from the latest checkpoint if present. RewardTrainer._load_from_checkpoint (above)
    # reloads the LoRA adapter + value head; the Trainer restores optimizer/scheduler/global_step.
    ckpts = (sorted([d for d in os.listdir(OUT) if d.startswith("checkpoint-")],
                    key=lambda d: int(d.split("-")[-1])) if os.path.isdir(OUT) else [])
    resume = os.path.join(OUT, ckpts[-1]) if ckpts else False
    if resume:
        print(f"[reward] resuming from {resume}", flush=True)
    tr.train(resume_from_checkpoint=resume)
    tr.save_model(OUT)
    print(f"[reward] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
