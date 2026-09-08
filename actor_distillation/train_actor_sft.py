"""
On-policy selection-distillation, STAGE D: SFT the ACTOR (MolmoWeb-4B) to generate
the arbiter-selected action. Standard causal-LM SFT (CE on the target tokens, prompt
masked). The actor's native prompt is reused verbatim from the candidate file, so
the training distribution matches generation.

Env: TRAIN_DATA (stage-C jsonl), TRAIN_OUT, ACTOR_MODEL(=allenai/MolmoWeb-4B),
     ACTOR_REVISION(=refs/pr/1), LORA_TARGETS(=all-linear), LORA_EXCLUDE, EPOCHS, BS, GA.

NOTE: MolmoWeb is a *different* model from the Qwen arbiter. mm-field handling in
collate mirrors the proven pilot_actor_infer tokenization; adjust if the molmo2
processor emits differently-named image tensors.
"""
import os, json, torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from PIL import Image

MODEL = os.environ.get("ACTOR_MODEL", "allenai/MolmoWeb-4B")
REVISION = os.environ.get("ACTOR_REVISION", "refs/pr/1")
DATA = os.environ.get("TRAIN_DATA", "output/onpolicy/round0/actor_sft.jsonl")
OUT = os.environ.get("TRAIN_OUT", "output/onpolicy/round0/actor_lora")
SMOKE = os.environ.get("SMOKE") == "1"
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "0"))

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, revision=REVISION)
tok = processor.tokenizer
SEQ = ("input_ids", "attention_mask", "labels")


def make_example(r):
    img = Image.open(r["screenshot"]).convert("RGB")
    user = [{"type": "text", "text": r["prompt"]}, {"type": "image", "image": img}]
    msgs_prompt = [{"role": "user", "content": user}]
    msgs_full = msgs_prompt + [{"role": "assistant", "content": [{"type": "text", "text": r["target"]}]}]
    full = processor.apply_chat_template(msgs_full, tokenize=True, return_dict=True,
                                         return_tensors="pt", add_generation_prompt=False)
    plen = processor.apply_chat_template(msgs_prompt, tokenize=True, return_dict=True,
                                         return_tensors="pt", add_generation_prompt=True)["input_ids"].shape[1]
    ex = {}
    for k, v in full.items():
        if k == "token_type_ids":
            continue
        ex[k] = v[0] if (k in ("input_ids", "attention_mask", "mm_token_type_ids")) else v
    labels = ex["input_ids"].clone(); labels[:plen] = -100
    ex["labels"] = labels
    return ex


class DS(Dataset):
    def __init__(self, path): self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return make_example(self.rows[i])


def collate(batch):
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    PADV = {"input_ids": tok.pad_token_id or 0, "attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}
    seq_keys = [k for k in PADV if k in batch[0]]
    out = {k: [] for k in seq_keys}; extra = {}
    for b in batch:
        for k in seq_keys:
            v = b[k]; pad = maxlen - v.shape[0]
            if pad > 0:
                v = torch.cat([v, torch.full((pad,), PADV[k], dtype=v.dtype)])
            out[k].append(v)
        for k, v in b.items():           # image / mm tensors -> concatenate along dim 0
            if k not in seq_keys and torch.is_tensor(v):
                extra.setdefault(k, []).append(v)
    res = {k: torch.stack(v) for k, v in out.items()}
    for k, vs in extra.items():
        res[k] = torch.cat(vs, dim=0)
    return res


def main():
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", revision=REVISION)
    base.config.use_cache = False
    _t = os.environ.get("LORA_TARGETS", "all-linear")
    cfg = dict(r=int(os.environ.get("LORA_R", "16")), lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
               lora_dropout=0.05, target_modules=("all-linear" if _t == "all-linear" else _t.split(",")),
               task_type="CAUSAL_LM")
    cfg["exclude_modules"] = os.environ.get("LORA_EXCLUDE", ".*(vision|visual|vit|image_proj|connector).*")
    model = get_peft_model(base, LoraConfig(**cfg))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # no enable_input_require_grads: molmo2 merges image tokens into inputs_embeds in-place,
    # which fails if the embeddings require grad. use_reentrant=False still reaches LoRA params.
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT, per_device_train_batch_size=int(os.environ.get("BS", "1")),
        gradient_accumulation_steps=1 if SMOKE else int(os.environ.get("GA", "8")),
        num_train_epochs=float(os.environ.get("EPOCHS", "1")),
        max_steps=3 if SMOKE else int(os.environ.get("MAX_STEPS", "-1")), learning_rate=1e-4,
        warmup_ratio=0.03, logging_steps=10, bf16=True, gradient_checkpointing=False, report_to=[],
        save_strategy="no" if SMOKE else ("steps" if SAVE_STEPS > 0 else "epoch"),
        save_steps=SAVE_STEPS if SAVE_STEPS > 0 else 500, remove_unused_columns=False)
    tr = Trainer(model=model, args=args, train_dataset=DS(DATA), data_collator=collate)
    resume = bool([d for d in os.listdir(OUT) if d.startswith("checkpoint-")]) if os.path.isdir(OUT) else False
    tr.train(resume_from_checkpoint=resume)
    tr.save_model(OUT)
    print(f"[actor-sft] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
