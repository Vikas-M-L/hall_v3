#!/usr/bin/env python3
"""Qwen LoRA finetune for M0-M4 diagnosis — GPU ONLY, never run on CPU.

    base (Qwen2.5-VL-3B/7B, 4-bit, FROZEN) + LoRA adapters -> M0-M4 token

What it does
------------
Supervised finetune on JSONL from finetune/build_lora_data.py
(image path + prompt + single-token label). Only adapter weights update;
the base model never moves. Loss is computed on the label token only.

Run (Colab A100 / 24GB card)
-----------------------------
    pip install -r requirements.txt peft bitsandbytes
    python finetune/finetune_qwen_lora.py \\
        --data data/lora/train.jsonl --val data/lora/val.jsonl \\
        --base Qwen/Qwen2.5-VL-3B-Instruct --out adapters/m0m4-r16

Evaluate: scripts compare adapter vs rule-based diagnose() on held-out
mechanism accuracy + the routing table (repair/policy.py). If the adapter
does not beat rules + outcome-policy control, do not ship it (plan section 7).

Do NOT run this file on CPU: 4-bit Qwen still needs CUDA for training, and
bitsandbytes import itself fails without it. All heavy imports live inside
main() so importing this module (e.g. for inspection) is always safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: str) -> list[dict]:
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert recs and all(set(r) >= {"image", "prompt", "label"} for r in recs)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="adapters/m0m4-r16")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    raise RuntimeError(
        "Legacy LoRA prototype is quarantined by research_audit.md: overlapping "
        "train/validation rows, unmasked prompt labels and incomplete multimodal "
        "collation were found. No validated adapter training run exists. Use the "
        "audited CPU diagnosis experiment until this trainer is repaired and "
        "verified on a GPU; see configs/lora_protocol.json."
    )

    import torch  # noqa: E402
    if not torch.cuda.is_available():
        raise RuntimeError("REFUSING to train on CPU: LoRA finetune needs CUDA. "
                           "Run on a GPU machine (see header).")

    from datasets import Dataset  # noqa: E402
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
    from transformers import (AutoProcessor, BitsAndBytesConfig,  # noqa: E402
                              Qwen2_5_VLForConditionalGeneration, TrainingArguments,
                              Trainer, DataCollatorForSeq2Seq)

    train_recs = load_records(args.data)
    val_recs = load_records(args.val) if args.val else train_recs[-max(1, len(train_recs) // 10):]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(args.base, max_pixels=1003520)

    def render(rec: dict) -> dict:
        from PIL import Image

        img = Image.open(rec["image"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": rec["prompt"]}]},
                    {"role": "assistant", "content": [{"type": "text",
                                                       "text": rec["label"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False)
        enc = processor(text=[text], images=[img], return_tensors="pt",
                        padding="max_length", truncation=True, max_length=512)
        labels = enc["input_ids"].clone()
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                "pixel_values": enc["pixel_values"][0], "labels": labels[0]}

    train_ds = Dataset.from_list([render(r) for r in train_recs])
    val_ds = Dataset.from_list([render(r) for r in val_recs])
    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs,
                              learning_rate=args.lr, per_device_train_batch_size=args.batch,
                              gradient_accumulation_steps=4, bf16=True,
                              evaluation_strategy="epoch", save_strategy="epoch",
                              logging_steps=10, report_to="none")
    Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=DataCollatorForSeq2Seq(processor.tokenizer)).train()
    model.save_pretrained(args.out)
    print("adapters saved to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
