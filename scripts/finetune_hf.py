"""Phase 8 on a REAL open-weight model: QLoRA SFT with TRL + PEFT + bitsandbytes.

Needs a CUDA GPU (a free Colab T4 works):
    pip install transformers peft trl bitsandbytes accelerate datasets
    python scripts/finetune_hf.py --model Qwen/Qwen2.5-1.5B-Instruct

It reuses finmind.data for the dataset, so the from-scratch LoRA in finmind/lora.py and this
script train on the same examples -- compare the maths there with the library here."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from finmind import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", default="artifacts/qlora-adapter")
args = ap.parse_args()

train, held = D.sft_examples(D.make_invoices(300))
to_chat = lambda xs: Dataset.from_list([{"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": r}]} for p, r in xs])

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
tok = AutoTokenizer.from_pretrained(args.model)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=to_chat(train), eval_dataset=to_chat(held),
                     peft_config=peft_cfg,
                     args=SFTConfig(output_dir=args.out, num_train_epochs=3, per_device_train_batch_size=4,
                                    gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
                                    logging_steps=10, eval_strategy="epoch", save_strategy="epoch", bf16=True))
trainer.train()
trainer.save_model(args.out)
print("adapter saved to", args.out, "| merge with: peft_model.merge_and_unload()")
