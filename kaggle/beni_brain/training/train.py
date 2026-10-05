"""QLoRA SFT then DPO on one T4 (§11.12). Standalone: runs in the training venv, never imported by the gateway.

    python train.py --base Qwen/Qwen2.5-VL-7B-Instruct --sft sft.jsonl [--dpo dpo.jsonl] --out adapter/

4-bit NF4 base, FP16 compute (T4 has no BF16), LoRA r=16 on the language model's attention + MLP projections only
(the vision tower is untouched, so vLLM can serve the adapter on the VL model). DPO continues the SFT adapter with a
frozen copy of it as the reference. Pins: transformers <4.50, peft 0.14, trl 0.15 (see brain_notebook.py).
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TARGET = r".*model\.(language_model\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def base_model(name):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
    kw = dict(quantization_config=q, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    if "VL" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    else:
        M = AutoModelForCausalLM
    m = M.from_pretrained(name, **kw)
    m.config.use_cache = False
    return prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)


def common(out):
    return dict(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=8, fp16=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=10, save_strategy="no", report_to=[], optim="paged_adamw_8bit",
                lr_scheduler_type="cosine", warmup_ratio=0.05, seed=0)


def run_sft(a, tok):
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    model = get_peft_model(base_model(a.base), LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                                         target_modules=TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    ds = Dataset.from_list([{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows(a.sft)])
    coll = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tok)    # loss on replies only
    cfg = SFTConfig(num_train_epochs=a.epochs, learning_rate=2e-4, max_seq_length=a.seq, dataset_text_field="text",
                    packing=False, **common(a.out + "/sft"))
    SFTTrainer(model=model, args=cfg, train_dataset=ds, data_collator=coll, processing_class=tok).train()
    model.save_pretrained(a.out)
    return model


def run_dpo(a, tok, model=None):
    import trl.trainer.dpo_trainer as D
    from trl import DPOConfig, DPOTrainer
    D.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}     # text-only rows: keep trl off its image-processor path for the VL
    if model is None:                               # no SFT this round: continue the adapter given with --init
        model = base_model(a.base)
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True, adapter_name="default")
    src = a.out if os.path.exists(os.path.join(a.out, "adapter_config.json")) else a.init
    model.load_adapter(src, adapter_name="reference")
    ds = Dataset.from_list(rows(a.dpo))
    cfg = DPOConfig(beta=0.1, learning_rate=5e-6, num_train_epochs=1, max_length=a.seq, max_prompt_length=a.seq - 160,
                    model_adapter_name="default", ref_adapter_name="reference", **common(a.out + "/dpo"))
    DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
    model.save_pretrained(a.out, selected_adapters=["default"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft")
    ap.add_argument("--dpo")
    ap.add_argument("--init", help="existing adapter dir to continue (DPO-only rounds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    a = ap.parse_args()
    if not a.sft and not (a.dpo and a.init):
        ap.error("need --sft, or --dpo with --init")
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = run_sft(a, tok) if a.sft else None
    if a.dpo:
        run_dpo(a, tok, model)
    print("adapter written to", a.out)


if __name__ == "__main__":
    main()
