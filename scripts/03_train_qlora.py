"""Phase 3: Fine-Tune Qwen-2.5-3B-Instruct using QLoRA.

Features:
  - 4-bit NormalFloat (NF4) quantization via BitsAndBytes.
  - LoRA adaptation on all projection layers (q, k, v, o, gate, up, down).
  - Effective batch size: 16 (per_device=4, grad_accum=4).
  - Adaptive learning rate: Cosine annealing with 5% warmup.
  - Step-wise evaluation every 100 steps on val.jsonl.
  - Early stopping (patience=3 steps without eval_loss improvement).
  - Checkpoint saving (keeps last/best 3) and seamless resumption across sessions.
  - Saves the best adapter and tokenizer to output directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer

ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chatml_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen-2.5-3B with QLoRA on Banking77 CoT")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "training_config.yaml"))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint folder to resume")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def format_chatml_prompt(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    # Formats messages via tokenizer.apply_chat_template
    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted_text}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml_config(config_path)

    # Overrides from CLI if specified
    train_path = Path(args.train_data) if args.train_data else ROOT / cfg["paths"]["train_data"]
    val_path = Path(args.val_data) if args.val_data else ROOT / cfg["paths"]["val_data"]
    output_dir = args.output_dir or cfg["training"]["output_dir"]
    final_adapter_dir = cfg["paths"]["final_adapter_dir"]
    num_epochs = args.epochs or cfg["training"]["num_train_epochs"]
    learning_rate = args.lr or cfg["training"]["learning_rate"]
    max_seq_len = cfg["model"].get("max_seq_length", 640)

    print(f"=== QLoRA Training Pipeline ===")
    print(f"Base Model:       {cfg['model']['base_model_name']}")
    print(f"Train Dataset:    {train_path}")
    print(f"Val Dataset:      {val_path}")
    print(f"Output Checkpoints: {output_dir}")
    print(f"Learning Rate:    {learning_rate}")
    print(f"Epochs:           {num_epochs}")
    print(f"Max Seq Length:   {max_seq_len}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA is NOT available! Running on CPU will be extremely slow.")
        print("For training, please run this on Kaggle GPU (T4 / P100) or an active CUDA environment.")
        device_map = "auto"
        quant_config = None
        torch_dtype = torch.float32
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device_map = {"": 0}
        compute_dtype = getattr(torch, cfg["model"]["bnb_4bit_compute_dtype"])
        quant_config = BitsAndBytesConfig(
            load_in_4bit=cfg["model"]["load_in_4bit"],
            bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
        )
        torch_dtype = compute_dtype

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model_name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Datasets
    print("\nLoading and formatting datasets into ChatML...")
    train_records = load_chatml_jsonl(train_path)
    val_records = load_chatml_jsonl(val_path)
    print(f"Loaded {len(train_records)} train records and {len(val_records)} val records.")

    train_ds = Dataset.from_list(train_records).map(
        lambda ex: format_chatml_prompt(ex, tokenizer),
        desc="Applying chat template to train",
    )
    val_ds = Dataset.from_list(val_records).map(
        lambda ex: format_chatml_prompt(ex, tokenizer),
        desc="Applying chat template to val",
    )

    # 3. Load Base Model in 4-bit
    print(f"\nLoading base model: {cfg['model']['base_model_name']}...")
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(cfg["model"]["base_model_name"], trust_remote_code=True)
    config.torch_dtype = torch.float16
    config.use_cache = False
    model_kwargs["config"] = config

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_name"],
        **model_kwargs,
    )

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        )

    # 4. Attach LoRA Adapter
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)

    # Ensure NO parameters remain in bfloat16 (preventing GradScaler unscale crash on Tesla T4)
    for name, param in model.named_parameters():
        if "bfloat16" in str(param.dtype).lower():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(torch.float16)

    model.print_trainable_parameters()

    # 5. Training Arguments
    import inspect
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    desired_args = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": cfg["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "per_device_eval_batch_size": cfg["training"]["per_device_eval_batch_size"],
        "learning_rate": learning_rate,
        "lr_scheduler_type": cfg["training"]["lr_scheduler_type"],
        "warmup_steps": 38,
        "weight_decay": cfg["training"]["weight_decay"],
        "fp16": cfg["training"]["fp16"] and torch.cuda.is_available(),
        "bf16": cfg["training"]["bf16"] and torch.cuda.is_available(),
        "logging_steps": cfg["training"]["logging_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "save_steps": cfg["training"]["save_steps"],
        "save_total_limit": cfg["training"]["save_total_limit"],
        "load_best_model_at_end": cfg["training"]["load_best_model_at_end"],
        "metric_for_best_model": cfg["training"]["metric_for_best_model"],
        "greater_is_better": cfg["training"]["greater_is_better"],
        "gradient_checkpointing": cfg["training"]["gradient_checkpointing"],
        "seed": cfg["training"]["seed"],
        "report_to": "none",
    }
    if "eval_strategy" in ta_params:
        desired_args["eval_strategy"] = "steps"
        desired_args["save_strategy"] = "steps"
    elif "evaluation_strategy" in ta_params:
        desired_args["evaluation_strategy"] = "steps"
        desired_args["save_strategy"] = "steps"

    filtered_ta_args = {k: v for k, v in desired_args.items() if k in ta_params}
    training_args = TrainingArguments(**filtered_ta_args)

    # Early stopping callback
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg["early_stopping"]["patience"],
            early_stopping_threshold=cfg["early_stopping"]["threshold"],
        )
    ]

    # 6. Initialize SFTTrainer (dynamic signature matching)
    sft_sig = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    sft_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "callbacks": callbacks,
    }
    if "dataset_text_field" in sft_sig:
        sft_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_sig:
        sft_kwargs["max_seq_length"] = max_seq_len
    if "processing_class" in sft_sig:
        sft_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_sig:
        sft_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**sft_kwargs)

    # 7. Start Training (or Resume from Checkpoint)
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint:
        print(f"\nResuming training from checkpoint: {resume_checkpoint}...")
    else:
        # Check if checkpoints already exist in output_dir
        if Path(output_dir).exists():
            checkpoints = list(Path(output_dir).glob("checkpoint-*"))
            if checkpoints:
                latest_ckpt = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
                print(f"\nDetected existing checkpoint: {latest_ckpt}. Resuming from here...")
                resume_checkpoint = str(latest_ckpt)

    print("\nStarting model training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 8. Save Final Model & Metrics
    print(f"\nTraining complete! Saving best adapter and tokenizer to: {final_adapter_dir}")
    os.makedirs(final_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    train_metrics = trainer.state.log_history
    metrics_path = Path(final_adapter_dir) / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Logged training history to: {metrics_path}")
    print("\nPhase 3 QLoRA fine-tuning successfully finished!")


if __name__ == "__main__":
    main()
