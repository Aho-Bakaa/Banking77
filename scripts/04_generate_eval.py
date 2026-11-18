"""Phase 4a (GPU, run on Kaggle): generate raw model outputs for evaluation.

Scoring is done separately and locally (04_score_eval.py) so metrics can be iterated on without a GPU.

For every (eval set, mode) pair this writes `<out_dir>/<set>__<mode>.jsonl`, one record per query:
  id, text, gold (intent or None), candidates_gold (ambiguity set), output (full assistant text),
  n_new_tokens, token_logprobs [[token_str, logprob], ...], seconds_per_query (batch time / batch size)

Modes
  free   model chooses <fast> or <think> itself (the real system)
  fast   assistant turn is prefilled with "<fast>"   -> "never think" ablation
  think  assistant turn is prefilled with "<think>"  -> "always think" ablation
  (use --adapter none for the untuned base-model zero-shot baseline)

Eval sets
  b77_test      Banking77 test.csv (3,080)
  amb_test      synthetic held-out ambiguity set (200)
  b77_train500  500 random Banking77 *train* queries (memorisation / train-test gap check)

Usage on Kaggle (inputs attached as datasets / kernel output):
  python scripts/04_generate_eval.py --adapter /kaggle/input/<adapter_dir> --data-dir /kaggle/input/<eval_data> \
      --out-dir /kaggle/working/eval --modes free fast think
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PREFILL = {"free": "", "fast": "<fast>", "think": "<think>"}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def system_prompt(data_dir: Path) -> str:
    """Reuse the exact system prompt from training so train/eval formats cannot drift."""
    with open(data_dir / "train.jsonl", encoding="utf-8") as handle:
        return json.loads(handle.readline())["messages"][0]["content"]


def build_sets(data_dir: Path, limit: int | None) -> dict[str, list[dict]]:
    test = pd.read_csv(data_dir / "test.csv")
    sets = {"b77_test": [{"id": f"test-{i:04d}", "text": t, "gold": c, "candidates_gold": None}
                         for i, (t, c) in enumerate(zip(test["text"], test["category"]))]}
    amb = load_jsonl(data_dir / "ambiguous_test.jsonl")
    sets["amb_test"] = [{"id": r["id"], "text": r["text"], "gold": None, "candidates_gold": r["candidates"],
                         "family": r["family"]} for r in amb]
    train = pd.read_csv(data_dir / "train.csv").sample(500, random_state=77)
    sets["b77_train500"] = [{"id": f"train-{i}", "text": t, "gold": c, "candidates_gold": None}
                            for i, t, c in zip(train.index, train["text"], train["category"])]
    if limit:
        sets = {name: rows[:limit] for name, rows in sets.items()}
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA adapter dir, or 'none' for the base model")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["free", "fast", "think"], choices=list(PREFILL))
    parser.add_argument("--sets", nargs="+", default=["b77_test", "amb_test", "b77_train500"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap queries per set")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(77)
    torch.manual_seed(77)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16),
        device_map={"": 0},
    )
    if args.adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    system = system_prompt(args.data_dir)
    eval_sets = build_sets(args.data_dir, args.limit)

    for set_name in args.sets:
        rows = eval_sets[set_name]
        for mode in args.modes:
            out_path = args.out_dir / f"{set_name}__{mode}.jsonl"
            if out_path.exists():
                print(f"skip {out_path.name} (exists)")
                continue
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": row["text"].strip()}],
                    tokenize=False, add_generation_prompt=True) + PREFILL[mode]
                for row in rows
            ]
            order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # length-sorted batches are faster
            results: dict[int, dict] = {}
            for start in range(0, len(order), args.batch_size):
                batch_idx = order[start:start + args.batch_size]
                encoded = tokenizer([prompts[i] for i in batch_idx], return_tensors="pt", padding=True).to(model.device)
                begin = time.time()
                with torch.no_grad():
                    generated = model.generate(
                        **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                        output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id)
                seconds = (time.time() - begin) / len(batch_idx)
                scores = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True)
                new_tokens = generated.sequences[:, encoded["input_ids"].shape[1]:]
                for row_pos, i in enumerate(batch_idx):
                    ids = new_tokens[row_pos].tolist()
                    keep = [t for t in range(len(ids)) if ids[t] != tokenizer.pad_token_id]
                    token_logprobs = [[tokenizer.decode(ids[t]), round(float(scores[row_pos, t]), 4)] for t in keep]
                    text = tokenizer.decode([ids[t] for t in keep], skip_special_tokens=True)
                    results[i] = {**rows[i], "mode": mode, "output": PREFILL[mode] + text,
                                  "n_new_tokens": len(keep), "token_logprobs": token_logprobs,
                                  "seconds_per_query": round(seconds, 4)}
                if (start // args.batch_size) % 10 == 0:
                    print(f"{set_name}/{mode}: {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
            with open(out_path, "w", encoding="utf-8") as handle:
                for i in range(len(rows)):
                    handle.write(json.dumps(results[i], ensure_ascii=False) + "\n")
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
