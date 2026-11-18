"""Phase 2: Dataset filtering, deduplication, target formatting, and cluster-based splitting.

Execution steps (in order):
1. Assign each train query a mode:
   - 'think' if probe gold_prob < 0.8, teacher marked ambiguous, or has trace from hard tier.
   - Everything else is 'fast'.
   - Save to data/processed/gate_labels.jsonl.
2. Drop all 181 label_noise rows from training.
3. Fast set comes from all ~6k easy train queries without teacher, using gold label directly.
   - Embed with BAAI/bge-small-en-v1.5.
   - Cluster at cosine similarity > 0.85 (complete linkage, distance_threshold=0.15).
   - Keep at most 2 per cluster and at most about 60 per intent.
4. Build targets:
   - Think, resolved:  <think>{thought}</think>{"status":"resolved","intent":"x"}
   - Think, ambiguous: <think>{thought}</think>{"status":"ambiguous","candidates":[...],"question":"..."}
   - Fast:             <fast>{"status":"resolved","intent":"x"}
5. ChatML format with 77 labels listed once in system prompt.
   - Include the 805 synthetic ambiguous train items as think examples.
6. Set mixture by sampler:
   - About 50% hard think, 25% fast, 25% ambiguous, stratified per intent.
7. Validation split by cluster (not by row):
   - Cluster with bge-small at cosine > 0.85.
   - Take about 10% of clusters for val.jsonl, ~90% for train.jsonl.
   - The test sets stay untouched (Banking77 test.csv and ambiguous_test.jsonl).
8. Strict validation:
   - JSON in assistant message parses cleanly.
   - Every intent is one of the 77 valid categories.
   - Candidates are valid lists of 2-3 intents.
   - The thought doesn't leak the label.
9. Print token-length stats and per-intent counts.
10. Write outputs:
   - data/processed/train.jsonl
   - data/processed/val.jsonl
   - data/processed/gate_labels.jsonl
   - data/processed/stats.json
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SYNTH_DIR = ROOT / "data" / "synthetic"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "configs"

SEED = 77
LEAK_MARKERS = ("gold", "label is", "labelled as", "labeled as", "annotat", "ground truth", "the hint")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_system_prompt(categories: list[str]) -> str:
    sorted_cats = sorted(categories)
    return (
        "You are an expert banking intent router. "
        "Classify customer queries into one of the 77 banking intents, or determine if the query is ambiguous.\n\n"
        "Valid intents:\n" + ", ".join(sorted_cats)
    )


def deduplicate_fast_pool(
    easy_queries: list[dict[str, Any]],
    encoder: SentenceTransformer,
    max_per_cluster: int = 2,
    max_per_intent: int = 60,
) -> list[dict[str, Any]]:
    print(f"Embedding {len(easy_queries)} easy queries with bge-small for fast set deduplication...")
    texts = [q["text"] for q in easy_queries]
    embs = encoder.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    by_intent = defaultdict(list)
    for idx, q in enumerate(easy_queries):
        by_intent[q["gold_intent"]].append(idx)

    kept_indices: list[int] = []
    rng = random.Random(SEED)
    for intent, indices in sorted(by_intent.items()):
        if len(indices) <= 1:
            kept_indices.extend(indices)
            continue

        sub_embs = embs[indices]
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.15,  # cosine similarity > 0.85
            metric="cosine",
            linkage="complete",
        ).fit(sub_embs)

        clusters = defaultdict(list)
        for local_idx, cluster_id in enumerate(clustering.labels_):
            clusters[cluster_id].append(indices[local_idx])

        intent_kept: list[int] = []
        for cluster_id, member_indices in clusters.items():
            intent_kept.extend(member_indices[:max_per_cluster])

        if len(intent_kept) > max_per_intent:
            rng.shuffle(intent_kept)
            intent_kept = intent_kept[:max_per_intent]

        kept_indices.extend(intent_kept)

    deduped = [easy_queries[i] for i in kept_indices]
    print(f"Fast set deduplicated: {len(easy_queries)} -> {len(deduped)} queries across {len(by_intent)} intents")
    return deduped


def validate_chatml_record(
    record: dict[str, Any],
    valid_intents: set[str],
) -> list[str]:
    problems = []
    messages = record.get("messages", [])
    if len(messages) != 3:
        problems.append("record must have exactly 3 messages (system, user, assistant)")
        return problems

    system_msg, user_msg, assistant_msg = messages
    if system_msg.get("role") != "system" or user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
        problems.append("invalid message roles")

    content = assistant_msg.get("content", "")
    if content.startswith("<think>"):
        end_think = content.find("</think>")
        if end_think == -1:
            problems.append("missing </think> closing tag")
            return problems
        thought = content[len("<think>"):end_think]
        payload_str = content[end_think + len("</think>"):]

        if any(marker in thought.lower() for marker in LEAK_MARKERS):
            problems.append(f"leakage marker in thought: {thought}")

        try:
            payload = json.loads(payload_str)
        except Exception as e:
            problems.append(f"malformed payload JSON: {payload_str} ({e})")
            return problems

        status = payload.get("status")
        if status == "resolved":
            intent = payload.get("intent")
            if intent not in valid_intents:
                problems.append(f"invalid resolved intent: {intent}")
        elif status == "ambiguous":
            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or not (2 <= len(candidates) <= 3):
                problems.append(f"invalid candidates length: {candidates}")
            elif any(c not in valid_intents for c in candidates):
                problems.append(f"invalid candidate intent in: {candidates}")
            question = payload.get("question", "")
            if not isinstance(question, str) or not question.strip().endswith("?"):
                problems.append(f"question must be a non-empty string ending with '?': {question}")
        else:
            problems.append(f"unknown status: {status}")

    elif content.startswith("<fast>"):
        payload_str = content[len("<fast>"):]
        try:
            payload = json.loads(payload_str)
        except Exception as e:
            problems.append(f"malformed fast payload JSON: {payload_str} ({e})")
            return problems

        if payload.get("status") != "resolved":
            problems.append(f"fast status must be resolved, got: {payload.get('status')}")
        if payload.get("intent") not in valid_intents:
            problems.append(f"invalid fast intent: {payload.get('intent')}")
    else:
        problems.append(f"assistant message does not start with <think> or <fast>: {content[:30]}")

    return problems


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data and taxonomy
    categories = json.loads((RAW_DIR / "categories.json").read_text(encoding="utf-8"))
    valid_intents = set(categories)
    hardness_rows = load_jsonl(RAW_DIR / "train_hardness.jsonl")
    cot_rows = load_jsonl(SYNTH_DIR / "cot_train.jsonl")
    amb_synth_rows = load_jsonl(SYNTH_DIR / "ambiguous_train.jsonl")

    cot_by_id = {r["id"]: r for r in cot_rows}
    noise_ids = {r["id"] for r in cot_rows if r.get("flag") == "label_noise"}
    print(f"Loaded {len(hardness_rows)} train hardness rows, {len(cot_rows)} CoT rows, {len(noise_ids)} noise flags.")

    # 2. Assign each train query a mode -> data/processed/gate_labels.jsonl
    gate_labels = []
    think_ids = set()
    for row in hardness_rows:
        row_id = row["id"]
        cot = cot_by_id.get(row_id)
        is_hard_tier = (cot is not None and cot.get("tier") == "hard")
        gold_prob_under_08 = (row.get("gold_prob", 1.0) < 0.8)
        teacher_ambiguous = (cot is not None and cot.get("status") == "ambiguous")

        if gold_prob_under_08 or teacher_ambiguous or is_hard_tier:
            mode = "think"
            think_ids.add(row_id)
        else:
            mode = "fast"

        gate_labels.append(
            {
                "id": row_id,
                "text": row["text"],
                "gold_intent": row["gold_intent"],
                "mode": mode,
                "gold_prob": row.get("gold_prob"),
            }
        )

    save_jsonl(PROCESSED_DIR / "gate_labels.jsonl", gate_labels)
    gate_counts = Counter(r["mode"] for r in gate_labels)
    print(f"Saved gate labels: {len(gate_labels)} rows (think={gate_counts['think']}, fast={gate_counts['fast']})")

    # 3. Drop all 181 label_noise rows from training pools
    hard_think_pool = [
        r for r in cot_rows
        if r.get("tier") == "hard" and r.get("status") == "resolved" and r.get("flag") != "label_noise"
    ]
    cot_amb_clean = [
        r for r in cot_rows
        if r.get("status") == "ambiguous" and r.get("flag") != "label_noise"
    ]
    cot_ids = set(cot_by_id.keys())
    test_b77 = pd.read_csv(RAW_DIR / "test.csv")
    amb_test = load_jsonl(SYNTH_DIR / "ambiguous_test.jsonl")

    def normalise_for_leak_check(t: str) -> str:
        return re.sub(r"[^a-z0-9]", "", t.lower()).strip()

    test_norm_set = {normalise_for_leak_check(t) for t in test_b77["text"]}.union(
        {normalise_for_leak_check(r["text"]) for r in amb_test}
    )

    easy_queries = [
        h for h in hardness_rows
        if h["gold_prob"] >= 0.8
        and h["id"] not in cot_ids
        and h["id"] not in noise_ids
        and normalise_for_leak_check(h["text"]) not in test_norm_set
    ]

    print(f"Clean candidate pools:")
    print(f"  Hard think resolved: {len(hard_think_pool)}")
    print(f"  CoT ambiguous:        {len(cot_amb_clean)}")
    print(f"  Synthetic ambiguous:  {len(amb_synth_rows)}")
    print(f"  Easy queries (fast):  {len(easy_queries)}")

    # Deduplicate fast set with bge-small
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    fast_pool = deduplicate_fast_pool(easy_queries, encoder, max_per_cluster=2, max_per_intent=60)

    # 4. Target Formatting & Mixture by Sampler: ~50% hard think, ~25% fast, ~25% ambiguous
    # Stratified per intent
    system_prompt = build_system_prompt(categories)

    hard_by_intent = defaultdict(list)
    for r in hard_think_pool:
        hard_by_intent[r["gold_intent"]].append(r)

    fast_by_intent = defaultdict(list)
    for r in fast_pool:
        fast_by_intent[r["gold_intent"]].append(r)

    # Group ambiguous by primary candidate
    amb_by_intent = defaultdict(list)
    # Include all 805 synthetic ambiguous items
    for r in amb_synth_rows:
        amb_by_intent[r["candidates"][0]].append(r)
    # Include CoT ambiguous items
    for r in cot_amb_clean:
        primary = r.get("gold_intent") or r["candidates"][0]
        amb_by_intent[primary].append(r)

    # All 805 synthetic ambiguous items must be included
    all_synth_ids = {r["id"] for r in amb_synth_rows}
    used_synth_ids = set()

    sampled_examples: list[dict[str, Any]] = []
    rng = random.Random(SEED)

    # For each intent, determine quota so hard:fast:amb ~ 2:1:1
    # 77 intents * 26 hard = 2,002 (50%)
    # 77 intents * 13 fast = 1,001 (25%)
    # 77 intents * 13 amb  = 1,001 (25%)
    # Total ~ 4,004
    for intent in sorted(categories):
        h_pool = hard_by_intent[intent]
        f_pool = fast_by_intent[intent]
        a_pool = amb_by_intent[intent]

        target_hard = 26
        target_fast = 13
        target_amb = 13 if len(a_pool) > 0 else 0

        # Sample hard think (using available unique items, sampling with replacement if pool < target)
        if len(h_pool) >= target_hard:
            chosen_hard = rng.sample(h_pool, target_hard)
        elif len(h_pool) > 0:
            chosen_hard = list(h_pool)
            chosen_hard.extend(rng.choices(h_pool, k=target_hard - len(h_pool)))
        else:
            chosen_hard = []

        # Sample fast
        if len(f_pool) >= target_fast:
            chosen_fast = rng.sample(f_pool, target_fast)
        elif len(f_pool) > 0:
            chosen_fast = list(f_pool)
            chosen_fast.extend(rng.choices(f_pool, k=target_fast - len(f_pool)))
        else:
            chosen_fast = []

        # Sample ambiguous
        if len(a_pool) >= target_amb:
            chosen_amb = rng.sample(a_pool, target_amb)
        elif len(a_pool) > 0:
            chosen_amb = list(a_pool)
            chosen_amb.extend(rng.choices(a_pool, k=target_amb - len(a_pool)))
        else:
            chosen_amb = []

        for item in chosen_hard:
            sampled_examples.append({
                "source_id": item["id"],
                "text": item["text"],
                "intent": item["intent"],
                "mode": "think_resolved",
                "thought": item["thought"],
                "primary_intent": intent,
            })

        for item in chosen_fast:
            sampled_examples.append({
                "source_id": item["id"],
                "text": item["text"],
                "intent": item["gold_intent"],
                "mode": "fast",
                "thought": None,
                "primary_intent": intent,
            })

        for item in chosen_amb:
            if item.get("id") in all_synth_ids:
                used_synth_ids.add(item["id"])
            sampled_examples.append({
                "source_id": item["id"],
                "text": item["text"],
                "intent": None,
                "candidates": item["candidates"],
                "question": item["question"],
                "mode": "think_ambiguous",
                "thought": item["thought"],
                "primary_intent": intent,
            })

    # Include any remaining synthetic ambiguous train items that weren't selected yet
    for item in amb_synth_rows:
        if item["id"] not in used_synth_ids:
            primary = item["candidates"][0]
            sampled_examples.append({
                "source_id": item["id"],
                "text": item["text"],
                "intent": None,
                "candidates": item["candidates"],
                "question": item["question"],
                "mode": "think_ambiguous",
                "thought": item["thought"],
                "primary_intent": primary,
            })
            used_synth_ids.add(item["id"])

    # Verify all 805 synthetic ambiguous items are included
    assert len(used_synth_ids) == len(amb_synth_rows), f"Missing synthetic ambiguous items: {len(used_synth_ids)} vs {len(amb_synth_rows)}"

    mode_counts = Counter(r["mode"] for r in sampled_examples)
    total_sampled = len(sampled_examples)
    print("\n--- SAMPLED MIXTURE BREAKDOWN ---")
    for m, c in sorted(mode_counts.items()):
        print(f"  {m:15s}: {c:5d} ({c / total_sampled * 100:5.1f}%)")

    # 5. Format into ChatML records
    chatml_records = []
    for item in sampled_examples:
        mode = item["mode"]
        text = item["text"]

        if mode == "think_resolved":
            payload_str = json.dumps({"status": "resolved", "intent": item["intent"]}, separators=(",", ":"))
            assistant_content = f"<think>{item['thought']}</think>{payload_str}"
        elif mode == "think_ambiguous":
            payload_str = json.dumps(
                {"status": "ambiguous", "candidates": item["candidates"], "question": item["question"]},
                separators=(",", ":"),
            )
            assistant_content = f"<think>{item['thought']}</think>{payload_str}"
        elif mode == "fast":
            payload_str = json.dumps({"status": "resolved", "intent": item["intent"]}, separators=(",", ":"))
            assistant_content = f"<fast>{payload_str}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        record = {
            "source_id": item["source_id"],
            "primary_intent": item["primary_intent"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        chatml_records.append(record)

    # 6. Make validation split by cluster (not by row, ~10% of clusters)
    print(f"\nEmbedding {len(chatml_records)} items with bge-small to cluster for validation split...")
    all_texts = [r["messages"][1]["content"] for r in chatml_records]
    all_embs = encoder.encode(all_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    by_intent_records = defaultdict(list)
    for idx, r in enumerate(chatml_records):
        by_intent_records[r["primary_intent"]].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for intent, indices in sorted(by_intent_records.items()):
        if len(indices) <= 2:
            train_indices.extend(indices)
            continue

        sub_embs = all_embs[indices]
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.15,
            metric="cosine",
            linkage="complete",
        ).fit(sub_embs)

        clusters = defaultdict(list)
        for local_idx, cluster_id in enumerate(clustering.labels_):
            clusters[cluster_id].append(indices[local_idx])

        cluster_keys = list(clusters.keys())
        rng.shuffle(cluster_keys)

        num_val_clusters = max(1, round(len(cluster_keys) * 0.10))
        val_clusters = set(cluster_keys[:num_val_clusters])

        for c_id, member_indices in clusters.items():
            if c_id in val_clusters:
                val_indices.extend(member_indices)
            else:
                train_indices.extend(member_indices)

    train_records = [chatml_records[i] for i in train_indices]
    val_records = [chatml_records[i] for i in val_indices]

    print(f"\nCluster-based split completed:")
    print(f"  Train: {len(train_records)} ({len(train_records) / len(chatml_records) * 100:.1f}%)")
    print(f"  Val:   {len(val_records)} ({len(val_records) / len(chatml_records) * 100:.1f}%)")

    # 7. Strict validation
    print("\nValidating all records (JSON, 77 intents, candidates, label leakage)...")
    validation_errors = []
    for split_name, dataset in [("train", train_records), ("val", val_records)]:
        for idx, rec in enumerate(dataset):
            errs = validate_chatml_record(rec, valid_intents)
            if errs:
                validation_errors.append(f"{split_name}[{idx}]: " + "; ".join(errs))

    if validation_errors:
        print(f"VALIDATION FAILED with {len(validation_errors)} errors! First 5:")
        for err in validation_errors[:5]:
            print("  ", err)
        raise RuntimeError("Validation checks failed!")
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY (0 errors)!")

    # 8. Token statistics and per-intent counts
    print("\nCalculating token statistics with Qwen/Qwen2.5-3B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    token_lengths = []
    thought_lengths = []
    for r in train_records + val_records:
        full_text = "".join(m["content"] for m in r["messages"])
        tokens = tokenizer.encode(full_text)
        token_lengths.append(len(tokens))

        assistant_text = r["messages"][2]["content"]
        if "<think>" in assistant_text:
            thought_text = assistant_text.split("</think>")[0].replace("<think>", "")
            thought_lengths.append(len(thought_text.split()))

    per_intent_train = Counter(r["primary_intent"] for r in train_records)
    per_intent_val = Counter(r["primary_intent"] for r in val_records)

    stats = {
        "dataset": {
            "total_examples": len(chatml_records),
            "train_examples": len(train_records),
            "val_examples": len(val_records),
            "gate_labels_total": len(gate_labels),
            "gate_think": gate_counts["think"],
            "gate_fast": gate_counts["fast"],
            "dropped_label_noise": len(noise_ids),
        },
        "mixture_breakdown_overall": {
            mode: count for mode, count in Counter(r["mode"] for r in sampled_examples).items()
        },
        "token_lengths": {
            "mean": round(float(np.mean(token_lengths)), 1),
            "median": float(np.median(token_lengths)),
            "min": int(np.min(token_lengths)),
            "max": int(np.max(token_lengths)),
            "p95": round(float(np.percentile(token_lengths, 95)), 1),
        },
        "thought_word_lengths": {
            "mean": round(float(np.mean(thought_lengths)), 1),
            "median": float(np.median(thought_lengths)),
            "min": int(np.min(thought_lengths)),
            "max": int(np.max(thought_lengths)),
        },
        "per_intent_counts": {
            intent: {"train": per_intent_train[intent], "val": per_intent_val[intent]}
            for intent in sorted(categories)
        },
    }

    print("\n--- TOKEN LENGTH STATS ---")
    print(f"Mean tokens per ChatML item: {stats['token_lengths']['mean']}")
    print(f"Median tokens: {stats['token_lengths']['median']}")
    print(f"Min / Max: {stats['token_lengths']['min']} / {stats['token_lengths']['max']}")
    print(f"95th percentile: {stats['token_lengths']['p95']}")
    print(f"Mean thought word length: {stats['thought_word_lengths']['mean']} words")

    print("\n--- PER INTENT COUNTS (Summary) ---")
    train_counts_list = [per_intent_train[intent] for intent in sorted(categories)]
    val_counts_list = [per_intent_val[intent] for intent in sorted(categories)]
    print(f"Train per intent: min={min(train_counts_list)}, max={max(train_counts_list)}, mean={np.mean(train_counts_list):.1f}")
    print(f"Val per intent:   min={min(val_counts_list)}, max={max(val_counts_list)}, mean={np.mean(val_counts_list):.1f}")

    # Strip auxiliary fields before saving
    clean_train = [{"messages": r["messages"]} for r in train_records]
    clean_val = [{"messages": r["messages"]} for r in val_records]

    save_jsonl(PROCESSED_DIR / "train.jsonl", clean_train)
    save_jsonl(PROCESSED_DIR / "val.jsonl", clean_val)
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAll files successfully written to {PROCESSED_DIR}:")
    print("  - train.jsonl")
    print("  - val.jsonl")
    print("  - gate_labels.jsonl")
    print("  - stats.json")


if __name__ == "__main__":
    main()
