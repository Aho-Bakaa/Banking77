"""Phase 1: teacher CoT distillation harness.

Sub-commands
  prepare   select the teacher queue (hardness-tiered) and cut it into batches
  show N    print batch N compactly for the teacher to read
  status    report which batches have traces
  collect   validate every trace file and merge into data/synthetic/cot_train.jsonl

Selection policy (~4k of 10,003 train queries):
  * every "hard" query: out-of-fold System-1 probe gives the gold intent prob < 0.8
  * plus a stratified sample of easy queries, EASY_PER_INTENT per intent
The remaining train queries are kept untouched (usable later as label-only data or
for self-distillation), and the test split is never shown to the teacher.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SYNTH_DIR = ROOT / "data" / "synthetic"
BATCH_DIR = SYNTH_DIR / "batches"
TRACE_DIR = SYNTH_DIR / "traces"
TAXONOMY = json.loads((ROOT / "configs" / "intent_taxonomy.json").read_text(encoding="utf-8"))["intents"]
VALID_INTENTS = set(TAXONOMY)

HARD_GOLD_PROB = 0.8
EASY_PER_INTENT = 26
BATCH_SIZE = 60
SEED = 77
LEAK_MARKERS = ("gold", "label is", "labelled as", "labeled as", "annotat", "ground truth", "the hint")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare() -> None:
    hardness_rows = load_jsonl(RAW_DIR / "train_hardness.jsonl")
    rng = random.Random(SEED)
    hard_rows = [row for row in hardness_rows if row["gold_prob"] < HARD_GOLD_PROB]
    easy_by_intent: dict[str, list[dict]] = {}
    for row in hardness_rows:
        if row["gold_prob"] >= HARD_GOLD_PROB:
            easy_by_intent.setdefault(row["gold_intent"], []).append(row)
    easy_rows = []
    for intent_name in sorted(easy_by_intent):
        pool = easy_by_intent[intent_name]
        easy_rows.extend(rng.sample(pool, min(EASY_PER_INTENT, len(pool))))

    queue = []
    for tier, rows in (("hard", hard_rows), ("easy", easy_rows)):
        for row in rows:
            queue.append(
                {
                    "id": int(row["id"].split("-")[1]),
                    "tier": tier,
                    "text": row["text"],
                    "gold_intent": row["gold_intent"],
                    "competitors": row["competitors"][:3],
                    "gold_prob": row["gold_prob"],
                }
            )
    # Group by cluster -> intent so the teacher contrasts siblings consistently.
    queue.sort(key=lambda item: (TAXONOMY[item["gold_intent"]]["cluster"], item["gold_intent"], item["id"]))

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SYNTH_DIR / "teacher_queue.jsonl", "w", encoding="utf-8") as handle:
        for item in queue:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    batch_count = 0
    for start in range(0, len(queue), BATCH_SIZE):
        with open(BATCH_DIR / f"in_{batch_count:03d}.jsonl", "w", encoding="utf-8") as handle:
            for item in queue[start : start + BATCH_SIZE]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        batch_count += 1
    tiers = Counter(item["tier"] for item in queue)
    print(f"queue={len(queue)} hard={tiers['hard']} easy={tiers['easy']} batches={batch_count}")


def show(batch_number: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for item in load_jsonl(BATCH_DIR / f"in_{batch_number:03d}.jsonl"):
        competitors = ",".join(item["competitors"])
        print(f"{item['id']}|{item['gold_intent']}|{competitors}|{item['text']}")


def validate_record(record: dict, queue_item: dict) -> list[str]:
    problems = []
    thought = record.get("thought", "")
    if not isinstance(thought, str) or len(thought.split()) < 8:
        problems.append("thought too short")
    if any(marker in thought.lower() for marker in LEAK_MARKERS):
        problems.append("hint leakage in thought")
    status = record.get("status")
    if status == "resolved":
        if record.get("intent") not in VALID_INTENTS:
            problems.append(f"invalid intent {record.get('intent')}")
        elif record["intent"] != queue_item["gold_intent"] and record.get("flag") != "label_noise":
            problems.append("intent != gold without label_noise flag")
        if f"-> {record.get('intent')}" not in thought:
            problems.append("thought does not conclude with '-> intent'")
    elif status == "ambiguous":
        candidates = record.get("candidates") or []
        if not (2 <= len(candidates) <= 3) or any(c not in VALID_INTENTS for c in candidates):
            problems.append("bad candidates")
        elif queue_item["gold_intent"] not in candidates and record.get("flag") != "label_noise":
            problems.append("gold missing from candidates")
        question = record.get("question", "")
        if not question.endswith("?") or len(question.split()) < 4:
            problems.append("bad clarification question")
        if "-> ambiguous" not in thought:
            problems.append("thought does not conclude with '-> ambiguous'")
    else:
        problems.append(f"bad status {status}")
    return problems


def collect() -> None:
    queue = {item["id"]: item for item in load_jsonl(SYNTH_DIR / "teacher_queue.jsonl")}
    merged, seen, problem_count = [], set(), 0
    for trace_path in sorted(TRACE_DIR.glob("out_*.jsonl")):
        for line_number, line in enumerate(open(trace_path, encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"{trace_path.name}:{line_number} JSON error {error}")
                problem_count += 1
                continue
            queue_item = queue.get(record.get("id"))
            if queue_item is None or record["id"] in seen:
                print(f"{trace_path.name}:{line_number} unknown/duplicate id {record.get('id')}")
                problem_count += 1
                continue
            problems = validate_record(record, queue_item)
            if problems:
                print(f"{trace_path.name}:{line_number} id={record['id']} {problems}")
                problem_count += 1
                continue
            seen.add(record["id"])
            merged.append(
                {
                    "id": f"train-{record['id']:05d}",
                    "source": "banking77_train",
                    "tier": queue_item["tier"],
                    "text": queue_item["text"],
                    "gold_intent": queue_item["gold_intent"],
                    "thought": record["thought"],
                    "status": record["status"],
                    "intent": record.get("intent"),
                    "candidates": record.get("candidates"),
                    "question": record.get("question"),
                    "flag": record.get("flag"),
                }
            )
    with open(SYNTH_DIR / "cot_train.jsonl", "w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    statuses = Counter(row["status"] for row in merged)
    flags = Counter(row["flag"] for row in merged if row["flag"])
    print(f"valid={len(merged)}/{len(queue)} problems={problem_count} status={dict(statuses)} flags={dict(flags)}")


def status() -> None:
    batch_numbers = sorted(int(path.stem[3:]) for path in BATCH_DIR.glob("in_*.jsonl"))
    done = {int(path.stem[4:]) for path in TRACE_DIR.glob("out_*.jsonl")}
    missing = [number for number in batch_numbers if number not in done]
    print(f"done {len(done)}/{len(batch_numbers)}; next missing: {missing[:10]}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "prepare":
        prepare()
    elif command == "show":
        show(int(sys.argv[2]))
    elif command == "collect":
        collect()
    else:
        status()
