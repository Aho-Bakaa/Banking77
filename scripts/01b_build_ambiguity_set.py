"""Phase 1b: build the synthetic ambiguity set from teacher-authored specs.

Spec format (data/synthetic/ambiguity_spec_*.txt):

    ## family_id | intent_a,intent_b[,intent_c] | why these candidates cannot be separated
    utterance || per-item rationale || clarification question
    utterance                                    <- rationale / question optional

Every utterance is genuinely underspecified between the family's candidate intents.
Lines without an explicit question reuse one of the family's authored questions
(deterministically by hash) so each utterance is still paired with a natural question.

Guards:
  * candidates must be valid Banking77 intents
  * questions must end with '?'
  * exact/normalised duplicates are dropped, including against Banking77 train+test
    (so the synthetic set cannot leak benchmark utterances)
  * a family-stratified 80/20 split -> ambiguous_train.jsonl / ambiguous_test.jsonl
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "data" / "synthetic"
VALID_INTENTS = set(json.loads((ROOT / "configs" / "intent_taxonomy.json").read_text(encoding="utf-8"))["intents"])
TEST_FRACTION = 0.2
SEED = 77


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def parse_specs() -> list[dict]:
    items, family_questions = [], defaultdict(list)
    families: dict[str, dict] = {}
    for spec_path in sorted(SYNTH_DIR.glob("ambiguity_spec_*.txt")):
        current = None
        for line_number, raw_line in enumerate(spec_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##"):
                family_id, candidates, reason = [part.strip() for part in line[2:].split("|", 2)]
                candidate_list = [c.strip() for c in candidates.split(",")]
                bad = [c for c in candidate_list if c not in VALID_INTENTS]
                if bad:
                    raise ValueError(f"{spec_path.name}:{line_number} unknown intents {bad}")
                families.setdefault(family_id, {"candidates": candidate_list, "reason": reason})
                current = family_id
                continue
            parts = [part.strip() for part in line.split("||")]
            text = parts[0]
            rationale = parts[1] if len(parts) > 1 and parts[1] else "no detail that separates the candidates"
            question = parts[2] if len(parts) > 2 and parts[2] else None
            if question:
                if not question.endswith("?"):
                    raise ValueError(f"{spec_path.name}:{line_number} question must end with '?'")
                family_questions[current].append(question)
            items.append({"family": current, "text": text, "rationale": rationale, "question": question,
                          "source_file": spec_path.name})
    for item in items:
        if item["question"] is None:
            pool = family_questions[item["family"]]
            digest = int(hashlib.md5(item["text"].encode()).hexdigest(), 16)
            item["question"] = pool[digest % len(pool)]
        family = families[item["family"]]
        item["candidates"] = family["candidates"]
        item["thought"] = (
            f"Cue: {item['rationale']}. {family['reason']} "
            f"-> ambiguous: {' | '.join(family['candidates'])}"
        )
    return items


def main() -> None:
    items = parse_specs()
    banking_texts = set()
    for split in ("train", "test"):
        banking_texts |= {normalise(t) for t in pd.read_csv(ROOT / "data" / "raw" / f"{split}.csv")["text"]}

    seen, kept, dropped = set(), [], Counter()
    for item in items:
        key = normalise(item["text"])
        if key in banking_texts:
            dropped["overlaps_banking77"] += 1
            continue
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(item)

    rng = random.Random(SEED)
    by_family = defaultdict(list)
    for item in kept:
        by_family[item["family"]].append(item)
    train_rows, test_rows = [], []
    for family_id in sorted(by_family):
        family_items = by_family[family_id]
        rng.shuffle(family_items)
        test_count = max(1, round(len(family_items) * TEST_FRACTION))
        test_rows.extend(family_items[:test_count])
        train_rows.extend(family_items[test_count:])

    for name, rows in (("ambiguous_train", train_rows), ("ambiguous_test", test_rows)):
        with open(SYNTH_DIR / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for index, item in enumerate(rows):
                record = {
                    "id": f"{name.split('_')[1]}-amb-{index:04d}",
                    "source": "synthetic_ambiguity",
                    "family": item["family"],
                    "text": item["text"],
                    "thought": item["thought"],
                    "status": "ambiguous",
                    "candidates": item["candidates"],
                    "question": item["question"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    family_counts = Counter(item["family"] for item in kept)
    print(f"parsed={len(items)} kept={len(kept)} dropped={dict(dropped)} train={len(train_rows)} test={len(test_rows)}")
    print(f"families={len(family_counts)} min/max per family={min(family_counts.values())}/{max(family_counts.values())}")


if __name__ == "__main__":
    main()
