"""Phase 1 summary: statistics over the distilled CoT data, written to data/synthetic/phase1_stats.json."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "data" / "synthetic"
TAXONOMY = json.loads((ROOT / "configs" / "intent_taxonomy.json").read_text(encoding="utf-8"))["intents"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    cot_rows = load_jsonl(SYNTH_DIR / "cot_train.jsonl")
    amb_train = load_jsonl(SYNTH_DIR / "ambiguous_train.jsonl")
    amb_test = load_jsonl(SYNTH_DIR / "ambiguous_test.jsonl")

    thought_words = [len(row["thought"].split()) for row in cot_rows + amb_train + amb_test]
    status_by_tier = Counter((row["tier"], row["status"]) for row in cot_rows)
    cluster_total, cluster_ambiguous = Counter(), Counter()
    for row in cot_rows:
        cluster = TAXONOMY[row["gold_intent"]]["cluster"]
        cluster_total[cluster] += 1
        cluster_ambiguous[cluster] += row["status"] == "ambiguous"
    ambiguous_pairs = Counter(
        tuple(sorted(row["candidates"][:2])) for row in cot_rows if row["status"] == "ambiguous"
    )
    noise_rows = [row for row in cot_rows if row["flag"] == "label_noise"]
    relabels = Counter(
        (row["gold_intent"], row["intent"]) for row in noise_rows if row["status"] == "resolved"
    )

    stats = {
        "banking77_cot": {
            "total": len(cot_rows),
            "status": dict(Counter(row["status"] for row in cot_rows)),
            "status_by_tier": {f"{tier}/{status}": count for (tier, status), count in status_by_tier.items()},
            "label_noise_flags": len(noise_rows),
            "label_noise_resolved_relabels_top": [
                {"gold": gold, "teacher": teacher, "count": count} for (gold, teacher), count in relabels.most_common(15)
            ],
            "ambiguity_rate_by_cluster": {
                cluster: round(cluster_ambiguous[cluster] / cluster_total[cluster], 3)
                for cluster in sorted(cluster_total)
            },
            "top_ambiguous_pairs": [
                {"pair": list(pair), "count": count} for pair, count in ambiguous_pairs.most_common(15)
            ],
            "intents_covered": len({row["gold_intent"] for row in cot_rows}),
        },
        "synthetic_ambiguity": {
            "train": len(amb_train),
            "test": len(amb_test),
            "families": len({row["family"] for row in amb_train + amb_test}),
        },
        "thought_words": {"mean": round(mean(thought_words), 1), "median": median(thought_words),
                          "max": max(thought_words)},
    }
    (SYNTH_DIR / "phase1_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
