import json
import re
from pathlib import Path
import numpy as np

ROOT = Path("C:/Users/anmol/OneDrive/Desktop/Work/banking77")
b77_p = ROOT / "eval_output/eval_base/eval_base/b77_test__free.jsonl"
amb_p = ROOT / "eval_output/eval_base/eval_base/amb_test__free.jsonl"
tax_p = ROOT / "configs/intent_taxonomy.json"

valid_intents = set(json.load(open(tax_p, encoding="utf-8"))["intents"])

# 1. b77_test
exact_matches = 0
hallucinations = 0
json_valid = 0
latencies = []
tokens = []
total_test = 0

with open(b77_p, encoding="utf-8") as f:
    for line in f:
        total_test += 1
        d = json.loads(line)
        gold = d["gold"]
        out = d["output"].strip()
        latencies.append(d.get("seconds_per_query", 0))
        tokens.append(d.get("n_new_tokens", 0))

        if out.startswith("{") and out.endswith("}"):
            json_valid += 1

        pred = out.strip("`\"' ")
        # if output contains newline or spaces, get clean token
        clean_pred = re.sub(r"[^a-zA-Z0-9_]", "", pred.split()[0] if pred.split() else "")
        
        if clean_pred == gold or pred == gold:
            exact_matches += 1
        elif clean_pred not in valid_intents:
            hallucinations += 1

# 2. amb_test
total_amb = 0
amb_detected = 0
with open(amb_p, encoding="utf-8") as f:
    for line in f:
        total_amb += 1
        d = json.loads(line)
        out = d["output"].strip().lower()
        if "ambiguous" in out:
            amb_detected += 1

print("=" * 60)
print("BASE MODEL ZERO-SHOT EVALUATION RESULTS")
print("=" * 60)
print(f"b77_test Total Queries:       {total_test}")
print(f"Top-1 Accuracy (Exact Match): {exact_matches / total_test * 100:.2f}% ({exact_matches}/{total_test})")
print(f"JSON Schema Adherence:        {json_valid / total_test * 100:.2f}% ({json_valid}/{total_test})")
print(f"Hallucinated / Invalid Names: {hallucinations / total_test * 100:.2f}% ({hallucinations}/{total_test})")
print(f"Ambiguity Detection Recall:   {amb_detected / total_amb * 100:.2f}% ({amb_detected}/{total_amb})")
print(f"Mean Latency per Query:       {np.mean(latencies) * 1000:.1f} ms")
print(f"Mean Tokens Generated:        {np.mean(tokens):.1f}")
