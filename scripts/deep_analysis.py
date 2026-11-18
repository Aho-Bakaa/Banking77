import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

EVAL_DIR = Path("C:/Users/anmol/OneDrive/Desktop/Work/banking77/eval_output")

def parse_record(text):
    thought = None
    if "<think>" in text:
        parts = text.split("<think>", 1)[1]
        if "</think>" in parts:
            thought = parts.split("</think>", 1)[0].strip()
    
    # Try finding json
    # Case 1: ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            d["_thought"] = thought
            return d
        except Exception:
            pass

    # Case 2: standard { ... }
    # Look for all {...} blocks
    matches = re.findall(r"\{[^{}]+\}", text)
    if matches:
        for block in reversed(matches):  # take the last complete JSON block
            try:
                d = json.loads(block)
                d["_thought"] = thought
                return d
            except Exception:
                continue

    # Case 3: shorthand like "-> intent" or "| intent"
    arrow_m = re.search(r"->\s*([a-zA-Z0-9_]+)", text)
    if arrow_m:
        return {"status": "resolved", "intent": arrow_m.group(1), "_thought": thought, "_shorthand": True}
    
    pipe_m = re.search(r"<ambiguous:\s*([a-zA-Z0-9_\s|]+)>", text)
    if pipe_m:
        cands = [c.strip() for c in pipe_m.group(1).split("|") if c.strip()]
        return {"status": "ambiguous", "candidates": cands, "_thought": thought, "_shorthand": True}

    return {"status": "parse_error", "raw": text, "_thought": thought}

print("=" * 80)
print("DEEP ANALYSIS OF EVALUATION RESULTS")
print("=" * 80)

# 1. Analyze b77_test across modes
for mode in ["free", "fast", "think"]:
    p = EVAL_DIR / f"b77_test__{mode}.jsonl"
    if not p.exists(): continue
    
    records = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    n = len(records)
    
    top1 = 0
    top3 = 0
    parse_errors = 0
    status_counts = Counter()
    shorthand_count = 0
    
    think_chosen = 0
    think_correct = 0
    fast_chosen = 0
    fast_correct = 0
    
    latencies = [r.get("seconds_per_query", 0) for r in records]
    tokens = [r.get("n_new_tokens", 0) for r in records]
    
    error_pairs = Counter()
    
    for r in records:
        gold = r["gold"]
        out = parse_record(r["output"])
        
        status = out.get("status")
        status_counts[status] += 1
        if out.get("_shorthand"):
            shorthand_count += 1
            
        is_think = ("<think>" in r["output"]) or (out.get("_thought") is not None)
        if is_think:
            think_chosen += 1
        else:
            fast_chosen += 1
            
        pred = None
        if status == "resolved":
            pred = out.get("intent")
            if pred == gold:
                top1 += 1
                top3 += 1
                if is_think: think_correct += 1
                else: fast_correct += 1
            else:
                error_pairs[(gold, pred)] += 1
        elif status == "ambiguous":
            cands = out.get("candidates", [])
            if cands:
                pred = cands[0]
                if pred == gold:
                    top1 += 1
                    if is_think: think_correct += 1
                    else: fast_correct += 1
                if gold in cands[:3]:
                    top3 += 1
                if pred != gold:
                    error_pairs[(gold, pred)] += 1
        else:
            parse_errors += 1
            error_pairs[(gold, "PARSE_ERROR")] += 1
            
    print(f"\n--- b77_test (Mode: {mode.upper()}) ---")
    print(f"Total samples:       {n}")
    print(f"Top-1 Accuracy:      {top1/n*100:.2f}% ({top1}/{n})")
    print(f"Top-3 Accuracy:      {top3/n*100:.2f}% ({top3}/{n})")
    print(f"Status distribution: {dict(status_counts)}")
    print(f"Shorthand outputs:   {shorthand_count} ({shorthand_count/n*100:.1f}%)")
    print(f"Parse errors:        {parse_errors}")
    print(f"Think triggered:     {think_chosen}/{n} ({think_chosen/n*100:.1f}%)")
    if think_chosen > 0:
        print(f"  Think path Acc:    {think_correct/think_chosen*100:.2f}%")
    if fast_chosen > 0:
        print(f"  Fast path Acc:     {fast_correct/fast_chosen*100:.2f}%")
    print(f"Mean Latency:        {np.mean(latencies)*1000:.1f} ms (p95: {np.percentile(latencies, 95)*1000:.1f} ms)")
    print(f"Mean Tokens:         {np.mean(tokens):.1f} (p95: {np.percentile(tokens, 95):.1f})")
    print(f"Top 5 confusion pairs (gold -> pred):")
    for (g, p), cnt in error_pairs.most_common(5):
        print(f"  {g} -> {p}: {cnt} times")

# 2. Analyze amb_test (Ambiguity Set)
print("\n" + "=" * 80)
print("AMBIGUITY SET (amb_test, 200 samples)")
print("=" * 80)

for mode in ["free", "fast", "think"]:
    p = EVAL_DIR / f"amb_test__{mode}.jsonl"
    if not p.exists(): continue
    
    records = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    n = len(records)
    
    caught = 0
    resolved = 0
    errors = 0
    jaccards = []
    
    for r in records:
        gold_cands = set(r.get("candidates_gold", []))
        out = parse_record(r["output"])
        status = out.get("status")
        if status == "ambiguous":
            caught += 1
            pred_cands = set(out.get("candidates", []))
            if gold_cands and pred_cands:
                j = len(gold_cands & pred_cands) / len(gold_cands | pred_cands)
                jaccards.append(j)
            else:
                jaccards.append(0.0)
        elif status == "resolved":
            resolved += 1
        else:
            errors += 1
            
    print(f"\n--- amb_test (Mode: {mode.upper()}) ---")
    print(f"Ambiguity Recall:    {caught/n*100:.2f}% ({caught}/{n})")
    print(f"False Positives (R): {resolved/n*100:.2f}% ({resolved}/{n})")
    print(f"Mean Cand Jaccard:   {np.mean(jaccards)*100:.2f}%" if jaccards else "N/A")

# 3. Analyze b77_train500 (Memorisation Check)
print("\n" + "=" * 80)
print("MEMORISATION CHECK (b77_train500, 500 samples)")
print("=" * 80)

for mode in ["free", "fast", "think"]:
    p = EVAL_DIR / f"b77_train500__{mode}.jsonl"
    if not p.exists(): continue
    
    records = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    n = len(records)
    
    top1 = 0
    top3 = 0
    for r in records:
        gold = r["gold"]
        out = parse_record(r["output"])
        status = out.get("status")
        if status == "resolved":
            if out.get("intent") == gold:
                top1 += 1
                top3 += 1
        elif status == "ambiguous":
            cands = out.get("candidates", [])
            if cands:
                if cands[0] == gold:
                    top1 += 1
                if gold in cands[:3]:
                    top3 += 1
                    
    print(f"--- train500 (Mode: {mode.upper()}) ---")
    print(f"Train500 Top-1:      {top1/n*100:.2f}% ({top1}/{n})")
    print(f"Train500 Top-3:      {top3/n*100:.2f}% ({top3}/{n})")

# 4. Calibration & Confidence Analysis on b77_test free
print("\n" + "=" * 80)
print("CALIBRATION & LOGPROB ANALYSIS (b77_test free)")
print("=" * 80)

p = EVAL_DIR / "b77_test__free.jsonl"
records = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]

confidences = []
corrects = []
avg_logprobs = []

for r in records:
    gold = r["gold"]
    out = parse_record(r["output"])
    status = out.get("status")
    
    is_corr = False
    if status == "resolved" and out.get("intent") == gold:
        is_corr = True
    elif status == "ambiguous":
        cands = out.get("candidates", [])
        if cands and cands[0] == gold:
            is_corr = True
            
    # Extract confidence
    conf = out.get("confidence")
    if conf is not None:
        try:
            conf = float(conf)
            confidences.append(conf)
            corrects.append(1 if is_corr else 0)
        except ValueError:
            pass
            
    # Token logprobs
    t_lps = r.get("token_logprobs", [])
    if t_lps:
        lps = [x[1] for x in t_lps if x[1] > -900]
        if lps:
            avg_logprobs.append(np.mean(lps))

if confidences:
    bins = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
    print("Confidence Bins vs Actual Accuracy:")
    for b_idx in range(len(bins) - 1):
        low, high = bins[b_idx], bins[b_idx + 1]
        mask = [low <= c < high for c in confidences]
        bin_count = sum(mask)
        if bin_count > 0:
            bin_acc = np.mean([corrects[i] for i, m in enumerate(mask) if m])
            bin_conf = np.mean([confidences[i] for i, m in enumerate(mask) if m])
            print(f"  [{low:.2f} - {high:.2f}): N={bin_count:4d} | Mean Conf: {bin_conf*100:.1f}% | Actual Acc: {bin_acc*100:.1f}%")

if avg_logprobs:
    print(f"\nMean Token Logprob:  {np.mean(avg_logprobs):.4f} (Perplexity approx: {np.exp(-np.mean(avg_logprobs)):.2f})")

