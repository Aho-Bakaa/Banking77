# Teacher Protocol (Phase 1 — CoT distillation)

The teacher (a frontier LLM) writes one record per queued train query. It is given the
query, the **gold label** (hint-guided rationalisation, as in STaR / "Distilling
step-by-step"), and the top competitor intents mined by the System-1 probe
(`scripts/00_mine_hardness.py`). Definitions and decisive cues come from
`configs/intent_taxonomy.json`.

## Output record (`data/synthetic/traces/out_XXX.jsonl`)

```json
{"id": 123, "thought": "...", "status": "resolved", "intent": "card_swallowed"}
{"id": 456, "thought": "...", "status": "ambiguous", "candidates": ["failed_transfer", "declined_transfer"], "question": "..."}
```

Optional: `"flag": "label_noise"` when the gold label contradicts the query text.

## Rules for the `thought`

1. **Never leak the hint.** Reason only from what the customer wrote. No "the label is",
   "gold", "annotated as". The trace must be reproducible by a student that never sees the label.
2. **Cue first.** Open with the decisive clue(s) in the customer's own terms (2–12 words).
3. **Contrast real siblings.** Eliminate 1–3 plausible competitors (prefer the probe's
   competitors) with a concrete reason each, formatted `vs <intent>: <reason>.`
4. **Conclude.** End with `-> <intent>` (resolved) or `-> ambiguous: <a> | <b>`.
5. Short: ~25–70 words. Easy queries get 1 contrast; hard ones 2–3.

## When to output `ambiguous` on real train data

Only when the text contains **no** cue that separates the gold intent from a sibling
(e.g. "My transfer didn't work" — failed vs declined). Then:
- `candidates` = gold first, then the indistinguishable sibling(s) (2–3 total).
- `question` = one short, natural, banking-compliant clarification that would split
  the candidates (no menus of intent names, no requests for PIN/password/full card number).

Being conservative matters: Banking77 test queries carry a single label, so over-flagging
ambiguity costs accuracy. Purely-generic wording that annotators consistently map to one
intent (e.g. "my card doesn't work" -> card_not_working) is **resolved**.

## Label-noise audit

If the gold label is clearly wrong for the text (known Banking77 noise), still write the
trace toward the most defensible intent and set `"flag": "label_noise"`; Phase 2 decides
whether to drop or relabel. The `intent` field then holds the teacher's intent, not gold.
