# Evaluation summary (offline, reproducible)

Harness: `evals/run_evals.py` v0.1 · dataset digest `5RWCKA5VJB5NKFOI` · 49 labeled cases

Scoring system: **legacy-deterministic-baseline (offline)** — deterministic keyword baseline and the shipped verifiers (verbatim quote check, token-coverage grounding). No LLM calls, no cost.

| Category | Cases | Correct | Accuracy |
|---|---|---|---|
| tone | 19 | 15 | 78.9% |
| evidence_support | 10 | 10 | 100.0% |
| unsupported_company_claims | 10 | 10 | 100.0% |
| factuality | 10 | 10 | 100.0% |

Total accuracy: 91.8%.

Interpretation: the baseline scores regression-test the harness wiring and the
shipped deterministic verifiers. They are keyword heuristics and a weak proxy —
they are **not** evidence of LLM output quality.

LLM evaluation: run `python evals/run_evals.py --live` (requires OPENAI_API_KEY). It records the model name, prompt version and per-case latency; token totals and cost are recorded only when the runtime reports real usage, priced at measured unit prices (gpt-4o-mini: $0.15/1M input, $0.60/1M output). Never estimated.
