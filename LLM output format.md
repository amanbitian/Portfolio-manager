# LLM Output Format

Reviewed: 2026-09-13

This note defines the run-output format we want for local or hosted LLM calls used by the portfolio application. It is based on the performance panel shown in the reference screenshot and should be treated as a UI and logging contract, not as model instructions.

## Purpose

Every LLM response should expose two layers:

- A user-facing answer that can be shown in the app.
- A run telemetry block that records how the model performed and what it used.

This is useful for comparing local models, hosted models, research-agent runs, strategy-copilot runs, and future prompt templates.

## Display Format

For each LLM run, show a compact header:

```text
<model_display_name> - <model_id> - <status>
```

Example:

```text
Qwen3.8-27B FP8 - qwen3.8-27b-fp8 - done
```

Below the header, show these metrics:

```text
TTFT        <milliseconds> ms
DECODE      <tokens_per_second> tok/s
E2E         <tokens_per_second> tok/s
TOKENS      <token_count>
TOTAL       <seconds> s
ITL P50/P95 <p50_ms> ms / <p95_ms> ms
PREFILL~    <tokens_per_second> tok/s
THINKING    <thinking_token_count> tokens
```

Reference values from the screenshot:

```text
TTFT        72 ms
DECODE      101 tok/s
E2E         101 tok/s
TOKENS      5850
TOTAL       58.00 s
ITL P50/P95 30 ms / 32 ms
PREFILL~    1046 tok/s
THINKING    2060 tokens
```

## Field Meanings

| Field | Meaning |
| --- | --- |
| `model_display_name` | Human-friendly model name shown in the UI. |
| `model_id` | Exact model/runtime identifier used by the backend. |
| `status` | Run status such as `queued`, `running`, `done`, `failed`, or `cancelled`. |
| `ttft_ms` | Time to first token in milliseconds. |
| `decode_tok_s` | Output token generation speed after generation begins. |
| `e2e_tok_s` | End-to-end tokens per second across the whole request. |
| `output_tokens` | Number of generated answer tokens. |
| `total_tokens` | Total tokens processed if the backend provides input plus output totals. |
| `total_seconds` | Wall-clock duration of the full request. |
| `itl_p50_ms` | Median inter-token latency. |
| `itl_p95_ms` | 95th percentile inter-token latency. |
| `prefill_tok_s` | Approximate prompt/context processing throughput. |
| `thinking_tokens` | Reasoning or hidden thinking token count when available. |
| `prompt_tokens` | Input/context token count when available. |
| `cache_hit` | Whether the response or feature context was served from cache. |
| `quality_score` | Optional normalized quality score for a specific task or benchmark. |
| `benchmark_name` | Optional benchmark or internal evaluation name, such as strategy explanation, code generation, or math reasoning. |
| `benchmark_score` | Raw score for the named benchmark when available. |
| `baseline_model_id` | Model used as the comparison baseline. |
| `delta_vs_baseline` | Difference from the baseline model on the same task, prompt, and evaluation settings. |
| `error` | Structured error message when status is `failed`. |

## JSON Log Format

Each LLM run should be logged as one JSON object.

```json
{
  "run_id": "llm-run-20260913-000001",
  "created_at": "2026-09-13T00:00:00+05:30",
  "workspace": "Research Agent",
  "feature": "strategy_copilot",
  "model_display_name": "Qwen3.8-27B FP8",
  "model_id": "qwen3.8-27b-fp8",
  "provider": "local",
  "status": "done",
  "ttft_ms": 72,
  "decode_tok_s": 101,
  "e2e_tok_s": 101,
  "prompt_tokens": null,
  "output_tokens": 5850,
  "total_tokens": null,
  "thinking_tokens": 2060,
  "total_seconds": 58.0,
  "itl_p50_ms": 30,
  "itl_p95_ms": 32,
  "prefill_tok_s": 1046,
  "cache_hit": false,
  "prompt_template_id": "strategy_copilot_v1",
  "input_fingerprint": "sha256:<hash>",
  "output_fingerprint": "sha256:<hash>",
  "quality_score": null,
  "benchmark_name": null,
  "benchmark_score": null,
  "baseline_model_id": null,
  "delta_vs_baseline": null,
  "error": null
}
```

## App Output Contract

LLM-backed app features should return this structure internally:

```json
{
  "answer": {
    "summary": "Short answer shown first.",
    "details": [],
    "tables": [],
    "warnings": [],
    "next_actions": []
  },
  "telemetry": {
    "model_display_name": "Qwen3.8-27B FP8",
    "model_id": "qwen3.8-27b-fp8",
    "status": "done",
    "ttft_ms": 72,
    "decode_tok_s": 101,
    "e2e_tok_s": 101,
    "output_tokens": 5850,
    "thinking_tokens": 2060,
    "total_seconds": 58.0,
    "itl_p50_ms": 30,
    "itl_p95_ms": 32,
    "prefill_tok_s": 1046
  }
}
```

## Model Metadata

The dashboard should also store static or slowly changing model metadata separately
from per-run telemetry.

```json
{
  "model_id": "noema-2b-code",
  "model_display_name": "Noema 2B AI Code Model",
  "provider": "local",
  "base_model": "qwen3.5-2b",
  "parameter_count": "2B",
  "quantization": "GGUF",
  "precision": "Q4_K_M",
  "model_family": "code",
  "reasoning_mode": "non_thinking",
  "backbone": "GDN hybrid",
  "offline_capable": true,
  "default_context_tokens": 32768,
  "tags": [
    "offline",
    "code",
    "low_latency"
  ],
  "notes": "Post-trained from Qwen3.5-2B base."
}
```

## UI Placement

Recommended placement in the app:

- Show the answer normally in the selected app tab.
- Put run metrics in a small expandable section named `LLM run details`.
- Keep the model name, status, total time, and token count visible.
- Keep detailed metrics such as TTFT, ITL, prefill, and thinking tokens inside the expanded view.
- For saved proposals or research notes, persist both the answer and telemetry.

## Model Performance Dashboard

Add a separate app dashboard for comparing model performance across local and hosted
models. The dashboard should use the telemetry records above as its primary source.

Recommended dashboard cards:

- `Token efficiency`: generated tokens per answer, prompt tokens, context size, and percent fewer tokens versus baseline.
- `Latency`: TTFT, total seconds, decode tokens/second, prefill tokens/second, and inter-token latency P50/P95.
- `Quality`: internal score, benchmark score, pass/fail checks, malformed-output rate, and human review status.
- `Reliability`: success rate, timeout rate, cancellation rate, retry count, and structured-output validity.
- `Cost/resource`: local memory use, CPU/GPU backend, estimated hosted cost, and tokens per rupee/dollar when available.

Recommended dashboard views:

- `Overview`: ranked model table with status, provider, total time, quality score, and reliability.
- `Model profile`: large model title, tags such as offline/local/code/reasoner, metadata badges, and latest benchmark cards.
- `Run history`: searchable table of individual runs with prompt template, workspace, input fingerprint, and telemetry.
- `Comparison`: side-by-side model comparison against a selected baseline on identical prompts and settings.
- `Benchmarks`: task-specific results for research summaries, strategy prompts, structured JSON validity, code assistance, and math checks.

Model profile cards can use the visual style from the reference image:

```text
Noema 2B AI Code Model
The Silent Offline Reasoner - Post-Trained from Qwen3.5-2B Base

[offline gguf] [non-thinking reasoner] [code model] [local]

Token efficiency       95.7%
Structured JSON pass   98.0%
Strategy prompt score  91.7%
Median TTFT            72 ms
Decode speed           101 tok/s
```

The dashboard should avoid marketing-only numbers. Each score must show its source,
sample size, date range, prompt template version, and baseline model when applicable.

## Optimization Notes

These metrics should feed the optimisation backlog:

- Compare local and hosted models using `total_seconds`, `e2e_tok_s`, and `quality`.
- Track `ttft_ms` separately because it affects perceived app responsiveness.
- Track `prefill_tok_s` because large portfolio context and research documents can make prompt processing expensive.
- Track `thinking_tokens` because high reasoning budgets can improve quality but increase latency and cost.
- Store `input_fingerprint` and `prompt_template_id` so repeated runs can be cached or compared.
- Keep failed and cancelled runs in logs so reliability can be measured honestly.
- Use the model dashboard to choose defaults by measured task performance, not by model size alone.

## Open Decisions

- Decide whether token counts mean generated tokens only or total input plus output tokens for each provider.
- Decide whether hidden reasoning token counts are available for every model provider.
- Decide retention policy for prompt and response logs before adding real user portfolio data.
- Decide whether LLM telemetry should live in JSON proposal snapshots first or move to Parquet with the experiment artifacts.
- Decide which internal quality benchmarks are required before a model can become the default for Research Agent or Strategy Copilot.
