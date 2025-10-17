---
title: Context Comparator Pro
version: 1.0.0
license: CC0-1.0
---

# Context Comparator Pro

## Overview & Purpose
Context Comparator Pro is a local-first evaluation harness for comparing Ollama-hosted models under identical prompts, decoding parameters, and hardware conditions. It pairs a Streamlit dashboard with a batch CLI so you can capture telemetry, judge quality, and export reproducible artifacts from the same machine that will host production workloads.

## Why Model Comparison Matters
Choosing an LLM for deployment is rarely about raw accuracy alone. Latency, throughput, memory pressure, GPU saturation, and qualitative fitness under real task profiles all shape ROI. By benchmarking on your own hardware and prompts, you remove guesswork and surface trade-offs before committing to a model stack.

## Setup & Dependencies
1. Install Python 3.10+ and Ollama ≥ 0.3.3 with at least one local model pulled.
2. (Optional) Install GPU telemetry extras for richer stats: `pip install GPUtil pynvml`.
3. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or use the provided Makefile:
   ```bash
   make venv
   make activate   # follow the printed instructions in your shell
   make install
   ```
   If your interpreter lives elsewhere, pass it explicitly: `make venv PYTHON=/usr/local/bin/python3`.
4. Copy `.env.example` to `.env` if you want to override default paths or logging verbosity.

## Parameters Explained
- **Decoding**: `temperature`, `top_p`, `top_k`, `repeat_penalty`, `min_p`, `repeat_last_n`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `seed`, `stop`.
- **Context & Prompting**: `num_ctx`, `few_shot_count` (via presets), `format=json`, `grammar` toggles, `input_tokens` (derived from prompt length).
- **Compute**: `num_thread`, `num_gpu`, `num_batch`, `quantization` (from model card), warm-up runs.
- **Telemetry**: latency, time-to-first-token, tokens/sec, eval/prompt durations, CPU %, RAM %, VRAM %, GPU power, thermal flags.
- **Quality**: automatic rubric scores (faithfulness, completeness, structure, conciseness) with configurable weights plus optional repeat similarity.

## Usage
### Streamlit Dashboard
```bash
streamlit run app.py
```
- Or simply run `make run-app`.
- Select 2–3 models from the sidebar, choose a preset (Eval Stable, Production, Creative), and tweak advanced decoding/compute knobs as needed.
- Choose one or multiple task profiles, set repeat counts, optionally run a warm-up, and click **Run Benchmark** to capture metrics and export artifacts.
- Explore the main tabs to guide your workflow:
  - **Run Overview**: configure experiments and view summary insights.
  - **Telemetry Snapshot**: monitor CPU/GPU utilisation for the host system.
  - **Resources & Guidance**: quick links to Ollama, Hugging Face model hubs, and accessibility best practices.
- Paste your own prompts in the **Custom Prompt** section (use `---` between prompts). They are logged with their own labels and included in the scoring/exports for direct comparisons.

### Batch CLI
```bash
python bench.py --models llama3 phi3:mini --profile summarization --csv prompts/samples.csv --repeats 3 --preset eval_stable
```
- Or call the Makefile wrapper:
  ```bash
  make bench MODELS="llama3 phi3:mini" PROFILE=summarization REPEATS=3 PRESET=eval_stable
  ```
- Outputs CSV/JSON logs under `results/`, Markdown summaries in `results/reports/`, and model generations in `results/outputs/`.
- Every run writes a telemetry bundle to `results/logs/run_YYYYMMDD_HHMMSS.json`.

### Git Workflow Helpers
- After staging your changes, you can commit and push to the `main` branch with:
  ```bash
  make push MSG="short summary of changes"
  ```

## Sample Results Screenshot
Add a screenshot of the Streamlit dashboard (e.g., `docs/streamlit-dashboard.png`) after your first run to share with reviewers and teammates.

## How to Interpret Metrics
- **Performance**: compare `latency_s`, `ttft_s`, and `tokens_per_sec` for throughput versus responsiveness.
- **Cost Proxy**: `cost_proxy = latency / tokens_per_sec` approximates relative efficiency on your hardware.
- **System Load**: watch CPU/GPU utilisation, RAM/VRAM headroom, and power draw to avoid thermal throttling.
- **Quality**: higher rubric aggregates indicate stronger adherence to task-specific expectations; investigate discrepancies via saved outputs.
- **Stability**: look at mean ± std dev plus repeat similarity to gauge determinism and variance.
- **Decision Support**: the Summary tab highlights top-performing models by quality, latency, and throughput per task to speed up selection conversations.
- **Decision Matrix**: export the summary CSV and assign custom weights to quality/latency/throughput to form a scorecard tailored to your objectives (e.g., 50% quality, 30% latency, 20% cost proxy).

## Extending Task Profiles
- Duplicate any file in `tasks/` and adjust the front-matter (`id`, `goal`, `metrics`, `expected_structure`) along with the canonical prompt.
- Reference the new profile via the CLI `--profile` flag or select it in the dashboard multiselect.
- Add matching rows to `prompts/samples.csv` to include batch prompts for that profile.

## Limitations & Future Work
- Ollama metrics currently expose limited GPU utilisation; consider integrating vendor-specific tooling for finer granularity.
- Quality rubric relies on lightweight heuristics—replace or extend with human ratings or automated reference comparisons as needed.
- Power telemetry is only captured when `pynvml` is available and supported by your GPU drivers.
- Charts and summary metrics focus on means; inspect the detailed metrics tab for distribution nuances before final decisions.

## License & Contributors
- Licensed under [CC0 1.0 Universal](LICENSE).
- Maintained by the Context Comparator Pro community—open issues or submit pull requests with improvements, task profiles, or telemetry integrations.
