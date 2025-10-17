"""Streamlit dashboard for Context Comparator Pro."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from bench import (
    DEFAULT_PROMPTS,
    PRESETS,
    RESULTS_DIR,
    compute_similarity,
    list_local_models,
    load_prompts,
    load_task_profile,
    merge_options,
    run_benchmark,
    summarise_metrics,
    TaskProfile,
    write_outputs,
)
from telemetry import collect_system_snapshot


st.set_page_config(page_title="Context Comparator Pro", layout="wide")
st.title("Context Comparator Pro")


def sanitise_identifier(label: str, fallback: str = "custom_prompt") -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", label.strip().lower())
    token = token.strip("_")
    return token or fallback


def parse_custom_prompts(raw_text: str) -> List[str]:
    if not raw_text.strip():
        return []
    prompts = [chunk.strip() for chunk in raw_text.split("\n---\n") if chunk.strip()]
    if not prompts:
        return [raw_text.strip()]
    return prompts


def preset_label_map() -> Dict[str, str]:
    return {
        "Eval Stable": "eval_stable",
        "Production": "production",
        "Creative": "creative",
        "Custom": "",
    }


PRESET_DESCRIPTIONS = {
    "eval_stable": "Balanced defaults for reproducible evaluation with deterministic sampling.",
    "production": "Conservative decoding with repeat penalty and grammar emphasis.",
    "creative": "Higher temperature and relaxed weighting for ideation tasks.",
}


def render_hardware_snapshot() -> None:
    snapshot = collect_system_snapshot()
    st.subheader("Hardware Snapshot")
    cpu = snapshot["cpu"]
    ram = snapshot["ram"]
    cpu_table = pd.DataFrame(
        [
            {
                "CPU Model": cpu["model"],
                "Physical Cores": cpu["physical_cores"],
                "Logical Cores": cpu["logical_cores"],
                "Utilisation %": cpu["utilization_percent"],
                "Load Avg": cpu["load_average"],
            }
        ]
    )
    ram_table = pd.DataFrame(
        [
            {
                "Total (GB)": ram["total_gb"],
                "Available (GB)": ram["available_gb"],
                "Used %": ram["used_percent"],
            }
        ]
    )
    cols = st.columns(2)
    cols[0].dataframe(cpu_table, use_container_width=True)
    cols[1].dataframe(ram_table, use_container_width=True)
    gpu_info = snapshot.get("gpu", [])
    if gpu_info:
        st.dataframe(pd.DataFrame(gpu_info), use_container_width=True)


def advanced_controls(defaults: Optional[Dict[str, float]]) -> Dict[str, float]:
    defaults = defaults or {}

    def d(key: str, fallback: float) -> float:
        value = defaults.get(key, fallback)
        return fallback if value is None else value

    with st.sidebar.expander("Advanced Controls", expanded=False):
        st.caption("Fine-tune generation sampling, compute, and mitigation knobs.")

        st.markdown("**Sampling**")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(d("temperature", 0.3)),
            step=0.05,
            help="Higher values increase randomness; lower values stabilise outputs.",
        )
        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=float(d("top_p", 0.9)),
            step=0.01,
            help="Select tokens from the smallest set whose probabilities sum to top_p.",
        )
        top_k = st.number_input(
            "Top-k",
            min_value=1,
            max_value=5000,
            value=int(d("top_k", 40)),
            step=10,
            help="Restrict sampling to the top-k tokens by probability.",
        )
        repeat_penalty = st.slider(
            "Repeat penalty",
            min_value=0.8,
            max_value=2.0,
            value=float(d("repeat_penalty", 1.1)),
            step=0.05,
            help="Penalise repeated tokens to reduce loops.",
        )
        min_p = st.slider(
            "Min-p",
            min_value=0.0,
            max_value=1.0,
            value=float(d("min_p", 0.0)),
            step=0.01,
            help="Lower bound for nucleus sampling; keep at 0 for default behaviour.",
        )
        repeat_last_n = st.number_input(
            "Repeat last n",
            min_value=0,
            max_value=4096,
            value=int(d("repeat_last_n", 64)),
            step=16,
            help="Apply repeat penalty over the last n tokens.",
        )
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(d("seed", 42)),
            step=1,
            help="Use a consistent seed to reproduce outputs when supported by the model.",
        )

        st.markdown("**Mirostat Adaptive Sampling**")
        mirostat_modes = [0, 1, 2]
        try:
            default_mirostat_index = mirostat_modes.index(int(d("mirostat", 0)))
        except ValueError:
            default_mirostat_index = 0
        mirostat = st.selectbox(
            "Mode",
            options=mirostat_modes,
            index=default_mirostat_index,
            help="0 disables Mirostat, 1 enables the original algorithm, 2 uses the updated variant.",
        )
        col_tau, col_eta = st.columns(2)
        with col_tau:
            mirostat_tau = st.slider(
                "Tau",
                min_value=0.5,
                max_value=10.0,
                value=float(d("mirostat_tau", 5.0)),
                step=0.1,
            )
        with col_eta:
            mirostat_eta = st.slider(
                "Eta",
                min_value=0.01,
                max_value=1.0,
                value=float(d("mirostat_eta", 0.1)),
                step=0.01,
            )

        st.markdown("**Compute & Context**")
        num_ctx = st.number_input(
            "Context window (tokens)",
            min_value=512,
            max_value=131072,
            value=int(d("num_ctx", 4096)),
            step=512,
            help="Maximum prompt+response length. Respect each model's documented limit.",
        )
        num_thread = st.number_input(
            "Threads",
            min_value=0,
            max_value=256,
            value=int(d("num_thread", 0)),
            step=1,
            help="0 lets Ollama infer physical cores; override to experiment with CPU parallelism.",
        )
        num_gpu = st.number_input(
            "GPU layers",
            min_value=-1,
            max_value=128,
            value=int(d("num_gpu", -1)),
            step=1,
            help="-1 enables auto-detection; higher values offload more layers to GPU (if supported).",
        )
        num_batch = st.number_input(
            "Batch size",
            min_value=1,
            max_value=4096,
            value=int(d("num_batch", 32)),
            step=8,
            help="Number of tokens to process per batch. Higher values trade memory for throughput.",
        )

        stop_sequences = st.text_area(
            "Stop sequences",
            value=", ".join(defaults.get("stop", [])) if isinstance(defaults.get("stop"), list) else defaults.get("stop", ""),
            help="Comma-separated strings that will terminate generation early.",
        )

    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repeat_penalty": float(repeat_penalty),
        "num_ctx": int(num_ctx),
        "num_thread": int(num_thread),
        "num_gpu": int(num_gpu),
        "num_batch": int(num_batch),
        "seed": int(seed),
        "min_p": float(min_p),
        "repeat_last_n": int(repeat_last_n),
        "mirostat": int(mirostat),
        "mirostat_tau": float(mirostat_tau),
        "mirostat_eta": float(mirostat_eta),
    }
    stop_sequences_list = [seq.strip() for seq in stop_sequences.split(",") if seq.strip()]
    if stop_sequences_list:
        options["stop"] = stop_sequences_list
    return options


def render_summary_insights(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        st.info("Run a benchmark to populate performance insights.")
        return

    st.markdown("#### Quick Insights")
    col1, col2, col3 = st.columns(3)
    if "aggregate_mean" in summary_df.columns and not summary_df["aggregate_mean"].isnull().all():
        best_quality = summary_df.loc[summary_df["aggregate_mean"].idxmax()]
        col1.metric(
            label=f"Best Quality · {best_quality['task_id']}",
            value=f"{best_quality['model']}",
            delta=f"{best_quality['aggregate_mean']} avg score",
        )
    if "latency_s_mean" in summary_df.columns and not summary_df["latency_s_mean"].isnull().all():
        fastest = summary_df.loc[summary_df["latency_s_mean"].idxmin()]
        col2.metric(
            label=f"Fastest Latency · {fastest['task_id']}",
            value=f"{fastest['model']}",
            delta=f"{fastest['latency_s_mean']} s mean",
        )
    if "tokens_per_sec_mean" in summary_df.columns and not summary_df["tokens_per_sec_mean"].isnull().all():
        throughput = summary_df.loc[summary_df["tokens_per_sec_mean"].idxmax()]
        col3.metric(
            label=f"Highest Throughput · {throughput['task_id']}",
            value=f"{throughput['model']}",
            delta=f"{throughput['tokens_per_sec_mean']} tok/s",
        )

    st.markdown("#### Task-Level Recommendations")
    for task_id, group in summary_df.groupby("task_id"):
        st.markdown(f"**{task_id}**")
        lines: List[str] = []
        if "aggregate_mean" in group.columns and not group["aggregate_mean"].isnull().all():
            best = group.loc[group["aggregate_mean"].idxmax()]
            lines.append(f"- Best overall quality: `{best['model']}` (avg score {best['aggregate_mean']})")
        if "latency_s_mean" in group.columns and not group["latency_s_mean"].isnull().all():
            fastest = group.loc[group["latency_s_mean"].idxmin()]
            lines.append(f"- Lowest latency: `{fastest['model']}` ({fastest['latency_s_mean']} s mean)")
        if "tokens_per_sec_mean" in group.columns and not group["tokens_per_sec_mean"].isnull().all():
            highest_tokens = group.loc[group["tokens_per_sec_mean"].idxmax()]
            lines.append(f"- Highest throughput: `{highest_tokens['model']}` ({highest_tokens['tokens_per_sec_mean']} tok/s)")
        if lines:
            st.markdown("\n".join(lines))
        else:
            st.write("No metrics recorded yet.")

    if {"aggregate_mean", "latency_s_mean", "tokens_per_sec_mean"}.issubset(summary_df.columns):
        st.caption("Tip: Balance quality vs. latency by picking the model that ranks consistently across metrics.")


def download_button(label: str, data_frame: pd.DataFrame, file_name: str) -> None:
    if data_frame.empty:
        return
    buffer = io.StringIO()
    data_frame.to_csv(buffer, index=False)
    st.download_button(label, buffer.getvalue(), file_name=file_name, mime="text/csv")


models_available = list_local_models()
preset_map = preset_label_map()
sidebar_models = st.sidebar.multiselect("Models", options=models_available, default=models_available[:2])
preset_label = st.sidebar.selectbox("Preset", options=list(preset_map.keys()), index=0)
selected_preset = preset_map[preset_label]
defaults = PRESETS.get(selected_preset, {}).get("decoding", {}) | PRESETS.get(selected_preset, {}).get("compute", {})
if selected_preset:
    preset_description = PRESET_DESCRIPTIONS.get(selected_preset, "")
    if preset_description:
        st.sidebar.caption(preset_description)
advanced_options = advanced_controls(defaults if selected_preset else None)

tasks_available = sorted([path.stem for path in Path("tasks").glob("*.md")])
selected_profiles = st.sidebar.multiselect("Task Profiles", options=tasks_available, default=["summarization"])
repeats = st.sidebar.slider("Repeats", min_value=1, max_value=5, value=1)
warmup = st.sidebar.checkbox("Warm-up run", value=False)
st.sidebar.caption("Warm-up sends a short prompt first to stabilise latency measurements.")

with st.sidebar.expander("Prompt Packs & CSVs", expanded=False):
    st.caption(
        "Batch prompts are read from `prompts/samples.csv`. Edit that file or point the CLI to a custom CSV to compare "
        "bespoke workloads."
    )

overview_tab, telemetry_tab, resources_tab = st.tabs(
    ["Run Overview", "Telemetry Snapshot", "Resources & Guidance"]
)

with telemetry_tab:
    render_hardware_snapshot()
    st.caption(
        "Hardware metrics are captured automatically in each run log. Keep this tab visible while benchmarking to monitor utilisation."
    )

with resources_tab:
    st.markdown("### Helpful Resources")
    st.markdown(
        "- [Ollama Model Library](https://ollama.com/library) — discover locally runnable models and quantisations.\n"
        "- [Hugging Face Text Generation Models](https://huggingface.co/models?pipeline_tag=text-generation) — explore broader model options for conversion.\n"
        "- [Hugging Face LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) — compare community benchmarks for additional context.\n"
        "- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) — reference parameters when extending the runner.\n"
        "- [Streamlit Accessibility Guide](https://docs.streamlit.io/knowledge-base/using-streamlit/accessible-apps) — keep dashboards inclusive when iterating on UI.",
    )
    st.info("Tip: Save relevant model cards or leaderboard snapshots alongside your benchmark runs for auditability.")

with overview_tab:
    st.subheader("Benchmark Setup")
    st.markdown(
        "Select models and task profiles, optionally add ad-hoc prompts, review the configuration summary, "
        "then run the benchmark. Results and actionable insights will appear below."
    )

    st.markdown("#### Custom Prompt Benchmarking (Optional)")
    custom_prompt_label = st.text_input("Scenario label", value="Custom Scenario")
    custom_prompt_goal = st.text_input("Goal / Use-case (optional)", value="Ad-hoc evaluation prompt")
    custom_prompt_text = st.text_area(
        "Custom prompt(s)",
        height=160,
        placeholder="Paste one or more prompts. Separate multiple prompts with a line containing only ---",
        help="If you include multiple prompts, add a line with --- between them.",
    )
    custom_prompt_structure = st.selectbox(
        "Expected structure",
        options=[None, "bullet_list", "numbered_list", "json", "paragraph_then_bullets", "headings_and_code"],
        index=0,
        format_func=lambda x: "Freeform" if x is None else x.replace("_", " ").title(),
    )
    custom_prompt_metrics = st.multiselect(
        "Quality metrics to emphasise",
        options=["faithfulness", "completeness", "structure", "conciseness"],
        default=["faithfulness", "completeness"],
    )
    custom_prompt_keywords_input = st.text_input(
        "Key terms to check (optional)",
        help="Comma-separated keywords that should appear in the response.",
    )
    custom_prompt_keywords = [kw.strip() for kw in custom_prompt_keywords_input.split(",") if kw.strip()]
    custom_prompt_targets = int(
        st.number_input(
            "Target word count (for conciseness heuristic)",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
        )
    )
    custom_prompts_list = parse_custom_prompts(custom_prompt_text)
    st.caption(
        "Custom prompts will be evaluated in addition to any selected task profiles. "
        "Each prompt is scored independently and exported with the label you provide."
    )

    with st.expander("Configuration Summary", expanded=False):
        summary_payload: Dict[str, object] = {
            "models": sidebar_models,
            "preset": preset_label,
            "task_profiles": selected_profiles,
            "repeats": repeats,
            "warmup": warmup,
            "overrides": advanced_options,
        }
        if custom_prompts_list:
            summary_payload["custom_prompts"] = [
                {"id": f"{sanitise_identifier(custom_prompt_label)}_{idx+1}", "prompt": prompt}
                for idx, prompt in enumerate(custom_prompts_list)
            ]
        if custom_prompt_keywords:
            summary_payload["custom_keywords"] = custom_prompt_keywords
        summary_payload["custom_target_words"] = custom_prompt_targets
        st.json(summary_payload)

    results_placeholder = st.container()
    run_clicked = st.button("Run Benchmark", type="primary", use_container_width=True)

    if run_clicked:
        if not sidebar_models:
            st.error("Select at least one model to proceed.")
        else:
            run_queue: List[Dict[str, object]] = []
            for profile in selected_profiles:
                task_profile = load_task_profile(profile)
                prompt_rows = load_prompts(DEFAULT_PROMPTS, profile)
                run_queue.append(
                    {
                        "task": task_profile,
                        "prompt_rows": prompt_rows,
                        "label": task_profile.identifier,
                    }
                )

            if custom_prompts_list:
                custom_identifier = sanitise_identifier(custom_prompt_label)
                custom_task = TaskProfile(
                    identifier=custom_identifier,
                    difficulty="custom",
                    goal=custom_prompt_goal or "Custom prompt benchmark",
                    metrics=custom_prompt_metrics,
                    expected_structure=custom_prompt_structure,
                    prompt=custom_prompts_list[0] if custom_prompts_list else "",
                    keywords=custom_prompt_keywords or None,
                    target_words=custom_prompt_targets,
                )
                custom_rows = []
                for idx, prompt_text in enumerate(custom_prompts_list):
                    label = f"{custom_identifier}_{idx+1}"
                    custom_rows.append(
                        {
                            "task_id": label,
                            "prompt": prompt_text,
                            "expected_output": "",
                        }
                    )
                custom_df = pd.DataFrame(custom_rows)
                run_queue.append(
                    {
                        "task": custom_task,
                        "prompt_rows": custom_df,
                        "label": custom_task.identifier,
                    }
                )

            with st.spinner("Running benchmarks across selected models..."):
                combined_records: List[pd.DataFrame] = []
                combined_summaries: List[pd.DataFrame] = []
                all_outputs: Dict[str, List[str]] = {}
                for item in run_queue:
                    task = item["task"]
                    prompt_rows = item["prompt_rows"]
                    profile_label = item["label"]
                    merged_options = merge_options(
                        decoding_overrides=advanced_options,
                        compute_overrides=None,
                        preset=selected_preset or None,
                    )
                    df, outputs = run_benchmark(
                        models=sidebar_models,
                        task=task,
                        repeats=repeats,
                        prompt_rows=prompt_rows,
                        options=merged_options,
                        preset=selected_preset or None,
                        warmup=warmup,
                    )
                    summary_df = summarise_metrics(df)
                    similarity_scores = compute_similarity(outputs)
                    if similarity_scores:
                        summary_df["similarity"] = summary_df.apply(
                            lambda row: similarity_scores.get(f"{row['model']}_{row['task_id']}"), axis=1
                        )
                    write_outputs(
                        df=df,
                        summary_df=summary_df,
                        outputs=outputs,
                        task=task,
                        models=sidebar_models,
                        preset=selected_preset or None,
                        repeats=repeats,
                    )
                    combined_records.append(df.assign(profile=profile_label))
                    combined_summaries.append(summary_df.assign(profile=profile_label))
                    all_outputs.update(outputs)

            with results_placeholder:
                st.success("Benchmark complete. Explore the insights below.")
                summary_tab, detail_tab, outputs_tab = st.tabs(
                    ["Summary Insights", "Detailed Metrics", "Model Outputs"]
                )

                if combined_summaries:
                    all_summary = pd.concat(combined_summaries, ignore_index=True)
                else:
                    all_summary = pd.DataFrame()

                with summary_tab:
                    render_summary_insights(all_summary)
                    if not all_summary.empty:
                        st.markdown("#### Aggregated Metrics Table")
                        st.dataframe(all_summary, use_container_width=True)
                        download_button("Export Summary CSV", all_summary, "context-comparator-summary.csv")
                        if "aggregate_mean" in all_summary.columns:
                            chart_data = (
                                all_summary.groupby("model")["aggregate_mean"]
                                .mean()
                                .sort_values(ascending=False)
                                .reset_index()
                            )
                            chart_data = chart_data.set_index("model")
                            st.bar_chart(chart_data)

                        available_metrics = [
                            col for col in all_summary.columns if col.endswith("_mean") and col not in {"aggregate_mean"}
                        ]
                        if available_metrics:
                            metric_choice = st.selectbox(
                                "Visualise another metric",
                                options=available_metrics,
                                help="Switch the chart to inspect latency, throughput, or other aggregate statistics.",
                            )
                            metric_chart = (
                                all_summary.groupby("model")[metric_choice]
                                .mean()
                                .sort_values(ascending=False)
                                .reset_index()
                            )
                            metric_chart = metric_chart.set_index("model")
                            st.bar_chart(metric_chart)
                        st.markdown(
                            "Consider weighting quality, latency, and throughput to create a simple decision matrix "
                            "that aligns with your deployment priorities (e.g., 50% quality, 30% latency, 20% cost proxy)."
                        )

                with detail_tab:
                    if combined_records:
                        all_results = pd.concat(combined_records, ignore_index=True)
                        st.markdown("#### Per-Run Metrics")
                        st.dataframe(all_results, use_container_width=True)
                        download_button("Export CSV", all_results, "context-comparator-results.csv")
                    else:
                        st.info("No detailed results available for the current selection.")
                    st.caption("Detailed logs and generated outputs are stored under the `results/` directory.")

                with outputs_tab:
                    if all_outputs:
                        st.markdown("#### Generated Responses")
                        for key, responses in all_outputs.items():
                            model, task_id = key.split("_", 1)
                            with st.expander(f"{model} · {task_id}", expanded=False):
                                for idx, response in enumerate(responses, 1):
                                    st.markdown(f"**Run {idx}**")
                                    st.markdown(response)
                    else:
                        st.info("Outputs will appear here once benchmarks produce generations.")
    else:
        with results_placeholder:
            st.info("Configure your run and click **Run Benchmark** to generate results.")
