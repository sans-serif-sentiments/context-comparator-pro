"""CLI batch runner for Context Comparator Pro."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from rubric import RubricRequest, score_response, score_similarity
from telemetry import build_run_log, write_log

try:
    import ollama  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("The `ollama` Python package is required to run the benchmark.") from exc


RESULTS_DIR = Path(Path.cwd() / "results")
DEFAULT_PROMPTS = Path("prompts/samples.csv")
TASKS_DIR = Path("tasks")


@dataclass
class TaskProfile:
    """Represents a benchmark task profile."""

    identifier: str
    difficulty: str
    goal: str
    metrics: List[str]
    expected_structure: Optional[str]
    prompt: str
    keywords: Optional[List[str]] = None
    target_words: int = 200


PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "eval_stable": {
        "decoding": {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 100,
            "seed": 42,
            "repeat_penalty": 1.1,
        },
        "compute": {
            "num_ctx": 8192,
            "num_thread": 0,  # 0 lets Ollama choose physical cores.
            "num_gpu": -1,  # -1 signals auto.
            "num_batch": 64,
        },
        "quality": {},
    },
    "production": {
        "decoding": {
            "temperature": 0.2,
            "repeat_penalty": 1.2,
        },
        "compute": {},
        "quality": {
            "rubric_weights": [0.4, 0.3, 0.2, 0.1],
            "grammar_enforced": True,
        },
    },
    "creative": {
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "compute": {},
        "quality": {
            "rubric_weights": [0.2, 0.2, 0.3, 0.3],
        },
    },
}


def list_local_models() -> List[str]:
    """Return locally available Ollama model identifiers."""

    try:
        response = ollama.list()
    except Exception:  # pragma: no cover - connectivity safety
        return []
    models = response.get("models", [])
    return [model.get("model") for model in models if model.get("model")]


def parse_task_file(path: Path) -> TaskProfile:
    """Read a task profile markdown file with lightweight front-matter."""

    metadata: Dict[str, str] = {}
    prompt_lines: List[str] = []
    in_front_matter = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "---" and not in_front_matter:
                in_front_matter = True
                continue
            if stripped.startswith("--------------------------------"):
                in_front_matter = False
                continue
            if in_front_matter:
                if ":" in stripped:
                    key, value = stripped.split(":", 1)
                    metadata[key.strip()] = value.strip()
            else:
                prompt_lines.append(line.rstrip("\n"))
    prompt = "\n".join(prompt_lines).strip()
    metrics = metadata.get("metrics", "")
    metrics_list = [item.strip() for item in metrics.strip("[]").split(",") if item.strip()]
    keywords_meta = metadata.get("keywords")
    keywords_list = (
        [item.strip() for item in keywords_meta.strip("[]").split(",") if item.strip()]
        if keywords_meta
        else None
    )
    target_words = 200
    if "target_words" in metadata:
        try:
            target_words = int(metadata["target_words"])
        except ValueError:
            target_words = 200
    return TaskProfile(
        identifier=metadata.get("id", path.stem),
        difficulty=metadata.get("difficulty", "unknown"),
        goal=metadata.get("goal", ""),
        metrics=metrics_list,
        expected_structure=metadata.get("expected_structure"),
        prompt=prompt.split("Prompt:", 1)[-1].strip() if "Prompt:" in prompt else prompt,
        keywords=keywords_list,
        target_words=target_words,
    )


def load_task_profile(identifier: str) -> TaskProfile:
    """Load the requested task profile by identifier."""

    path = TASKS_DIR / f"{identifier}.md"
    if not path.exists():
        available = [p.stem for p in TASKS_DIR.glob("*.md")]
        raise FileNotFoundError(f"Task profile '{identifier}' not found. Available: {available}")
    return parse_task_file(path)


def load_prompts(csv_path: Path, profile: Optional[str]) -> pd.DataFrame:
    """Load prompts from CSV and optionally filter by task profile."""

    if not csv_path.exists():
        return pd.DataFrame(columns=["task_id", "prompt", "expected_output"])
    df = pd.read_csv(csv_path)
    if profile and "task_id" in df.columns:
        df = df[df["task_id"] == profile]
    return df


def merge_options(
    decoding_overrides: Optional[Dict[str, float]] = None,
    compute_overrides: Optional[Dict[str, float]] = None,
    preset: Optional[str] = None,
) -> Dict[str, float]:
    """Merge preset, decoding, and compute options into a single dict."""

    options: Dict[str, float] = {}
    if preset and preset in PRESETS:
        options.update(PRESETS[preset]["decoding"])
        options.update(PRESETS[preset]["compute"])
    if decoding_overrides:
        options.update(decoding_overrides)
    if compute_overrides:
        options.update(compute_overrides)
    return options


def _rubric_weights_for_preset(preset: Optional[str]) -> Optional[List[float]]:
    if not preset or preset not in PRESETS:
        return None
    candidate = PRESETS[preset]["quality"].get("rubric_weights")
    if isinstance(candidate, list):
        return candidate
    return None


def _collect_chunk_metrics(chunk: Dict[str, float], accumulator: Dict[str, float]) -> None:
    for key in ("eval_count", "prompt_eval_count", "eval_duration", "prompt_eval_duration", "total_duration"):
        if key in chunk:
            accumulator[key] = chunk[key]


def generate_with_metrics(model: str, prompt: str, options: Dict[str, float]) -> Dict[str, any]:
    """Run a streaming generation request and capture timing/usage metrics."""

    start_time = time.perf_counter()
    response_parts: List[str] = []
    accumulator: Dict[str, float] = {}
    ttft: Optional[float] = None
    for chunk in ollama.generate(model=model, prompt=prompt, stream=True, options=options):
        if ttft is None:
            ttft = time.perf_counter() - start_time
        response_parts.append(chunk.get("response", ""))
        _collect_chunk_metrics(chunk, accumulator)
    total_latency = time.perf_counter() - start_time
    eval_duration_s = (accumulator.get("eval_duration") or 0.0) / 1e9
    prompt_eval_duration_s = (accumulator.get("prompt_eval_duration") or 0.0) / 1e9
    total_duration_s = (accumulator.get("total_duration") or 0.0) / 1e9 or total_latency
    eval_count = accumulator.get("eval_count") or 0
    tokens_per_sec = eval_count / eval_duration_s if eval_duration_s else None

    metrics = {
        "latency_s": round(total_latency, 4),
        "ttft_s": round(ttft or total_latency, 4),
        "eval_count": int(eval_count),
        "prompt_eval_count": int(accumulator.get("prompt_eval_count") or 0),
        "eval_duration_s": round(eval_duration_s, 4),
        "prompt_eval_duration_s": round(prompt_eval_duration_s, 4),
        "total_duration_s": round(total_duration_s, 4),
        "tokens_per_sec": round(tokens_per_sec, 4) if tokens_per_sec else None,
        "cost_proxy": round(total_latency / tokens_per_sec, 4) if tokens_per_sec else None,
    }
    return {
        "response": "".join(response_parts).strip(),
        "metrics": metrics,
    }


def run_benchmark(
    models: Sequence[str],
    task: TaskProfile,
    repeats: int,
    prompt_rows: pd.DataFrame,
    options: Dict[str, float],
    preset: Optional[str],
    warmup: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Execute the benchmark runs and return a dataframe of metrics."""

    records: List[Dict[str, any]] = []
    outputs: Dict[str, List[str]] = defaultdict(list)
    rubric_weights = _rubric_weights_for_preset(preset)
    task_keywords = getattr(task, "keywords", None)
    target_words = getattr(task, "target_words", 200)

    prompts_to_run: List[Tuple[str, str]] = []
    if prompt_rows.empty:
        prompts_to_run.append((task.identifier, task.prompt))
    else:
        for _, row in prompt_rows.iterrows():
            prompt_value = row.get("prompt")
            if not isinstance(prompt_value, str) or not prompt_value.strip():
                continue
            prompts_to_run.append((row.get("task_id", task.identifier), prompt_value))

    for model in models:
        if warmup:
            try:
                ollama.generate(model=model, prompt="Warm-up run.", stream=False)
            except Exception:
                pass
        for task_id, prompt in prompts_to_run:
            for run_idx in range(1, repeats + 1):
                generation = generate_with_metrics(model, prompt, options)
                response = generation["response"]
                metrics = generation["metrics"]
                rubric_request = RubricRequest(
                    prompt=prompt,
                    response=response,
                    expected_structure=task.expected_structure,
                    keywords=task_keywords,
                    weights=None if rubric_weights is None else {
                        "faithfulness": rubric_weights[0],
                        "completeness": rubric_weights[1],
                        "structure": rubric_weights[2],
                        "conciseness": rubric_weights[3],
                    },
                    target_word_count=target_words,
                )
                quality_scores = score_response(rubric_request)
                outputs[f"{model}_{task_id}"].append(response)
                record = {
                    "model": model,
                    "task_id": task_id,
                    "run": run_idx,
                    "target_words": target_words,
                    "keywords": ", ".join(task_keywords) if task_keywords else "",
                    **metrics,
                    **quality_scores,
                }
                records.append(record)
    df = pd.DataFrame(records)
    return df, outputs


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation metrics grouped by model/task."""

    if df.empty:
        return df
    grouped = []
    for (model, task_id), group in df.groupby(["model", "task_id"]):
        row: Dict[str, any] = {"model": model, "task_id": task_id}
        for metric in ["latency_s", "tokens_per_sec", "ttft_s", "aggregate"]:
            values = group[metric].dropna().tolist()
            if not values:
                continue
            row[f"{metric}_mean"] = round(mean(values), 4)
            row[f"{metric}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
        row["runs"] = len(group)
        grouped.append(row)
    return pd.DataFrame(grouped)


def compute_similarity(outputs: Dict[str, List[str]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for key, responses in outputs.items():
        similarity = score_similarity(responses)
        if similarity is not None:
            scores[key] = similarity
    return scores


def write_outputs(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outputs: Dict[str, List[str]],
    task: TaskProfile,
    models: Sequence[str],
    preset: Optional[str],
    repeats: int,
) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    compare_path = results_dir / f"compare_{timestamp}.csv"
    json_path = results_dir / f"compare_{timestamp}.json"
    markdown_path = results_dir / "reports" / f"report_{timestamp}.md"

    df.to_csv(compare_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    outputs_dir = results_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for key, responses in outputs.items():
        model, task_id = key.split("_", 1)
        output_file = outputs_dir / f"{model}_{task_id}.md"
        content_lines = [f"# {model} · {task_id}", ""]
        for idx, response in enumerate(responses, 1):
            content_lines.append(f"## Run {idx}")
            content_lines.append("")
            content_lines.append(response)
            content_lines.append("")
        output_file.write_text("\n".join(content_lines), encoding="utf-8")

    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(reports_dir / f"summary_{timestamp}.csv", index=False)

    recommendation = build_recommendation(summary_df)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(recommendation, encoding="utf-8")

    run_log = build_run_log(
        models=models,
        parameters={"preset": preset, "repeats": repeats},
        task_profiles=[task.identifier],
        extra={"timestamp": timestamp, "records": len(df)},
    )
    log_path = results_dir / "logs" / f"run_{timestamp}.json"
    write_log(run_log, log_path)


def build_recommendation(summary_df: pd.DataFrame) -> str:
    """Generate a lightweight Markdown recommendation report."""

    if summary_df.empty:
        return "# Context Comparator Pro Report\n\nNo runs were recorded."

    lines = [
        "# Context Comparator Pro Report",
        "",
        "## Ranking",
        "",
        "| Model | Task | Aggregate Mean | Latency Mean (s) | Tokens/s Mean |",
        "|-------|------|----------------|------------------|---------------|",
    ]
    for _, row in summary_df.sort_values("aggregate_mean", ascending=False).iterrows():
        lines.append(
            f"| {row['model']} | {row['task_id']} | {row.get('aggregate_mean', 'n/a')} | "
            f"{row.get('latency_s_mean', 'n/a')} | {row.get('tokens_per_sec_mean', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Inspect individual run logs in `results/logs/` for telemetry details.",
            "- Generated outputs are saved under `results/outputs/` for qualitative review.",
        ]
    )
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch benchmark local Ollama models.")
    parser.add_argument("--models", nargs="+", required=False, help="Models to compare.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PROMPTS, help="Prompt CSV path.")
    parser.add_argument("--profile", type=str, default="summarization", help="Task profile identifier.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per prompt.")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Apply decoding preset.")
    parser.add_argument("--warmup", action="store_true", help="Run a warm-up call before benchmarking.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    models = args.models or list_local_models()
    if not models:
        raise SystemExit("No models provided or discovered. Use --models to specify targets.")

    task = load_task_profile(args.profile)
    prompt_rows = load_prompts(args.csv, args.profile)
    options = merge_options(preset=args.preset)

    df, outputs = run_benchmark(
        models=models,
        task=task,
        repeats=max(1, args.repeats),
        prompt_rows=prompt_rows,
        options=options,
        preset=args.preset,
        warmup=args.warmup,
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
        models=models,
        preset=args.preset,
        repeats=max(1, args.repeats),
    )

    print("Benchmark complete.")
    print(f"Models: {', '.join(models)}")
    print(f"Task: {task.identifier}")
    print(f"Repeats: {args.repeats}")


if __name__ == "__main__":
    main()
