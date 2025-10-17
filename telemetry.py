"""Utility helpers to capture host and model telemetry for benchmark runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psutil

try:
    import GPUtil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    GPUtil = None

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

try:
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ollama = None


@dataclass
class GPUStats:
    """Lightweight container for GPU telemetry."""

    name: str
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    utilization: Optional[float] = None
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gpu_stats_via_gputil() -> List[GPUStats]:
    stats: List[GPUStats] = []
    if GPUtil is None:
        return stats

    try:
        for gpu in GPUtil.getGPUs():
            stats.append(
                GPUStats(
                    name=gpu.name,
                    memory_total_mb=gpu.memoryTotal,
                    memory_used_mb=gpu.memoryUsed,
                    utilization=gpu.load * 100,
                    temperature_c=gpu.temperature,
                )
            )
    except Exception:
        return []
    return stats


def _gpu_stats_via_nvml() -> List[GPUStats]:
    stats: List[GPUStats] = []
    if pynvml is None:
        return stats

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = None
            stats.append(
                GPUStats(
                    name=name,
                    memory_total_mb=mem_info.total / (1024 * 1024),
                    memory_used_mb=mem_info.used / (1024 * 1024),
                    utilization=None,
                    temperature_c=None,
                    power_watts=power,
                )
            )
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return stats


def collect_gpu_stats() -> List[Dict[str, Any]]:
    """Gather GPU telemetry using optional providers."""

    stats = _gpu_stats_via_gputil()
    if not stats:
        stats = _gpu_stats_via_nvml()
    return [entry.to_dict() for entry in stats]


def _safe_cpu_percent() -> float:
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0.0


def _safe_load_average() -> Optional[Iterable[float]]:
    try:
        return os.getloadavg()
    except OSError:
        return None


def _ollama_version() -> Optional[str]:
    if ollama is None:
        return None
    return getattr(ollama, "__version__", None)


def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """Fetch metadata for an Ollama model if available."""
    if ollama is None:
        return {}

    try:
        payload = ollama.show(model=model_name)
    except Exception:
        return {}
    details = payload.get("details", {}) if isinstance(payload, dict) else {}
    return {
        "model": payload.get("model"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization"),
        "num_ctx": details.get("context_length"),
        "num_gpu": details.get("gpu_layers"),
    }


def collect_system_snapshot() -> Dict[str, Any]:
    """Collect a host-level telemetry snapshot."""

    virtual_mem = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    load_avg = _safe_load_average()
    uname = platform.uname()
    cpu_info = {
        "model": uname.processor or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "frequency_mhz": cpu_freq.current if cpu_freq else None,
        "utilization_percent": _safe_cpu_percent(),
        "load_average": list(load_avg) if load_avg else None,
    }
    ram_info = {
        "total_gb": round(virtual_mem.total / (1024**3), 2),
        "available_gb": round(virtual_mem.available / (1024**3), 2),
        "used_percent": virtual_mem.percent,
    }

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "python": platform.python_version(),
        "cpu": cpu_info,
        "ram": ram_info,
        "gpu": collect_gpu_stats(),
        "ollama_version": _ollama_version(),
        "env": {
            "shell": os.environ.get("SHELL"),
            "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        },
    }
    return snapshot


def build_run_log(
    models: Iterable[str],
    parameters: Dict[str, Any],
    task_profiles: Iterable[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose a full JSON log payload for a benchmark run."""

    system_snapshot = collect_system_snapshot()
    model_details = {model: get_model_metadata(model) for model in models}

    run_log = {
        "system": system_snapshot,
        "models": model_details,
        "parameters": parameters,
        "task_profiles": list(task_profiles),
    }
    if extra:
        run_log["run_metadata"] = extra
    return run_log


def write_log(payload: Dict[str, Any], destination: Path) -> None:
    """Persist a telemetry payload to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as log_file:
        json.dump(payload, log_file, indent=2)
