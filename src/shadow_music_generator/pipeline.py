from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any

REQUIRED_STAGES = ("plan", "generate_semantic", "synthesize", "decode")
FROZEN_CONTEXT_KEYS = {"prompt", "lyrics", "model", "source_audio", "output_dir"}


class PipelineNotConfigured(RuntimeError):
    """Raised when local mode is requested without a configured pipeline factory."""


class PipelineContractError(RuntimeError):
    """Raised when the configured factory does not implement the staged contract."""


class PipelineStageError(RuntimeError):
    """Raised when one stage fails while running a job."""

    def __init__(self, stage: str, message: str, stages: "list[StageRecord]") -> None:
        super().__init__(f"stage '{stage}' failed: {message}")
        self.stage = stage
        self.stages = stages


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: str
    duration_ms: int
    detail: str | None = None


def resolve_spec(spec: "str | None" = None, environ: "dict | None" = None) -> "str | None":
    """Return the configured factory spec, preferring an explicit argument."""
    if spec and spec.strip():
        return spec.strip()
    environment = os.environ if environ is None else environ
    value = (environment.get("SHADOW_PIPELINE_FACTORY") or "").strip()
    return value or None


def load_object(spec: str) -> Any:
    """Import `module:attribute`, `module.attribute` or `path/to/file.py:attribute`."""
    module_ref, separator, attribute = spec.partition(":")
    if not separator or not attribute:
        if "." in spec and not spec.endswith(".py"):
            module_ref, _, attribute = spec.rpartition(".")
        else:
            raise PipelineContractError(
                f"SHADOW_PIPELINE_FACTORY must look like 'my_module:factory' or '/path/adapter.py:factory' (got {spec!r})"
            )
    if not module_ref or not attribute:
        raise PipelineContractError(f"SHADOW_PIPELINE_FACTORY is incomplete: {spec!r}")
    module_path = Path(module_ref).expanduser()
    if module_ref.endswith(".py") or os.sep in module_ref:
        if not module_path.is_file():
            raise PipelineContractError(f"adapter file does not exist: {module_path}")
        parent = str(module_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module_spec = importlib.util.spec_from_file_location(f"shadow_pipeline_adapter_{module_path.stem}", module_path)
        if module_spec is None or module_spec.loader is None:
            raise PipelineContractError(f"cannot load adapter file: {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_ref)
        except ImportError as exc:
            raise PipelineContractError(f"cannot import '{module_ref}': {exc}") from exc
    target = getattr(module, attribute, None)
    if target is None:
        raise PipelineContractError(f"'{module_ref}' has no attribute '{attribute}'")
    return target


def load_pipeline(spec: "str | None" = None, environ: "dict | None" = None) -> "tuple[Any, str]":
    """Build the pipeline object and return it with the spec that produced it."""
    resolved = resolve_spec(spec, environ)
    if not resolved:
        raise PipelineNotConfigured(
            "local generation needs an explicitly installed pipeline: set SHADOW_PIPELINE_FACTORY=my_module:factory "
            "(see README.md). Nothing was downloaded or executed."
        )
    factory = load_object(resolved)
    pipeline = factory() if callable(factory) else factory
    missing = [name for name in REQUIRED_STAGES if not callable(getattr(pipeline, name, None))]
    if missing:
        raise PipelineContractError(
            f"pipeline '{resolved}' does not implement the staged interface; missing: {', '.join(missing)}"
        )
    return pipeline, resolved


def _collect_outputs(context: dict, fallback: Any) -> "list[str]":
    candidates = context.get("artifacts") or []
    if not candidates and fallback is not None:
        if isinstance(fallback, (str, os.PathLike)):
            candidates = [fallback]
        elif isinstance(fallback, (list, tuple)):
            candidates = list(fallback)
    outputs: "list[str]" = []
    for item in candidates:
        if isinstance(item, (str, os.PathLike)):
            outputs.append(str(item))
        elif isinstance(item, dict) and item.get("path"):
            outputs.append(str(item["path"]))
    return outputs


def run_stages(pipeline: Any, request: Any, output_dir: "str | Path") -> "tuple[list[StageRecord], list[str]]":
    """Run plan → generate_semantic → synthesize → decode against one context dict.

    Every stage receives the same mutable context and may return a mapping that is
    merged back into it (except for the frozen request keys). The final stage is
    expected to fill `context["artifacts"]` with output paths.
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    context: "dict[str, Any]" = {
        "prompt": request.prompt,
        "lyrics": request.lyrics,
        "model": request.model,
        "source_audio": request.source_audio,
        "output_dir": str(output_path),
        "artifacts": [],
    }
    records: "list[StageRecord]" = []
    last_result: Any = None
    for name in REQUIRED_STAGES:
        stage = getattr(pipeline, name)
        started = time.perf_counter()
        try:
            result = stage(context)
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000)
            records.append(StageRecord(name, "failed", elapsed, f"{type(exc).__name__}: {exc}"))
            raise PipelineStageError(name, f"{type(exc).__name__}: {exc}", records) from exc
        elapsed = round((time.perf_counter() - started) * 1000)
        last_result = result
        detail = None
        if isinstance(result, dict):
            for key, value in result.items():
                if key not in FROZEN_CONTEXT_KEYS:
                    context[key] = value
            detail = ", ".join(sorted(str(key) for key in result)) or None
        elif result is not None:
            context[name] = result
            detail = type(result).__name__
        records.append(StageRecord(name, "completed", elapsed, detail))
    return records, _collect_outputs(context, last_result)
