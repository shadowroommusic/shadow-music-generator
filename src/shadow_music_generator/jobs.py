from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import uuid

from .pipeline import PipelineStageError, StageRecord, load_pipeline, run_stages

DEFAULT_MODEL_LICENSE = "unknown; confirm the checkpoint license before any commercial use"
NONCOMMERCIAL_WEIGHTS_LICENSE = "CC-BY-NC-4.0 weights; a separate license is required for commercial use"
PERMISSIVE_CODE_LICENSE = "Apache-2.0 code; verify the checkpoint weight terms separately"

# 上游模型名只出现在许可提示里。许可警告必须点名它指的是哪套权重，否则没有意义；
# 除此之外，工具、包、命令与界面文案一律使用 Shadow* 命名。
CHECKPOINT_LICENSES = (
    ("yue2", NONCOMMERCIAL_WEIGHTS_LICENSE),
    ("yue", PERMISSIVE_CODE_LICENSE),
)


def model_license_for(model: str | None) -> str:
    name = (model or "").lower()
    for marker, license_text in CHECKPOINT_LICENSES:
        if marker in name:
            return license_text
    return DEFAULT_MODEL_LICENSE


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    mode: str = "dry-run"
    model: str | None = None
    output_dir: str | None = None
    lyrics: str | None = None
    source_audio: str | None = None


@dataclass(frozen=True)
class GenerationJob:
    id: str
    created_at: str
    status: str
    request: GenerationRequest
    model_license: str
    pipeline: str | None = None
    stages: tuple[StageRecord, ...] = ()
    outputs: tuple[str, ...] = ()
    output: str | None = None
    error: str | None = None


class JobStore:
    """Persist generation jobs as JSON and run them through the staged adapter."""

    MODES = ("dry-run", "local")

    def __init__(self, root: "str | Path | None" = None) -> None:
        self.root = Path(root or os.environ.get("SHADOW_JOB_DIR", ".shadow-jobs"))
        self.root.mkdir(parents=True, exist_ok=True)

    def submit(self, request: GenerationRequest) -> GenerationJob:
        if not request.prompt.strip():
            raise ValueError("prompt must not be empty")
        if request.mode not in self.MODES:
            raise ValueError("mode must be dry-run or local")
        job = GenerationJob(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            status="queued",
            request=request,
            model_license=model_license_for(request.model),
        )
        self._write(job)
        return job

    def get(self, job_id: str) -> GenerationJob:
        path = self.root / f"{job_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no such job: {job_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return GenerationJob(
            id=data["id"],
            created_at=data["created_at"],
            status=data["status"],
            request=GenerationRequest(**data["request"]),
            model_license=data.get("model_license", DEFAULT_MODEL_LICENSE),
            pipeline=data.get("pipeline"),
            stages=tuple(StageRecord(**item) for item in data.get("stages") or []),
            outputs=tuple(data.get("outputs") or ()),
            output=data.get("output"),
            error=data.get("error"),
        )

    def jobs(self) -> "list[GenerationJob]":
        return sorted(
            (self.get(path.stem) for path in self.root.glob("*.json")),
            key=lambda job: job.created_at,
        )

    def run(self, job_id: str, spec: "str | None" = None) -> GenerationJob:
        """Run a queued job. Failures are recorded on the job instead of raised."""
        job = self.get(job_id)
        if job.request.mode == "dry-run":
            result = replace(
                job,
                status="completed",
                stages=(StageRecord("dry-run", "skipped", 0, "dry-run: no model was executed"),),
                output="dry-run: no model executed",
                error=None,
            )
            self._write(result)
            return result

        output_dir = (
            Path(job.request.output_dir).expanduser()
            if job.request.output_dir
            else self.root / "outputs" / job.id
        )
        try:
            pipeline, resolved = load_pipeline(spec)
        except Exception as exc:
            failed = replace(job, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._write(failed)
            return failed

        running = replace(job, status="running", pipeline=resolved)
        self._write(running)
        try:
            stages, outputs = run_stages(pipeline, job.request, output_dir)
        except PipelineStageError as exc:
            failed = replace(running, status="failed", stages=tuple(exc.stages), error=str(exc))
            self._write(failed)
            return failed
        except Exception as exc:
            failed = replace(running, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._write(failed)
            return failed

        result = replace(
            running,
            status="completed",
            stages=tuple(stages),
            outputs=tuple(outputs),
            output=outputs[0] if outputs else f"no artifacts returned (working directory: {output_dir})",
            error=None,
        )
        self._write(result)
        return result

    def _write(self, job: GenerationJob) -> None:
        tmp = self.root / f".{job.id}.tmp"
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.root / f"{job.id}.json")
