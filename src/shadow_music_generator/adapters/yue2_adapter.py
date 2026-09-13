"""YuE / YuE2 adapter for shadow-music-generator.

YuE2's staged Python API is ``plan() → generate_semantic() → synthesize() → decode()``, which maps
1:1 onto this plugin's four-stage contract, so a YuE2 job gets per-stage timings and artifacts.

Two ways to run it:

1. **Pipeline mode** (default, recommended): the plugin runs inside the YuE environment and
   ``yue2.YuE2Pipeline`` is importable. Each contract stage calls the matching YuE2 stage and the
   decoded audio is written into the job's output directory.
2. **Command mode**: set ``YUE_COMMAND`` to a command template that runs YuE anywhere — a GPU box
   over ssh, a container, a wrapper script. The command performs the whole generation;
   ``generate_semantic`` runs it, ``synthesize`` checks the expected audio files, ``decode``
   collects them.

Environment:

===========================  ==================================================
``YUE_MODEL``                checkpoint (default ``m-a-p/YuE2-3B``)
``YUE_DEVICE``               torch device (default ``cuda``)
``YUE_VAE``                  optional VAE override, e.g. ``m-a-p/YuE2-Vae-legacy``
``YUE_COT``                  ``full`` | ``melody`` | ``off`` (default ``full``)
``YUE_SEED``                 optional integer seed
``YUE_LYRICS``               lyrics text, or a path to a lyrics file
``YUE_ABC``                  ABC score text, or a path to an ``.abc`` file
``YUE_REQUEST_JSON``         full YuE2 request file; its keys are used as defaults
``YUE_SAMPLE_RATE``          sample rate for the written audio (default ``48000``)
``YUE_COMMAND``              command template for command mode
``YUE_OUTPUTS``              glob for produced audio (default ``**/*.flac``)
===========================  ==================================================

Command templates are split with :func:`shlex.split` and run without a shell; the placeholders
``{prompt}``, ``{lyrics}``, ``{model}``, ``{output_dir}`` and ``{request_json}`` are substituted.
"""
from __future__ import annotations

import glob
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_DEVICE = "cuda"
DEFAULT_COT = "full"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_OUTPUT_GLOB = "**/*.flac"
AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".aiff", ".aif", ".m4a", ".ogg", ".opus"}


class YuEAdapterError(RuntimeError):
    """Raised when the YuE environment or the produced outputs are not usable."""


def _env(name: str, default: "str | None" = None) -> "str | None":
    value = (os.environ.get(name) or "").strip()
    return value or default


def _text_or_file(value: "str | None") -> "str | None":
    """``YUE_LYRICS`` / ``YUE_ABC`` may hold literal text or a path."""
    if not value:
        return None
    candidate = Path(value).expanduser()
    try:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    except OSError:
        pass
    return value


class YuE2Adapter:
    """Four-stage adapter (plan / generate_semantic / synthesize / decode) for YuE2."""

    def __init__(
        self,
        *,
        model: "str | None" = None,
        device: "str | None" = None,
        vae: "str | None" = None,
        cot: "str | None" = None,
        seed: "int | None" = None,
        lyrics: "str | None" = None,
        abc: "str | None" = None,
        request_json: "str | None" = None,
        sample_rate: "int | None" = None,
        command: "str | None" = None,
        outputs_glob: "str | None" = None,
        pipeline_factory: Any = None,
    ) -> None:
        self.model = model or _env("YUE_MODEL", DEFAULT_MODEL)
        self.device = device or _env("YUE_DEVICE", DEFAULT_DEVICE)
        self.vae = vae if vae is not None else _env("YUE_VAE")
        self.cot = cot or _env("YUE_COT", DEFAULT_COT)
        seed_text = _env("YUE_SEED")
        self.seed = seed if seed is not None else (int(seed_text) if seed_text else None)
        self.lyrics = _text_or_file(lyrics if lyrics is not None else _env("YUE_LYRICS"))
        self.abc = _text_or_file(abc if abc is not None else _env("YUE_ABC"))
        self.request_json = request_json if request_json is not None else _env("YUE_REQUEST_JSON")
        sample_rate_text = _env("YUE_SAMPLE_RATE")
        self.sample_rate = sample_rate or (int(sample_rate_text) if sample_rate_text else DEFAULT_SAMPLE_RATE)
        self.command = command if command is not None else _env("YUE_COMMAND")
        self.outputs_glob = outputs_glob or _env("YUE_OUTPUTS", DEFAULT_OUTPUT_GLOB)
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._plan: Any = None
        self._semantic: Any = None
        self._latents: Any = None

    # ------------------------------------------------------------------ helpers

    def _request(self, context: "dict[str, Any]") -> "dict[str, Any]":
        request: "dict[str, Any]" = {}
        if self.request_json:
            path = Path(self.request_json).expanduser()
            if not path.is_file():
                raise YuEAdapterError(f"YUE_REQUEST_JSON does not exist: {path}")
            request.update(json.loads(path.read_text(encoding="utf-8")))
        request["style"] = context.get("prompt") or request.get("style") or ""
        lyrics = context.get("lyrics") or self.lyrics
        if lyrics:
            request["lyrics"] = lyrics
        if self.seed is not None:
            request["seed"] = self.seed
        if self.abc:
            request["abc"] = self.abc
        request.setdefault("cot", self.cot)
        if self.model:
            request.setdefault("model", self.model)
        return request

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        factory = self._pipeline_factory
        if factory is None:
            try:
                module = importlib.import_module("yue2")
            except ImportError as exc:  # pragma: no cover - depends on the YuE environment
                raise YuEAdapterError(
                    "the 'yue2' package is not importable: run this plugin inside the YuE environment "
                    "(python -m pip install . in the YuE checkout), or set YUE_COMMAND to run YuE elsewhere"
                ) from exc
            factory = getattr(module, "YuE2Pipeline", None)
            if factory is None:
                raise YuEAdapterError("'yue2' does not expose YuE2Pipeline")
        kwargs: "dict[str, Any]" = {"device": self.device}
        if self.vae:
            kwargs["vae"] = self.vae
        pipeline = factory.from_pretrained(self.model, **kwargs)
        if hasattr(pipeline, "__enter__"):
            pipeline = pipeline.__enter__()
        self._pipeline = pipeline
        return pipeline

    def _audio_outputs(self, output_dir: Path) -> "list[str]":
        found: "list[str]" = []
        for pattern in (self.outputs_glob, "**/*.wav", "**/*.mp3", "**/*.aiff", "**/*.m4a"):
            for item in glob.glob(str(output_dir / pattern), recursive=True):
                path = Path(item)
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and str(path) not in found:
                    found.append(str(path))
        return sorted(found)

    # ------------------------------------------------------------------- stages

    def plan(self, context: "dict[str, Any]") -> "dict[str, Any]":
        """Validate the environment and freeze the YuE request for this job."""
        output_dir = Path(context["output_dir"]).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        request = self._request(context)
        (output_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.command:
            return {"plan": {"mode": "command", "command": self.command, "request": request}}
        pipeline = self._load_pipeline()
        self._plan = pipeline.plan(**request)
        if hasattr(self._plan, "save"):
            self._plan.save(str(output_dir / "plan"))
        return {
            "plan": {
                "mode": "pipeline",
                "model": self.model,
                "device": self.device,
                "cot": request.get("cot"),
                "seed": request.get("seed"),
            }
        }

    def generate_semantic(self, context: "dict[str, Any]") -> "dict[str, Any]":
        """Command mode: run the generation command. Pipeline mode: YuE2 ``generate_semantic``."""
        if self.command:
            output_dir = Path(context["output_dir"]).expanduser()
            substitutions = {
                "prompt": context.get("prompt") or "",
                "lyrics": context.get("lyrics") or self.lyrics or "",
                "model": self.model or "",
                "output_dir": str(output_dir),
                "request_json": str(output_dir / "request.json"),
            }
            argv = [part.format(**substitutions) for part in shlex.split(self.command)]
            completed = subprocess.run(argv, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
                raise YuEAdapterError(f"YUE_COMMAND failed ({completed.returncode}): {' | '.join(tail)}")
            return {"command_returncode": completed.returncode}
        if self._plan is None:
            raise YuEAdapterError("generate_semantic called before plan")
        self._semantic = self._pipeline.generate_semantic(self._plan)
        return {"semantic_ready": True}

    def synthesize(self, context: "dict[str, Any]") -> "dict[str, Any]":
        """Command mode: check the outputs. Pipeline mode: YuE2 ``synthesize``."""
        output_dir = Path(context["output_dir"]).expanduser()
        if self.command:
            produced = self._audio_outputs(output_dir)
            if not produced:
                raise YuEAdapterError(f"YUE_COMMAND produced no audio under {output_dir}")
            return {"produced": len(produced)}
        if self._semantic is None:
            raise YuEAdapterError("synthesize called before generate_semantic")
        self._latents = self._pipeline.synthesize(self._semantic)
        return {"latents_ready": True}

    def decode(self, context: "dict[str, Any]") -> "dict[str, Any]":
        """Command mode: collect the outputs. Pipeline mode: YuE2 ``decode`` → ``audio.flac``."""
        output_dir = Path(context["output_dir"]).expanduser()
        if self.command:
            return {"artifacts": self._audio_outputs(output_dir)}
        if self._latents is None:
            raise YuEAdapterError("decode called before synthesize")
        audio = self._pipeline.decode(self._latents)
        target = output_dir / "audio.flac"
        try:
            import soundfile  # type: ignore

            soundfile.write(str(target), audio, self.sample_rate)
        except ImportError:  # pragma: no cover - YuE installs soundfile, this is a safety net
            target = output_dir / "audio.wav"
            _write_wav(target, audio, self.sample_rate)
        if hasattr(self._pipeline, "__exit__"):
            try:
                self._pipeline.__exit__(None, None, None)
            finally:
                self._pipeline = None
        return {"artifacts": [str(target)]}


def _write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    """Minimal float→16-bit WAV writer, used only when soundfile is unavailable."""
    import struct
    import wave

    samples = audio
    try:  # numpy arrays: (frames, channels)
        frames = len(samples)
        channels = len(samples[0]) if frames and hasattr(samples[0], "__len__") else 1
    except TypeError:  # pragma: no cover - defensive
        raise YuEAdapterError("cannot write audio: unsupported container returned by YuE2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for frame in samples:
            values = frame if channels > 1 else (frame,)
            handle.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, float(v))) * 32767)) for v in values))


def make_pipeline() -> YuE2Adapter:
    """Entry point for ``SHADOW_PIPELINE_FACTORY``."""
    return YuE2Adapter()
