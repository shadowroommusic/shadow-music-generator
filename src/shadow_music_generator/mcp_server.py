from __future__ import annotations

import json
import sys

from .cli import job_payload
from .jobs import GenerationRequest, JobStore
from .notes import SCALES, transform_notes
from .structure import OPERATIONS, arrange
from .synth import (
    container_capabilities,
    export_midi,
    export_stems,
    mix_arrangement,
    regenerate_part,
    render_part,
    render_song,
)

TOOLS = {
    "submit_generation": "Queue a Shadow Music Generator job. Dry-run validates the request and never runs a model.",
    "run_job": "Run a queued job through the configured SHADOW_PIPELINE_FACTORY adapter.",
    "job_status": "Read one job: status, stages, outputs, model license and errors.",
    "render_part": "Render one musical part (drums/bass/chords/lead/pad) to a WAV file with the local synth.",
    "render_song": "Render a whole sketch — several parts plus a mix — to WAV files with the local synth.",
    "mix_arrangement": "Bounce an arrangement of clips to one file, at a chosen sample rate and container (WAV/AIFF).",
    "export_stems": "Bounce every track of an arrangement to its own file (stems).",
    "export_midi": "Write an arrangement of notated clips to a Type-1 MIDI file, one track per part.",
    "arrange": "Edit the arrangement: insert empty bars, repeat a section, delete one — every clip's new position comes back as data.",
    "regenerate_part": "Re-render ONE part of a song from the plan it was made with, plus a small patch (gain/swing/cutoff/wave/pattern…).",
    "transform_notes": "Deterministically edit a note list (transpose / quantise / scale / humanise / shift / velocity) — the arithmetic an agent should not do in its head.",
    "list_export_formats": "Which containers this machine can write (wav/aiff always; flac/alac/aac and mp3/ogg/opus need an encoder).",
}

#: Clip shape shared by the arrangement tools.
_CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Audio clip — WAV, or mp3/m4a/flac/aiff through a decoder — or a .mid for MIDI export."},
        "start": {"type": "number", "description": "Position in bars."},
        "bars": {"type": "number", "description": "Length on the grid, in bars."},
        "gain": {"type": "number"},
        "notes": {
            "type": "array",
            "description": "Notated notes, milliseconds relative to the clip.",
            "items": {
                "type": "object",
                "properties": {
                    "midi": {"type": "integer"},
                    "start_ms": {"type": "number"},
                    "end_ms": {"type": "number"},
                    "velocity": {"type": "integer"},
                },
                "required": ["midi", "start_ms", "end_ms"],
            },
        },
    },
}

_TRACK_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "gain": {"type": "number"},
        "clips": {"type": "array", "items": _CLIP_SCHEMA},
    },
    "required": ["clips"],
}

_ARRANGEMENT_PROPERTIES = {
    "bpm": {"type": "number", "default": 120},
    "bars": {"type": "number", "default": 8},
    "tracks": {"type": "array", "items": _TRACK_SCHEMA},
}

SERVER_NAME = "shadow-music-generator"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSION = "2024-11-05"

# Methods other MCP clients probe during capability negotiation. Answering with a
# valid empty result keeps the server usable from any agent, not just Codex.
EMPTY_RESULTS = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
    "logging/setLevel": {},
}


def _schema(name: str) -> dict:
    if name == "arrange":
        return {
            "type": "object",
            "properties": {
                "revision": {"type": "string", "description": "The arrangement revision these positions came from; echoed back."},
                "bars": {"type": "number", "description": "Current length in bars."},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "start": {"type": "number"},
                            "bars": {"type": "number"},
                        },
                        "required": ["name", "start", "bars"],
                    },
                },
                "tracks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "clips": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "start": {"type": "number"},
                                        "length": {"type": "number"},
                                    },
                                    "required": ["id", "start", "length"],
                                },
                            },
                        },
                        "required": ["name", "clips"],
                    },
                },
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": list(OPERATIONS)},
                            "at_bar": {"type": "number", "description": "insert_bars: 0-based bar"},
                            "count": {"type": "number", "description": "insert_bars: how many bars"},
                            "name": {"type": "string", "description": "insert_bars: what to call the new section"},
                            "section": {"type": "string", "description": "duplicate/remove: section name or id"},
                        },
                        "required": ["op"],
                    },
                },
            },
            "required": ["bars", "sections", "tracks", "operations"],
            "additionalProperties": False,
        }
    if name == "regenerate_part":
        return {
            "type": "object",
            "properties": {
                "out_path": {"type": "string", "description": "Where the new stem goes."},
                "clip_id": {"type": "string", "description": "The clip this part belongs to; echoed back so the UI can replace it."},
                "bpm": {"type": "number", "default": 120},
                "bars": {"type": "number", "default": 4},
                "sections": {"type": "array", "items": {"type": "object"}},
                "seed": {"type": "integer"},
                "part": {"type": "object", "description": "The part exactly as render_song reported it under `spec`."},
                "patch": {
                    "type": "object",
                    "description": "The change to make; only these keys are accepted.",
                    "properties": {
                        "gain_db": {"type": "number", "description": "-24…+6 dB"},
                        "cutoff": {"type": "number", "description": "0–1 lowpass ratio, + is brighter"},
                        "swing": {"type": "number", "description": "added swing, 0–0.4"},
                        "humanize_ms": {"type": "number", "description": "added timing spread, 0–40 ms"},
                        "pan": {"type": "number", "description": "-1 left … +1 right"},
                        "reverb": {"type": "number", "description": "added wet amount"},
                        "delay": {"type": "number", "description": "added delay mix"},
                        "delay_ms": {"type": "number"},
                        "wave": {"type": "string", "enum": ["sine", "triangle", "square", "saw"]},
                        "pattern": {"type": "object", "additionalProperties": {"type": "string"}},
                        "remove_voices": {"type": "array", "items": {"type": "string"}},
                        "transpose": {"type": "integer"},
                        "velocity_scale": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["out_path", "clip_id", "part", "patch"],
            "additionalProperties": False,
        }
    if name == "transform_notes":
        return {
            "type": "object",
            "properties": {
                "clip_id": {"type": "string", "description": "The clip these notes belong to; echoed back."},
                "revision": {"type": "string", "description": "The revision the caller sent; echoed back so a stale edit can be refused."},
                "seed": {"type": "integer", "default": 7, "description": "Makes humanise reproducible."},
                "notes": {
                    "type": "array",
                    "description": "The notes exactly as the piano roll holds them.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "pitch": {"type": "integer"},
                            "start_ms": {"type": "number"},
                            "end_ms": {"type": "number"},
                            "velocity": {"type": "integer"},
                        },
                        "required": ["pitch", "start_ms", "end_ms"],
                    },
                },
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["transpose", "quantise", "set_scale", "humanise", "shift", "velocity"],
                            },
                            "semitones": {"type": "integer", "description": "transpose"},
                            "grid_ms": {"type": "number", "description": "quantise grid, e.g. 125 = a 16th at 120 BPM"},
                            "strength": {"type": "number", "description": "quantise: 1 = exactly on the grid"},
                            "root": {"type": "string", "description": "set_scale root, e.g. A"},
                            "scale": {"type": "string", "enum": sorted(SCALES)},
                            "timing_ms": {"type": "number", "description": "humanise timing spread"},
                            "velocity": {"type": "number", "description": "humanise velocity spread"},
                            "ms": {"type": "number", "description": "shift"},
                            "add": {"type": "number", "description": "velocity: added to each note"},
                            "scale_velocity": {"type": "number"},
                            "note_ids": {"type": "array", "items": {"type": "string"}, "description": "Limit the operation to these notes"},
                        },
                        "required": ["op"],
                    },
                },
            },
            "required": ["notes", "operations"],
            "additionalProperties": False,
        }
    if name == "list_export_formats":
        return {"type": "object", "properties": {}, "additionalProperties": False}
    if name == "submit_generation":
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "mode": {"type": "string", "enum": ["dry-run", "local"], "default": "dry-run"},
                "model": {"type": "string"},
                "lyrics": {"type": "string"},
                "source_audio": {"type": "string"},
                "output_dir": {"type": "string"},
                "job_dir": {"type": "string"},
                "run": {"type": "boolean", "default": False, "description": "Run immediately. Dry-run jobs always run."},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }
    if name == "mix_arrangement":
        return {
            "type": "object",
            "properties": {
                "out_path": {"type": "string"},
                "sample_rate": {"type": "integer", "enum": [44100, 48000, 96000], "default": 44100},
                "bit_depth": {"type": "integer", "enum": [16, 24], "default": 16},
                "container": {
                    "type": "string",
                    "enum": ["wav", "aiff", "flac", "alac", "m4a", "mp3", "ogg", "opus"],
                    "default": "wav",
                    "description": (
                        "wav / aiff are written here; flac / alac / m4a(AAC) use afconvert (macOS) or ffmpeg; "
                        "mp3 / ogg / opus need ffmpeg. Call list_export_formats to see what this machine has."
                    ),
                },
                "bitrate": {"type": "integer", "description": "Lossy bitrate (m4a default 256000, mp3 320000, ogg 192000, opus 128000)."},
                **_ARRANGEMENT_PROPERTIES,
            },
            "required": ["out_path", "tracks"],
        }
    if name == "export_stems":
        return {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "description": "Directory that receives one file per track."},
                "name": {"type": "string", "description": "Prefix for the stem files."},
                "sample_rate": {"type": "integer", "enum": [44100, 48000, 96000], "default": 44100},
                "bit_depth": {"type": "integer", "enum": [16, 24], "default": 16},
                "container": {
                    "type": "string",
                    "enum": ["wav", "aiff", "flac", "alac", "m4a", "mp3", "ogg", "opus"],
                    "default": "wav",
                    "description": (
                        "wav / aiff are written here; flac / alac / m4a(AAC) use afconvert (macOS) or ffmpeg; "
                        "mp3 / ogg / opus need ffmpeg. Call list_export_formats to see what this machine has."
                    ),
                },
                "bitrate": {"type": "integer", "description": "Lossy bitrate (m4a default 256000, mp3 320000, ogg 192000, opus 128000)."},
                **_ARRANGEMENT_PROPERTIES,
            },
            "required": ["out_dir", "tracks"],
        }
    if name == "export_midi":
        return {
            "type": "object",
            "properties": {
                "out_path": {"type": "string", "description": "MIDI file to write."},
                **_ARRANGEMENT_PROPERTIES,
            },
            "required": ["out_path", "tracks"],
        }
    if name in ("render_part", "render_song"):
        note = {
            "type": "object",
            "properties": {
                "midi": {"type": "integer", "description": "MIDI note number (60 = C4)."},
                "start": {"type": "number", "description": "Start in beats from the beginning."},
                "length": {"type": "number", "description": "Length in beats."},
                "gain": {"type": "number"},
            },
            "required": ["midi", "start", "length"],
        }
        part = {
            "type": "object",
            "properties": {
                "part": {"type": "string", "description": "drums | bass | chords | lead | pad"},
                "gain": {"type": "number", "default": 1.0},
                "wave": {"type": "string", "enum": ["sine", "triangle", "square", "saw"]},
                "cutoff": {"type": "number", "description": "0–1 lowpass, for saw parts."},
                "attack": {"type": "number"},
                "release": {"type": "number"},
                "vibrato": {"type": "number"},
                "pan": {"type": "number", "description": "-1 hard left … +1 hard right (default 0)."},
                "reverb": {"type": "number", "description": "0–1 wet amount (default 0.12)."},
                "reverb_decay": {"type": "number", "description": "Tail length in seconds."},
                "delay": {"type": "number", "description": "0–1 delay mix."},
                "delay_ms": {"type": "number", "default": 375},
                "delay_feedback": {"type": "number", "default": 0.28},
                "swing": {"type": "number", "description": "0–1; pushes the offbeat sixteenths later (drums)."},
                "humanize_ms": {"type": "number", "description": "Random timing spread per hit, in ms (drums)."},
                "repeat": {
                    "type": ["boolean", "number"],
                    "description": "Tile the written notes until the part is full (true) or that many times (a number).",
                },
                "notes": {"type": "array", "items": note, "description": "For pitched parts."},
                "pattern": {
                    "type": "object",
                    "description": "For drums: 16-step rows such as {\"kick\": \"x...x...x...x...\"}.",
                    "additionalProperties": {"type": "string"},
                },
                "steps": {"type": "integer", "default": 16},
                "segments": {
                    "type": "array",
                    "description": (
                        "Arrange the part over time: each entry is a stretch of bars rendered with the "
                        "part's settings as defaults and its own keys overriding them, then concatenated. "
                        "`{\"bars\": 8, \"gain\": 0}` (or silent: true) leaves that stretch empty."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "bars": {"type": "number", "description": "Length of this stretch, in bars."},
                            "silent": {"type": "boolean", "description": "Leave this stretch empty."},
                            "gain": {"type": "number"},
                            "pattern": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "notes": {"type": "array", "items": note},
                            "wave": {"type": "string", "enum": ["sine", "triangle", "square", "saw"]},
                            "cutoff": {"type": "number"},
                            "pan": {"type": "number"},
                            "reverb": {"type": "number"},
                            "delay": {"type": "number"},
                            "swing": {"type": "number"},
                            "humanize_ms": {"type": "number"},
                            "repeat": {"type": ["boolean", "number"]},
                        },
                        "required": ["bars"],
                    },
                },
                "out_path": {"type": "string"},
            },
            "required": ["part"],
        }
        if name == "render_part":
            return {
                "type": "object",
                "properties": {
                    "out_path": {"type": "string", "description": "WAV file to write."},
                    "bpm": {"type": "number", "default": 120},
                    "bars": {"type": "number", "default": 4},
                    **part["properties"],
                },
                "required": ["out_path", "part"],
            }
        return {
            "type": "object",
            "properties": {
                "out_path": {"type": "string", "description": "Mixed WAV to write; stems land beside it."},
                "midi_path": {
                    "type": "string",
                    "description": (
                        "Optional .mid to write as well: the same arrangement as notes (one track per part, "
                        "drums on MIDI's drum channel), notated from the plan rather than transcribed."
                    ),
                },
                "bpm": {"type": "number", "default": 120},
                "bars": {"type": "number", "default": 4},
                "sections": {
                    "type": "array",
                    "description": (
                        "The song's shape, in order (`intro` 8, `drop` 16, …). It names the timeline and, "
                        "when `bars` is not given, its lengths add up to the song's length."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "bars": {"type": "number"},
                        },
                        "required": ["name", "bars"],
                    },
                },
                "parts": {"type": "array", "items": part, "minItems": 1},
            },
            "required": ["out_path", "parts"],
        }
    if name == "run_job":
        return {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "job_dir": {"type": "string"},
                "factory": {"type": "string", "description": "Override SHADOW_PIPELINE_FACTORY for this run."},
            },
            "required": ["job_id"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {"job_id": {"type": "string"}, "job_dir": {"type": "string"}},
        "required": ["job_id"],
        "additionalProperties": False,
    }


def response(request_id: object, result: object = None, error: object = None) -> dict:
    value = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        value["error"] = {"code": -32000, "message": str(error)}
    else:
        value["result"] = result
    return value


def handle(message: dict) -> "dict | None":
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        return response(
            request_id,
            {
                "protocolVersion": requested if isinstance(requested, str) and requested else PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "ping":
        return response(request_id, {})
    if method in EMPTY_RESULTS:
        return response(request_id, EMPTY_RESULTS[method])
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if method == "tools/list":
        return response(
            request_id,
            {"tools": [{"name": name, "description": description, "inputSchema": _schema(name)} for name, description in TOOLS.items()]},
        )
    if method != "tools/call":
        return response(request_id, error=f"Unsupported method: {method}")
    name = params.get("name")
    arguments = params.get("arguments") or {}
    try:
        payload = _run(name, arguments)
        return response(request_id, {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]})
    except Exception as exc:
        # Tool failures are reported inside the result (MCP `isError`), not as protocol errors.
        return response(request_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})


def _run(name: str, arguments: dict) -> dict:
    store = JobStore(arguments.get("job_dir"))
    if name == "submit_generation":
        job = store.submit(
            GenerationRequest(
                prompt=str(arguments.get("prompt") or ""),
                mode=str(arguments.get("mode") or "dry-run"),
                model=arguments.get("model"),
                output_dir=arguments.get("output_dir"),
                lyrics=arguments.get("lyrics"),
                source_audio=arguments.get("source_audio"),
            )
        )
        if job.request.mode == "dry-run" or arguments.get("run"):
            job = store.run(job.id)
        return job_payload(job)
    if name == "run_job":
        return job_payload(store.run(str(arguments["job_id"]), arguments.get("factory")))
    if name == "job_status":
        return job_payload(store.get(str(arguments["job_id"])))
    if name == "render_part":
        return render_part(arguments)
    if name == "render_song":
        return render_song(arguments)
    if name == "mix_arrangement":
        return mix_arrangement(arguments)
    if name == "export_stems":
        return export_stems(arguments)
    if name == "list_export_formats":
        return {"containers": container_capabilities()}
    if name == "transform_notes":
        return transform_notes(arguments)
    if name == "regenerate_part":
        return regenerate_part(arguments)
    if name == "arrange":
        return arrange(arguments)
    if name == "export_midi":
        return export_midi(arguments)
    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            result = handle(json.loads(line))
            if result is not None:
                print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps(response(None, error=exc), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
