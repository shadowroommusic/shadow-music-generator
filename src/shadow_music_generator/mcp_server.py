from __future__ import annotations

import json
import sys

from .cli import job_payload
from .jobs import GenerationRequest, JobStore

TOOLS = {
    "submit_generation": "Queue a Shadow Music Generator job. Dry-run validates the request and never runs a model.",
    "run_job": "Run a queued job through the configured SHADOW_PIPELINE_FACTORY adapter.",
    "job_status": "Read one job: status, stages, outputs, model license and errors.",
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
