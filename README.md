# Shadow Music Generator

An MCP server that queues and runs **music generation jobs** (YuE and compatible pipelines) for
Shadow Producers — without downloading checkpoints or starting a model unless you explicitly ask
for it.

[中文说明](README.zh-CN.md) · License: [AGPL-3.0](LICENSE)

## Features

- **Dry run by default.** Submitting a job validates the request, records the model license and
  writes a job file; nothing else happens until you run it.
- **Job queue with history.** Every job is a JSON file with its status, pipeline stages, timings,
  outputs and errors.
- **Bring your own pipeline.** Local execution happens through an adapter you point at with
  `SHADOW_PIPELINE_FACTORY`; the plugin just sequences the stages.
- **License aware.** The model license is recorded in every job result, and no weights are
  downloaded or bundled.

## Requirements

| | |
| --- | --- |
| OS | macOS, Linux or Windows |
| Python | 3.9 or newer |
| Runtime deps | none (your adapter brings its own model stack) |

## Install

### As a Codex plugin

```sh
codex plugin marketplace add shadowroommusic/shadow-music-generator
codex plugin add shadow-music-generator@shadowroom
```

### In any other MCP client

```json
{
  "mcpServers": {
    "shadow-music-generator": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/shadow-music-generator"
    }
  }
}
```

### CLI only

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/shadow-music-generator --help
```

## Configuration

| Option | Default | Used for |
| --- | --- | --- |
| `SHADOW_JOB_DIR` | `./.shadow-jobs` | where job files are stored |
| `--job-dir` | same as above | per-command override of the job directory |
| `SHADOW_PIPELINE_FACTORY` | – | `module:callable` or `/path/file.py:callable` adapter for local runs |
| `--mode dry-run \| local` | `dry-run` | validate only, or queue for local execution |

## Tools

| Tool | What it does |
| --- | --- |
| `submit_generation` | Queue a job (`prompt`, `mode`, `model`, `lyrics`, `source_audio`, `output_dir`, `job_dir`, `run`) |
| `run_job` | Run a queued job, optionally with a `factory` override |
| `job_status` | Read status, stages, outputs and errors for one job |

CLI equivalents: `shadow-music-generator submit`, `run`, `status`, `list`.

## Usage

```sh
# dry run: validate the request and write a job file
.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno, 128 bpm'

# queue for local execution, then run it
.venv/bin/shadow-music-generator submit --prompt 'techno' --mode local
.venv/bin/shadow-music-generator run --job JOB_ID

# check on it
.venv/bin/shadow-music-generator status --job JOB_ID
.venv/bin/shadow-music-generator list
```

Running locally needs an adapter, for example:

```sh
export SHADOW_PIPELINE_FACTORY=/path/to/my_yue_adapter.py:make_pipeline
.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno' --mode local --run
```

See [docs/internals.md](docs/internals.md) for the adapter contract (stages, context, outputs).

## Model licensing

This plugin ships **no model weights**. YuE's code is Apache-2.0, while current YuE2 checkpoint
weights are licensed **CC BY-NC 4.0** — do not use them in a commercial product without a separate
license. Every job result records the license it ran under.

## Safety

- Nothing is downloaded and no model is started unless you pass `--mode local` **and** configure an
  adapter.
- Jobs only write inside `output_dir` and `job_dir`; failures keep the stages that succeeded.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `status: failed`, stage named | Read `stages`/`error` in the job file — the failing stage is reported explicitly. |
| Local run refuses to start | Set `SHADOW_PIPELINE_FACTORY`; without it local mode has nothing to run. |
| Adapter not importable | Use an absolute `/path/file.py:callable` spec, or make sure the module is on `PYTHONPATH`. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Implementation notes live in
[docs/internals.md](docs/internals.md).

## License

AGPL-3.0 — see [LICENSE](LICENSE). Upstream YuE code and model weights keep their own licenses.
