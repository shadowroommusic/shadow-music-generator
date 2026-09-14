# Shadow Music Generator

An MCP server that queues and runs **music generation jobs** (YuE and compatible pipelines) for any
MCP-compatible agent — without downloading checkpoints or starting a model unless you explicitly
ask for it.

面向任何支持 MCP 的 agent 的**音乐生成任务队列**（YuE 及兼容流程）：默认 dry-run，不下载权重、
不启动模型，除非你明确要求。

The plugin itself is lightweight by design — no models, no ML dependencies. The heavy generation
runs wherever your adapter points it: a local GPU machine, a rented box over ssh, or a cloud API.

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
- **A local synth you can actually arrange with.** `render_song` writes a mix and one stem per part
  from a plain JSON plan: step patterns and notes, sections (`intro` / `drop` / …), per-section
  segments, and — when asked — the same arrangement as a Type-1 MIDI file, notated from the plan
  rather than transcribed from the audio.

## Current scope

This plugin is the **generation scheduling layer**: a lightweight job queue that records prompts, lyrics,
parameters, stage timings, artifacts and license notes, and drives whatever backend you point it at.
It ships no model and no ML dependency.

Out of the box it is used in **dry-run / records-only mode** — submit jobs to keep a searchable
history of prompts and settings. Wiring a real provider (local GPU, remote box, or a cloud API) is
planned to happen inside the `producer-tools` app. The YuE2 adapter further down is already
implemented and tested for whenever you want to point it at a machine that can generate.

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
| `render_part` | Render one part (drums / bass / chords / lead / pad) to a WAV with the local synth |
| `render_song` | Render a whole arrangement: mix + one stem per part, optionally its MIDI (`midi_path`) |
| `mix_arrangement` | Bounce an arrangement of clips to one file (WAV/AIFF, 44.1/48/96 kHz, 16/24-bit) |
| `export_stems` | Bounce every track of an arrangement to its own file |
| `export_midi` | Write an arrangement of notated clips (or `.mid` clips) as a Type-1 MIDI file |

CLI equivalents: `shadow-music-generator submit`, `run`, `status`, `list`.

### The local synth

`render_song` is what a sentence like *"a 32-bar tech house: intro 8, drop 16, outro 8"* turns into
audio. The plan is small on purpose:

```json
{
  "out_path": "~/Music/drive.wav",
  "midi_path": "~/Music/drive.mid",
  "bpm": 126,
  "sections": [{"name": "intro", "bars": 8}, {"name": "drop", "bars": 16}, {"name": "outro", "bars": 8}],
  "parts": [
    {
      "part": "drums", "swing": 0.14, "humanize_ms": 16,
      "segments": [
        {"bars": 8,  "pattern": {"hat": "..x...x...x...x."}},
        {"bars": 16, "pattern": {"kick": "x...x...x...x...", "clap": "....x.......x...", "hat": "..x...x...x...x."}},
        {"bars": 8,  "pattern": {"kick": "x...x...x...x...", "hat": "..x...x...x...x."}}
      ]
    },
    {
      "part": "bass", "wave": "saw", "cutoff": 0.32, "repeat": true,
      "segments": [
        {"bars": 8,  "gain": 0},
        {"bars": 16, "notes": [{"midi": 33, "start": 0, "length": 0.75}]},
        {"bars": 8,  "gain": 0}
      ]
    }
  ]
}
```

- **Segments** arrange a part over time: each entry is a stretch of bars rendered with the part's
  settings as defaults and its own keys overriding them, then concatenated. `{"bars": 8, "gain": 0}`
  (or `silent: true`) leaves the stretch empty — that is how a breakdown loses its drums.
- **Sections** name the timeline and, when `bars` is not given, their lengths add up to the song.
  They come back in the report, so a host app can draw the song's shape.
- **Drums tile.** A 16-step row is one bar and repeats across the part; a 32-step row is a two-bar
  block. Pitched parts repeat their written notes only when `repeat` asks for it.
- **`midi_path`** writes the arrangement as notes as well: one MIDI track per part, drums on MIDI's
  drum channel (10), and swing applied — swing is part of the plan, humanised timing is not (that
  belongs to the performance).

Every stem is exactly as long as the song, so the stems, the mix and the MIDI line up.

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

### YuE / YuE2 adapter

The official [YuE](https://github.com/multimodal-art-projection/YuE) pipeline plugs in with one line:

```sh
export SHADOW_PIPELINE_FACTORY=shadow_music_generator.adapters.yue2_adapter:make_pipeline
.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno, 128 bpm' --mode local --run
```

Two ways to run it:

- **Inside a YuE environment** (default): the adapter imports `yue2` and calls YuE2's own staged API
  (`plan` → `generate_semantic` → `synthesize` → `decode`), so every stage shows up in the job report.
- **Anywhere else** (GPU box, container, remote host): set `YUE_COMMAND` to a command template, e.g.
  `ssh gpu 'yue2 generate --request {request_json} --output {output_dir}'`.

Relevant environment variables: `YUE_MODEL` (default `m-a-p/YuE2-3B`), `YUE_DEVICE`, `YUE_VAE`,
`YUE_COT` (`full`/`melody`/`off`), `YUE_SEED`, `YUE_LYRICS`, `YUE_ABC`, `YUE_REQUEST_JSON`,
`YUE_COMMAND`, `YUE_OUTPUTS`.

The queue never post-processes audio: in command mode the files YuE writes are exactly the files you
get, and in pipeline mode the decoded samples are written out as-is at 48 kHz.

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
