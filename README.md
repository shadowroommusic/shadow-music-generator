# Shadow Music Generator

This plugin queues isolated YuE generation jobs for Shadow Producers. Installing it never downloads a checkpoint and never starts a model: the default `dry-run` mode validates the request, records the model license and writes a job file, then stops.

YuE's current repository is [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE). The code is Apache-2.0, while current YuE2 checkpoint weights are separately licensed CC BY-NC 4.0. The worker therefore requires an explicit model path or provider configuration and reports the model license in every job result. Do not bundle YuE2 weights into a commercial product without a separate license.

## Commands

```sh
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .

.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno, 128 bpm'          # dry-run, runs immediately
.venv/bin/shadow-music-generator submit --prompt 'techno' --mode local                   # queued, no model started
.venv/bin/shadow-music-generator run --job JOB_ID                                        # run a queued job
.venv/bin/shadow-music-generator status --job JOB_ID
.venv/bin/shadow-music-generator list
```

Every job lives as a JSON file under `./.shadow-jobs` (or `SHADOW_JOB_DIR`, or `--job-dir`). A job records `status`, `model_license`, the pipeline that ran, one `stages` entry per pipeline stage with its duration, the `outputs` it produced and any `error`.

## Local mode: `SHADOW_PIPELINE_FACTORY`

Local execution only happens when a pipeline adapter is configured explicitly:

```sh
export SHADOW_PIPELINE_FACTORY=/path/to/my_yue_adapter.py:make_pipeline
# or: export SHADOW_PIPELINE_FACTORY=my_module:make_pipeline
.venv/bin/shadow-music-generator submit --prompt 'dark melodic techno' --mode local --run
```

The factory is called with no arguments and must return an object with four callable stages, executed in this order:

| Stage | Purpose |
| --- | --- |
| `plan` | Turn the prompt into a structure (sections, bars, tags). |
| `generate_semantic` | Produce semantic tokens / the model's intermediate representation. |
| `synthesize` | Render the vocoder or audio stage. |
| `decode` | Write the final file(s) and register them in `context["artifacts"]`. |

Every stage receives the same mutable `context` dict:

```python
{
    "prompt": str, "lyrics": str | None, "model": str | None,
    "source_audio": str | None, "output_dir": str,
    "artifacts": [ ... ],        # fill this with output paths
    # plus whatever earlier stages merged in
}
```

A stage may return a mapping, which is merged back into the context for the next stage. The request keys (`prompt`, `lyrics`, `model`, `source_audio`, `output_dir`) cannot be overwritten. Anything listed in `context["artifacts"]` (or returned by `decode`) becomes the job's `outputs`.

```python
import os


class MyPipeline:
    def plan(self, context):
        return {"plan": {"sections": [{"prompt": context["prompt"], "bars": 16}]}}

    def generate_semantic(self, context):
        return {"tokens": self.model.encode(context["plan"])}

    def synthesize(self, context):
        return {"vocoder_output": "vocoder.wav"}

    def decode(self, context):
        path = os.path.join(context["output_dir"], "take-1.wav")
        with open(path, "wb") as handle:
            handle.write(b"RIFF...")
        context["artifacts"].append(path)
        return {"artifacts": list(context["artifacts"])}


def make_pipeline():
    return MyPipeline()
```

A missing stage, an unimportable module, an invalid spec or a stage exception is reported as `status: "failed"` with the failing stage name; the job keeps the stages that did succeed. Weight loading, GPU selection and licensing stay entirely inside your adapter — this plugin only sequences the call.

## MCP

`.mcp.json` exposes three tools:

- `submit_generation` — queue a job (`prompt`, `mode`, `model`, `lyrics`, `source_audio`, `output_dir`, `job_dir`, `run`).
- `run_job` — run a queued job, optionally with a `factory` override.
- `job_status` — read status, stages, outputs and errors for one job.

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests build their own adapter modules in a temp directory, so they exercise the real import path, the real stage sequence and the real failure reporting without any model weights.

## License

MIT for this plugin. Model weights and the upstream YuE code keep their own licenses.
