# Internals

Maintainer notes for `shadow-music-generator`. The user-facing docs live in
[../README.md](../README.md).

## Job files

Every job is a JSON file under `./.shadow-jobs` (or `SHADOW_JOB_DIR`, or `--job-dir`) containing:

- `status`, `model_license`, `prompt`/`lyrics`/`model`/`source_audio`, `output_dir`
- `pipeline` (the adapter that ran)
- `stages`: one entry per stage with its duration
- `outputs`: the artifacts that were produced
- `error`: present when a stage failed

## Local mode: the pipeline factory

Local execution only happens when `SHADOW_PIPELINE_FACTORY=<module or /path/file.py>:<callable>` is
configured. The factory is called with no arguments and must return an object with four callable
stages, executed in this order:

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

A stage may return a mapping, which is merged back into the context for the next stage. The request
keys (`prompt`, `lyrics`, `model`, `source_audio`, `output_dir`) cannot be overwritten. Anything in
`context["artifacts"]` (or returned by `decode`) becomes the job's `outputs`.

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

A missing stage, an unimportable module, an invalid spec or a stage exception is reported as
`status: "failed"` with the failing stage name; the job keeps the stages that did succeed. Weight
loading, GPU selection and licensing stay entirely inside the adapter — this plugin only sequences
the calls.

## Tests

The suite builds its own adapter modules in a temp directory, so it exercises the real import path,
stage sequence and failure reporting without any model weights:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
