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

## YuE2 adapter (`shadow_music_generator.adapters.yue2_adapter`)

YuE2's staged Python API is `plan() → generate_semantic() → synthesize() → decode()`, so the adapter
maps one contract stage onto one YuE2 stage:

| Contract stage | Pipeline mode (inside the YuE environment) | Command mode (`YUE_COMMAND`) |
| --- | --- | --- |
| `plan` | `YuE2Pipeline.from_pretrained(model, device, vae)` + `pipe.plan(**request)`, saved to `<output_dir>/plan` | writes `<output_dir>/request.json`, validates the template |
| `generate_semantic` | `pipe.generate_semantic(plan)` | runs the command (placeholders substituted, no shell) |
| `synthesize` | `pipe.synthesize(semantic)` | requires at least one audio file to exist |
| `decode` | `pipe.decode(latents)` → `audio.flac` (48 kHz via `soundfile`, stdlib WAV fallback) | collects the produced audio files, untouched |

Requests are built from the job (`prompt` → YuE `style`, `lyrics`) plus `YUE_REQUEST_JSON` as a base,
then `YUE_COT`, `YUE_SEED`, `YUE_ABC`, `YUE_MODEL`. The adapter never decodes, re-encodes or
normalises audio; a failure is reported per stage and keeps the earlier stages in the job record.

### Proving the queue changes nothing

1. Run a request directly in the YuE environment:
   `yue2 generate --request request.json --output out-direct`
2. Run the same request through the plugin:
   `SHADOW_PIPELINE_FACTORY=shadow_music_generator.adapters.yue2_adapter:make_pipeline`
   with the same seed (and, for a strict comparison, the same model/VAE revisions and device).
3. Compare: `shasum -a 256 out-direct/audio.flac <job output>/audio.flac`.

Identical hashes mean the queue added nothing to the audio; a difference points at the command
(GPU, runtime version, sampling settings) rather than at this plugin.

## Tests

The suite builds its own adapter modules in a temp directory, so it exercises the real import path,
stage sequence and failure reporting without any model weights:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
