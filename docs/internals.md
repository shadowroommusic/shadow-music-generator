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

## The local synth (`synth.py`)

`render_part` / `render_song` / `mix_arrangement` / `export_stems` / `export_midi` are the second
half of this plugin: the queue records what to generate, the synth generates. Everything is numpy —
`_oscillator`, `_lowpass`, `_envelope`, `_kick` / `_snare` / `_clap` / `_hat` — so the plugin keeps
its "no ML dependency" property. Parts are rendered mono, panned (equal power), delayed and reverbed
per channel, and written as stereo WAV (16/24-bit) or AIFF.

Three things about the arrangement model are worth keeping in mind when editing it:

1. **A part is a chain of segments.** `_render_part_arrangement` renders each `segments` entry with
   the part's settings as defaults and the entry's keys on top, then concatenates; `{"gain": 0}` or
   `silent: true` renders silence. `_fit` then makes every part exactly the song's length, which is
   what keeps the stems, the mix and the MIDI on one timeline. A part without `segments` renders one
   block of `bars` — the loop case, byte for byte what it always was.
2. **Drums tile; notes repeat only when asked.** `_render_drums` repeats a step row across the part
   (a 16-step row is one bar, a 32-step row is a two-bar block). This was a real bug until
   2026-09-15: rows were placed once, so a 4-bar beat was one bar of audio followed by three bars of
   silence — measured on the workbench's own stems (`night-drive-drums.wav`, 8 bars: bar 1 = 0.108
   RMS, bars 2–8 = 0.000). Pitched parts keep the written positions unless `repeat` says otherwise.
3. **The MIDI is the plan, not a transcription.** `song_midi_parts` reads the same spec through
   `_stretch_notes` — segments, repeats, sections — and writes it with `write_midi_multitrack`, drums
   on channel 10 (index 9). Swing is applied (it is part of the plan); `humanize_ms` is not, because
   it is random per hit and belongs to the performance.

## Tests

The suite builds its own adapter modules in a temp directory, so it exercises the real import path,
stage sequence and failure reporting without any model weights; the synth tests render into temp
directories and read the files back (levels per bar, sections, the MIDI the render hands out).

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
