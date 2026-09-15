"""Reshaping one rendered part, without touching the rest of the song.

When a user says "make the drums heavier" they mean the drums — every other track must come back
byte-identical. That is only possible if the *plan* for that part was kept: the workbench stores each
part's render spec on its clip, and this module applies a small, closed vocabulary of changes to it.

The vocabulary is deliberately finite (`gain_db`, `cutoff`, `swing`, `humanize_ms`, `pan`, `reverb`,
`delay`, `wave`, `pattern`, `remove_voices`, `transpose`, `velocity_scale`) because a model inventing
parameter names produces edits nobody can reproduce. Every entry returns a Chinese sentence describing
what it did — that sentence is what the user reads, and what the undo stack is labelled with.
"""
from __future__ import annotations

import copy

__all__ = ["apply_part_patch", "REGENERATE_KEYS"]

#: Every key a patch may use. Anything else is refused, loudly.
REGENERATE_KEYS = (
    "gain_db",
    "cutoff",
    "swing",
    "humanize_ms",
    "pan",
    "reverb",
    "delay",
    "delay_ms",
    "wave",
    "pattern",
    "remove_voices",
    "transpose",
    "velocity_scale",
)

WAVES = ("sine", "triangle", "square", "saw")


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_part_patch(part: dict, patch: dict) -> "tuple[dict, list[str]]":
    """Apply a patch to one part spec.

    @param part: the part as it was rendered (`{part, pattern|notes, wave, cutoff, …}`).
    @param patch: only keys from `REGENERATE_KEYS`.
    @returns the new spec and one human-readable line per change.
    """
    unknown = [key for key in patch if key not in REGENERATE_KEYS]
    if unknown:
        raise ValueError(f"unknown patch keys: {', '.join(sorted(unknown))}")
    if not patch:
        raise ValueError("regenerate_part needs at least one change")
    new = copy.deepcopy(part)
    notes: "list[str]" = []

    if "gain_db" in patch:
        factor = 10 ** (float(patch["gain_db"]) / 20.0)
        new["gain"] = round(float(new.get("gain", 1.0)) * factor, 4)
        notes.append(f"音量 {float(patch['gain_db']):+.1f} dB")

    if "cutoff" in patch:
        # the synth's cutoff is a 0–1 lowpass ratio; keep it in that unit rather than pretending
        # to know a frequency, and clamp to something audible.
        new["cutoff"] = round(_clip(float(new.get("cutoff", 0.55)) + float(patch["cutoff"]), 0.05, 0.95), 3)
        notes.append(f"亮度 {float(patch['cutoff']):+.2f}")

    if "swing" in patch:
        new["swing"] = round(_clip(float(new.get("swing", 0.0)) + float(patch["swing"]), 0.0, 0.4), 3)
        notes.append(f"摇摆 {float(patch['swing']):+.2f}")

    if "humanize_ms" in patch:
        new["humanize_ms"] = round(
            _clip(float(new.get("humanize_ms", 0.0)) + float(patch["humanize_ms"]), 0.0, 40.0), 1
        )
        notes.append(f"人性化 {float(patch['humanize_ms']):+.1f} ms")

    if "pan" in patch:
        pan = _clip(float(patch["pan"]), -1.0, 1.0)
        new["pan"] = round(pan, 3)
        notes.append("声像居中" if abs(pan) < 0.01 else "声像 {:+.2f}".format(pan))

    if "reverb" in patch:
        new["reverb"] = round(_clip(float(new.get("reverb", 0.12)) + float(patch["reverb"]), 0.0, 0.6), 3)
        notes.append(f"混响 {float(patch['reverb']):+.2f}")

    if "delay" in patch:
        new["delay"] = round(_clip(float(new.get("delay", 0.0)) + float(patch["delay"]), 0.0, 0.5), 3)
        notes.append(f"延迟 {float(patch['delay']):+.2f}")

    if "delay_ms" in patch:
        new["delay_ms"] = round(_clip(float(patch["delay_ms"]), 20.0, 1500.0), 1)
        notes.append(f"延迟时间 {float(patch['delay_ms']):.0f} ms")

    if "wave" in patch:
        wave = str(patch["wave"])
        if wave not in WAVES:
            raise ValueError(f"unknown wave {wave!r}; use one of {', '.join(WAVES)}")
        new["wave"] = wave
        notes.append(f"波形换成 {wave}")

    if "pattern" in patch:
        if str(new.get("part")) != "drums" and "pattern" not in new:
            raise ValueError("only a drums part has a pattern")
        merged = dict(new.get("pattern") or {})
        for voice, row in dict(patch["pattern"]).items():
            if not isinstance(row, str):
                raise ValueError(f"pattern row for {voice} must be a string of steps")
            merged[str(voice)] = row
        new["pattern"] = merged
        notes.append(f"鼓组换成 {', '.join(sorted(dict(patch['pattern'])))}")

    if "remove_voices" in patch:
        merged = dict(new.get("pattern") or {})
        for voice in list(patch["remove_voices"]):
            merged.pop(str(voice), None)
        new["pattern"] = merged
        notes.append(f"去掉 {', '.join(str(voice) for voice in patch['remove_voices'])}")

    if "transpose" in patch:
        semitones = int(patch["transpose"])
        written = new.get("notes")
        if not written:
            raise ValueError("only a pitched part can be transposed")
        new["notes"] = [
            {**note, "midi": max(0, min(127, int(note.get("midi", 60)) + semitones))} for note in written
        ]
        notes.append(f"移调 {semitones:+d} 半音")

    if "velocity_scale" in patch:
        scale = _clip(float(patch["velocity_scale"]), 0.1, 3.0)
        written = new.get("notes")
        if not written:
            raise ValueError("only a pitched part has note velocities")
        new["notes"] = [{**note, "gain": round(float(note.get("gain", 1.0)) * scale, 4)} for note in written]
        notes.append(f"力度 ×{scale:g}")

    return new, notes
