"""Deterministic note transforms — the arithmetic an agent must not do in its head.

An agent can say "raise that melody two semitones" or "keep it in A minor", but the actual pitch
numbers, snapping and velocity changes are computed here: same input, same output, every time. The
host app applies the result to its own state and owns undo, so this module is pure — it never looks at
a project, only at the notes it is handed.

Notes are `{id, pitch, start_ms, end_ms, velocity}` — the shape the workbench's piano roll stores.
Every response echoes the `clip_id` and `revision` it was given, so the caller can refuse a result
that arrived after the user moved something.
"""
from __future__ import annotations

import random

__all__ = ["transform_notes", "SCALES", "scale_pitches"]

#: Scale degrees in semitones from the root, for the modes an electronic producer reaches for.
SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
}

#: Note names to pitch class; both spellings are accepted so `Bb` and `A#` behave the same.
NOTE_TO_PITCH_CLASS = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
    "Bb": 10, "B": 11, "Cb": 11,
}


def scale_pitches(root: str, scale: str) -> "set[int]":
    """Every pitch class that belongs to one key."""
    if root not in NOTE_TO_PITCH_CLASS:
        raise ValueError(f"unknown root note: {root}")
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; try one of {', '.join(sorted(SCALES))}")
    tonic = NOTE_TO_PITCH_CLASS[root]
    return {(tonic + degree) % 12 for degree in SCALES[scale]}


def _clamp_pitch(value: int) -> int:
    return max(0, min(127, int(value)))


def _nearest_in_scale(pitch: int, allowed: "set[int]") -> int:
    """Closest scale tone; a tie goes up, the way a player would lean."""
    for distance in range(0, 7):
        for candidate in (pitch + distance, pitch - distance):
            if 0 <= candidate <= 127 and candidate % 12 in allowed:
                return candidate
    return pitch


def _apply(notes: "list[dict]", op: dict, rng: random.Random) -> "tuple[list[dict], list[str], set[str]]":
    """One operation over the whole note list: `(notes, descriptions, touched ids)`."""
    kind = str(op.get("op") or op.get("operation") or "").strip().lower()
    targets = op.get("note_ids")
    touched: "set[str]" = set()

    def selected() -> "list[dict]":
        return [note for note in notes if targets is None or note.get("id") in targets]

    if kind in ("transpose", "transpose_notes"):
        semitones = int(op.get("semitones", 0))
        for note in selected():
            note["pitch"] = _clamp_pitch(int(note.get("pitch", 60)) + semitones)
            touched.add(str(note.get("id")))
        direction = "升高" if semitones >= 0 else "降低"
        return notes, [f"{direction} {abs(semitones)} 个半音"], touched

    if kind in ("quantise", "quantize", "quantise_notes"):
        grid = max(1.0, float(op.get("grid_ms", 125)))
        strength = max(0.0, min(1.0, float(op.get("strength", 1.0))))
        for note in selected():
            for key in ("start_ms", "end_ms"):
                original = float(note.get(key, 0))
                snapped = round(original / grid) * grid
                note[key] = int(round(original * (1 - strength) + snapped * strength))
            if note["end_ms"] <= note["start_ms"]:
                note["end_ms"] = int(note["start_ms"] + grid)
            touched.add(str(note.get("id")))
        label = "量化到网格" if strength >= 0.999 else f"向网格靠 {int(strength * 100)}%"
        return notes, [f"{label}（{grid:g} ms）"], touched

    if kind in ("set_scale", "scale", "scale_lock"):
        allowed = scale_pitches(str(op.get("root", "C")), str(op.get("scale", "minor")))
        for note in selected():
            note["pitch"] = _nearest_in_scale(int(note.get("pitch", 60)), allowed)
            touched.add(str(note.get("id")))
        return notes, [f"音阶约束到 {op.get('root', 'C')} {op.get('scale', 'minor')}"], touched

    if kind in ("humanise", "humanize"):
        timing = float(op.get("timing_ms", 0.0))
        velocity = float(op.get("velocity", 0.0))
        for note in selected():
            jitter = rng.uniform(-timing, timing) if timing > 0 else 0.0
            note["start_ms"] = max(0, int(round(float(note.get("start_ms", 0)) + jitter)))
            note["end_ms"] = max(note["start_ms"] + 1, int(round(float(note.get("end_ms", 0)) + jitter)))
            if velocity > 0:
                note["velocity"] = max(1, min(127, int(round(float(note.get("velocity", 100)) + rng.uniform(-velocity, velocity)))))
            touched.add(str(note.get("id")))
        return notes, [f"人性化（时间 ±{timing:g} ms，力度 ±{velocity:g}）"], touched

    if kind in ("shift", "shift_time", "move"):
        delta = float(op.get("ms", 0))
        for note in selected():
            start = max(0, float(note.get("start_ms", 0)) + delta)
            length = float(note.get("end_ms", 0)) - float(note.get("start_ms", 0))
            note["start_ms"] = int(round(start))
            note["end_ms"] = int(round(max(start + 1, start + length)))
            touched.add(str(note.get("id")))
        return notes, [f"{'后移' if delta >= 0 else '前移'} {abs(delta):g} ms"], touched

    if kind in ("velocity", "set_velocity"):
        add = float(op.get("add", 0))
        scale = float(op.get("scale", 1.0))
        for note in selected():
            note["velocity"] = max(1, min(127, int(round(float(note.get("velocity", 100)) * scale + add))))
            touched.add(str(note.get("id")))
        return notes, [f"力度 ×{scale:g}{f' {add:+g}' if add else ''}"], touched

    raise ValueError(f"unknown note operation: {kind or '(missing)'}")


def transform_notes(spec: dict) -> dict:
    """Apply a list of operations to a note list and describe what changed.

    `spec` is `{clip_id?, revision?, notes: [...], operations: [{op, ...}], seed?}`. The ids, the
    revision and any unknown keys on a note are passed through untouched, so the caller can apply the
    result directly and keep its own bookkeeping.
    """
    notes = [dict(note) for note in (spec.get("notes") or [])]
    operations = spec.get("operations") or []
    if not isinstance(operations, list) or not operations:
        raise ValueError("transform_notes needs at least one operation")
    rng = random.Random(int(spec.get("seed", 7)))
    descriptions: "list[str]" = []
    affected: "set[str]" = set()
    for op in operations:
        notes, described, touched = _apply(notes, dict(op), rng)
        descriptions.extend(described)
        affected |= touched
    notes.sort(key=lambda note: (float(note.get("start_ms", 0)), int(note.get("pitch", 60))))
    return {
        "schema_version": 1,
        "mode": "transform-notes",
        "clip_id": spec.get("clip_id"),
        "revision": spec.get("revision"),
        "notes": notes,
        "note_count": len(notes),
        "affected": sorted(affected),
        "affected_count": len(affected),
        "diff_summary": "；".join(descriptions) if descriptions else "没有变化",
    }
