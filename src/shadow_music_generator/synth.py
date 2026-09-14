"""Offline synthesis: turn a musical spec into real audio.

This is the local, model-free renderer behind `render_part` / `render_song`. The agent writes the
music — patterns, notes, tempo — and these functions turn it into 16-bit WAV files that the studio can
play. A model-backed backend (YuE2, a cloud API, …) can be dropped in behind the same interface later.

Everything is numpy + the standard library; no audio packages are required.
"""
from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np

__all__ = ["render_part", "render_song", "write_wav"]

SAMPLE_RATE = 44_100
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def write_wav(path: "str | Path", samples: np.ndarray, rate: int = SAMPLE_RATE) -> Path:
    """Write mono float samples as a 16-bit PCM WAV."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    data = (clipped * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return target


def _seconds_per_beat(bpm: float) -> float:
    return 60.0 / max(1.0, float(bpm))


def _envelope(count: int, rate: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """Simple ADSR over `count` samples (times in seconds)."""
    attack_n = max(1, int(attack * rate))
    decay_n = max(1, int(decay * rate))
    release_n = max(1, int(release * rate))
    sustain_n = max(0, count - attack_n - decay_n - release_n)
    parts = [
        np.linspace(0.0, 1.0, attack_n, endpoint=False),
        np.linspace(1.0, sustain, decay_n, endpoint=False),
        np.full(sustain_n, sustain),
        np.linspace(sustain, 0.0, count - attack_n - decay_n - sustain_n, endpoint=False),
    ]
    envelope = np.concatenate(parts)
    if envelope.size < count:
        envelope = np.pad(envelope, (0, count - envelope.size))
    return envelope[:count]


def _oscillator(wave: str, frequency: np.ndarray, phase0: float = 0.0) -> np.ndarray:
    """Naive but serviceable waveforms from an instantaneous-frequency array."""
    phase = phase0 + np.cumsum(2 * np.pi * frequency / SAMPLE_RATE)
    if wave == "sine":
        return np.sin(phase)
    if wave == "triangle":
        return 2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1
    if wave == "square":
        return np.sign(np.sin(phase))
    # saw: a few harmonics keep it from aliasing too badly
    out = np.zeros_like(phase)
    for harmonic in range(1, 12):
        out += np.sin(phase * harmonic) / harmonic
    return out * (2 / np.pi)


def _lowpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    """One-pole lowpass; `cutoff` is 0–1 (1 = wide open)."""
    alpha = float(np.clip(cutoff, 0.001, 1.0))
    out = np.empty_like(signal)
    previous = 0.0
    for index, value in enumerate(signal):
        previous += alpha * (value - previous)
        out[index] = previous
    return out


def _decay_noise(count: int, rate: int, decay: float, tone: float, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(count) / rate
    noise = rng.standard_normal(count)
    shape = np.exp(-time / max(0.001, decay))
    body = np.sin(2 * np.pi * tone * time) * 0.35
    return (noise * 0.8 + body) * shape


def _kick(rate: int, decay: float = 0.32) -> np.ndarray:
    count = int(rate * decay)
    time = np.arange(count) / rate
    sweep = 48 + 90 * np.exp(-time * 26)
    phase = np.cumsum(2 * np.pi * sweep / rate)
    return np.sin(phase) * np.exp(-time * 7.5)


def _snare(rate: int, rng: np.random.Generator) -> np.ndarray:
    return _decay_noise(int(rate * 0.25), rate, 0.11, 190.0, rng) * 0.7


def _clap(rate: int, rng: np.random.Generator) -> np.ndarray:
    count = int(rate * 0.22)
    out = np.zeros(count)
    for offset in (0.0, 0.012, 0.024):
        start = int(offset * rate)
        piece = _decay_noise(count - start, rate, 0.05, 1000.0, rng) * 0.55
        out[start:] += piece
    return out


def _hat(rate: int, rng: np.random.Generator, open_hat: bool = False) -> np.ndarray:
    count = int(rate * (0.24 if open_hat else 0.06))
    noise = _decay_noise(count, rate, 0.19 if open_hat else 0.035, 8000.0, rng)
    # crude highpass: subtract the running lowpass
    return noise - _lowpass(noise, 0.35)


DRUM_VOICES = {
    "kick": lambda rate, rng: _kick(rate),
    "snare": lambda rate, rng: _snare(rate, rng),
    "clap": lambda rate, rng: _clap(rate, rng),
    "hat": lambda rate, rng: _hat(rate, rng),
    "openhat": lambda rate, rng: _hat(rate, rng, open_hat=True),
}


def _render_drums(spec: dict, beats: float, rate: int, rng: np.random.Generator) -> np.ndarray:
    """One bar of a step pattern, tiled across the part's bars."""
    pattern = spec.get("pattern") or {}
    steps = int(spec.get("steps", 16))
    bars = int(spec.get("bars", 1))
    beat_samples = rate * _seconds_per_beat(spec.get("bpm", 120))
    step_samples = beat_samples / (steps / 4)  # 16 steps = one bar of 4 beats
    total = int(round(beat_samples * beats))
    out = np.zeros(total)
    for voice, row in pattern.items():
        factory = DRUM_VOICES.get(voice)
        if factory is None or not isinstance(row, str):
            continue
        hit = factory(rate, rng) * float(spec.get("gain", 1.0))
        for index, symbol in enumerate(row.replace("|", "")):
            if symbol not in "xXoO":
                continue
            step = index % steps
            position = int(round(step * step_samples))
            length = min(hit.size, total - position)
            if length > 0:
                out[position : position + length] += hit[:length]
    return out


def _render_notes(spec: dict, beats: float, rate: int) -> np.ndarray:
    """Pitched part: notes with midi number, start (beats) and length (beats)."""
    notes = spec.get("notes") or []
    beat_samples = rate * _seconds_per_beat(spec.get("bpm", 120))
    total = int(round(beat_samples * beats))
    out = np.zeros(total)
    wave_kind = str(spec.get("wave", "saw"))
    cutoff = float(spec.get("cutoff", 0.55))
    attack = float(spec.get("attack", 0.01))
    release = float(spec.get("release", 0.08))
    gain = float(spec.get("gain", 0.7))
    vibrato = float(spec.get("vibrato", 0.0))
    for note in notes:
        midi = float(note.get("midi", 60))
        start = int(round(float(note.get("start", 0)) * beat_samples))
        length = int(round(float(note.get("length", 1)) * beat_samples))
        length = max(1, min(length, total - start))
        if length <= 1 or start >= total:
            continue
        time = np.arange(length) / rate
        frequency = np.full(length, midi_to_hz(midi))
        if vibrato > 0:
            frequency = frequency * (1 + vibrato * np.sin(2 * np.pi * 5.0 * time))
        voice = _oscillator(wave_kind, frequency)
        if wave_kind == "saw" and cutoff < 1.0:
            voice = _lowpass(voice, cutoff)
        envelope = _envelope(length, rate, attack, 0.05, 0.8, release)
        out[start : start + length] += voice * envelope * gain * float(note.get("gain", 1.0))
    return out


def renders_part(spec: dict, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render one part (drums, bass, chords, lead, pad) to mono samples."""
    bpm = float(spec.get("bpm", 120))
    beats = float(spec.get("beats") or spec.get("bars", 1) * 4)
    kind = str(spec.get("part", "bass"))
    rng = np.random.default_rng(int(spec.get("seed", 7)))
    if kind == "drums":
        return _render_drums({**spec, "bpm": bpm, "bars": spec.get("bars", 1)}, beats, rate, rng)
    return _render_notes({**spec, "bpm": bpm}, beats, rate)


def _normalise(signal: np.ndarray, peak: float = 0.89) -> np.ndarray:
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum <= 0:
        return signal
    return signal / maximum * peak


def _duration_ms(samples: int, rate: int) -> int:
    return int(round(samples / rate * 1000))


def render_part(spec: dict) -> dict:
    """Render one part to a WAV file and describe what was written."""
    out_path = spec.get("out_path")
    if not out_path:
        raise ValueError("render_part needs an out_path")
    samples = _normalise(renders_part(spec))
    target = write_wav(out_path, samples)
    beats = float(spec.get("beats") or spec.get("bars", 1) * 4)
    return {
        "schema_version": 1,
        "mode": "render-part",
        "part": str(spec.get("part", "bass")),
        "path": str(target),
        "bpm": float(spec.get("bpm", 120)),
        "bars": beats / 4,
        "duration_ms": _duration_ms(samples.size, SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
    }


def render_song(spec: dict) -> dict:
    """Render every part of a song and a mix of all of them.

    `parts` entries carry `part`, the pattern/notes, an optional `gain` and an optional `out_path`
    (defaults to a sibling of the mix, suffixed with the part name).
    """
    out_path = spec.get("out_path")
    if not out_path:
        raise ValueError("render_song needs an out_path")
    song = Path(out_path).expanduser()
    song.parent.mkdir(parents=True, exist_ok=True)
    bpm = float(spec.get("bpm", 120))
    bars = float(spec.get("bars", 4))
    parts = spec.get("parts") or []
    if not parts:
        raise ValueError("render_song needs at least one part")

    rendered: "list[dict]" = []
    mix: np.ndarray | None = None
    for index, part in enumerate(parts):
        kind = str(part.get("part", f"part{index + 1}"))
        payload = {**part, "bpm": bpm, "bars": bars, "seed": part.get("seed", 7 + index)}
        samples = _normalise(renders_part(payload), peak=0.85)
        target = Path(part["out_path"]).expanduser() if part.get("out_path") else song.with_name(f"{song.stem}-{kind}.wav")
        write_wav(target, samples)
        rendered.append(
            {
                "part": kind,
                "path": str(target),
                "duration_ms": _duration_ms(samples.size, SAMPLE_RATE),
                "gain": float(part.get("gain", 1.0)),
            }
        )
        contribution = samples * float(part.get("gain", 1.0))
        mix = contribution if mix is None else _pad_to(mix, contribution.size) + _pad_to(contribution, mix.size)

    assert mix is not None
    mix = np.tanh(mix * 0.9)  # soft clip instead of hard clipping
    mix = _normalise(mix, peak=0.92)
    write_wav(song, mix)
    return {
        "schema_version": 1,
        "mode": "render-song",
        "path": str(song),
        "bpm": bpm,
        "bars": bars,
        "duration_ms": _duration_ms(mix.size, SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
        "parts": rendered,
        "notes": [note_name(int(note.get("midi", 60))) for part in parts for note in (part.get("notes") or [])][:64],
        "warnings": [],
    }


def _pad_to(signal: np.ndarray, size: int) -> np.ndarray:
    if signal.size >= size:
        return signal
    return np.pad(signal, (0, size - signal.size))


def _main() -> int:  # pragma: no cover - manual runs
    import argparse

    parser = argparse.ArgumentParser(prog="shadow-render", description="Render a song spec to WAV.")
    parser.add_argument("spec", help="JSON file with the song spec")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    result = render_song(spec) if "parts" in spec else render_part(spec)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
