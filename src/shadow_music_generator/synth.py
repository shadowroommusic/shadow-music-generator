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

__all__ = [
    "export_midi",
    "export_stems",
    "mix_arrangement",
    "read_midi_notes",
    "render_part",
    "render_song",
    "write_audio",
    "write_midi_multitrack",
    "write_wav",
]

SAMPLE_RATE = 44_100
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _as_frames(samples: np.ndarray) -> np.ndarray:
    """Audio as (frames, channels); 1-D input counts as mono."""
    return samples if samples.ndim == 2 else samples[:, None]


def _fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Convolve one channel with an impulse response through the FFT (no scipy needed)."""
    size = signal.size + impulse.size - 1
    spectrum = np.fft.rfft(signal, size) * np.fft.rfft(impulse, size)
    return np.fft.irfft(spectrum, size)


def _reverb(signal: np.ndarray, rate: int, amount: float = 0.22, decay_s: float = 0.9, seed: int = 11) -> np.ndarray:
    """Small-room reverb: a decaying noise impulse per channel, decorrelated so it widens the image."""
    if amount <= 0:
        return signal
    frames = _as_frames(signal).astype(np.float32)
    rng = np.random.default_rng(seed)
    tail = int(decay_s * rate)
    time = np.arange(tail) / rate
    out = np.empty_like(frames)
    for channel in range(frames.shape[1]):
        impulse = rng.standard_normal(tail) * np.exp(-time / (decay_s / 4.2))
        impulse[: int(0.012 * rate)] = 0.0  # pre-delay keeps transients tight
        impulse /= float(np.sqrt(np.sum(impulse**2))) + 1e-9
        wet = _fft_convolve(frames[:, channel], impulse)[: frames.shape[0]]
        out[:, channel] = (1.0 - amount) * frames[:, channel] + amount * wet * float(np.std(frames[:, channel]) or 1.0)
    return out


def _delay(signal: np.ndarray, rate: int, time_ms: float = 375.0, feedback: float = 0.28, mix: float = 0.18) -> np.ndarray:
    """Two-tap delay with feedback, dotted-eighth by default."""
    if mix <= 0:
        return signal
    frames = _as_frames(signal).astype(np.float32)
    delay = int(rate * time_ms / 1000)
    if delay <= 0:
        return frames
    out = frames.copy()
    tap = np.zeros_like(frames)
    tap[delay:] = frames[:-delay]
    out += tap * mix
    tap2 = np.zeros_like(frames)
    tap2[2 * delay :] = frames[: -2 * delay] if frames.shape[0] > 2 * delay else 0
    out += tap2 * mix * feedback
    return out


def _pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Equal-power pan: -1 hard left, 0 centre, +1 hard right."""
    angle = (float(np.clip(position, -1.0, 1.0)) + 1.0) * np.pi / 4.0
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=1)


def _swing_offset(step: int, swing: float, step_samples: float) -> float:
    """Swing pushes the offbeat eighths (grid steps 2, 6, 10, …) later."""
    if swing <= 0 or step % 4 != 2:
        return 0.0
    return float(swing) * step_samples


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


def write_audio(
    path: "str | Path",
    samples: np.ndarray,
    rate: int,
    *,
    bit_depth: int = 16,
    container: str = "wav",
) -> Path:
    """Write audio as WAV (16/24-bit) or AIFF (16-bit); 1-D input is mono, 2-D is interleaved.

    Anything lossy (MP3/AAC) needs an encoder binary, which this project deliberately does not depend
    on — the export UI says so instead of pretending.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = _as_frames(samples)
    clipped = np.clip(frames, -1.0, 1.0)
    channels = clipped.shape[1]
    if container == "aiff":
        payload = (clipped * 32767.0).astype(">i2").tobytes()
        # 80-bit IEEE-754 extended sample rate, the one awkward part of the AIFF header
        exponent = 16398
        mantissa = int(rate) << 48
        header = (
            b"FORM"
            + struct.pack(">I", 4 + 8 + 18 + 8 + len(payload))
            + b"AIFF"
            + b"COMM"
            + struct.pack(">I", 18)
            + struct.pack(">hIh", channels, clipped.shape[0], 16)
            + struct.pack(">HQ", exponent, mantissa)
            + b"SSND"
            + struct.pack(">I", len(payload) + 8)
            + struct.pack(">II", 0, 0)
        )
        target.write_bytes(header + payload)
        return target
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(3 if bit_depth == 24 else 2)
        handle.setframerate(rate)
        if bit_depth == 24:
            scaled = (clipped * 8388607.0).astype("<i4")
            packed = scaled.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
            handle.writeframes(packed)
        else:
            handle.writeframes((clipped * 32767.0).astype("<i2").tobytes())
    return target


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample — plenty for a mixdown, and no scipy dependency."""
    if samples.ndim == 2:
        return np.stack(
            [_resample(samples[:, channel], source_rate, target_rate) for channel in range(samples.shape[1])],
            axis=1,
        )
    if source_rate == target_rate or samples.size == 0:
        return samples
    count = int(round(samples.size * target_rate / source_rate))
    if count <= 0:
        return samples[:0]
    source_index = np.linspace(0, samples.size - 1, count)
    return np.interp(source_index, np.arange(samples.size), samples).astype(np.float32)


def read_audio(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Read a PCM WAV as float frames: `(N,)` for mono, `(N, channels)` for interleaved."""
    with wave.open(str(Path(path).expanduser()), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16))
        values = np.where(values >= 1 << 23, values - (1 << 24), values)
        data = values.astype(np.float32) / 8388608.0
    elif width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    else:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, rate


def read_wav(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Read a PCM WAV folded to mono (melody/groove analysis wants one channel)."""
    data, rate = read_audio(path)
    return (data.mean(axis=1) if data.ndim == 2 else data), rate


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
    swing = float(spec.get("swing", 0.0))
    humanize = float(spec.get("humanize_ms", 0.0)) / 1000 * rate
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
            jitter = rng.uniform(-humanize, humanize) if humanize > 0 else 0.0
            position = int(round(step * step_samples + _swing_offset(step, swing, step_samples) + jitter))
            position = max(0, position)
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
    """Render one part (drums, bass, chords, lead, pad) to a stereo buffer.

    The voice is rendered mono, then placed with `pan`, touched by `delay` and `reverb`, so a part
    arrives with its own position in the image instead of a dry centre line.
    """
    bpm = float(spec.get("bpm", 120))
    beats = float(spec.get("beats") or spec.get("bars", 1) * 4)
    kind = str(spec.get("part", "bass"))
    rng = np.random.default_rng(int(spec.get("seed", 7)))
    if kind == "drums":
        mono = _render_drums({**spec, "bpm": bpm, "bars": spec.get("bars", 1)}, beats, rate, rng)
    else:
        mono = _render_notes({**spec, "bpm": bpm}, beats, rate)
    stereo = _pan(mono, float(spec.get("pan", 0.0)))
    stereo = _delay(
        stereo,
        rate,
        time_ms=float(spec.get("delay_ms", 375.0)),
        feedback=float(spec.get("delay_feedback", 0.28)),
        mix=float(spec.get("delay", 0.0)),
    )
    return _reverb(
        stereo,
        rate,
        amount=float(spec.get("reverb", 0.12)),
        decay_s=float(spec.get("reverb_decay", 0.9)),
        seed=int(spec.get("seed", 7)),
    )


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
    target = write_audio(out_path, samples, SAMPLE_RATE)
    beats = float(spec.get("beats") or spec.get("bars", 1) * 4)
    return {
        "schema_version": 1,
        "mode": "render-part",
        "part": str(spec.get("part", "bass")),
        "path": str(target),
        "bpm": float(spec.get("bpm", 120)),
        "bars": beats / 4,
        "duration_ms": _duration_ms(samples.shape[0], SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "pan": float(spec.get("pan", 0.0)),
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
        write_audio(target, samples, SAMPLE_RATE)
        rendered.append(
            {
                "part": kind,
                "path": str(target),
                "duration_ms": _duration_ms(samples.shape[0], SAMPLE_RATE),
                "gain": float(part.get("gain", 1.0)),
                "pan": float(part.get("pan", 0.0)),
            }
        )
        contribution = samples * float(part.get("gain", 1.0))
        mix = contribution if mix is None else _pad_to(mix, contribution.shape[0]) + _pad_to(contribution, mix.shape[0])

    assert mix is not None
    mix = np.tanh(mix * 0.9)  # soft clip instead of hard clipping
    mix = _normalise(mix, peak=0.92)
    write_audio(song, mix, SAMPLE_RATE)
    return {
        "schema_version": 1,
        "mode": "render-song",
        "path": str(song),
        "bpm": bpm,
        "bars": bars,
        "duration_ms": _duration_ms(mix.shape[0], SAMPLE_RATE),
        "channels": 2,
        "sample_rate": SAMPLE_RATE,
        "parts": rendered,
        "notes": [note_name(int(note.get("midi", 60))) for part in parts for note in (part.get("notes") or [])][:64],
        "warnings": [],
    }


def _pad_to(signal: np.ndarray, size: int) -> np.ndarray:
    if signal.shape[0] >= size:
        return signal
    padding = [(0, size - signal.shape[0])] + ([(0, 0)] if signal.ndim == 2 else [])
    return np.pad(signal, padding)


def mix_arrangement(spec: dict) -> dict:
    """Bounce an arrangement to one file.

    `tracks` mirror the studio: each clip carries a file path and its position in bars, so the mix is
    the arrangement the user sees — placed, gained, and rendered at the requested sample rate, bit
    depth and container.
    """
    out_path = spec.get("out_path")
    if not out_path:
        raise ValueError("mix_arrangement needs an out_path")
    bpm = float(spec.get("bpm", 120))
    bars = float(spec.get("bars", 8))
    sample_rate = int(spec.get("sample_rate", SAMPLE_RATE))
    bit_depth = int(spec.get("bit_depth", 16))
    container = str(spec.get("container", "wav")).lower()
    if container not in ("wav", "aiff"):
        raise ValueError("container must be wav or aiff (lossy formats need an encoder)")
    seconds_per_bar = 60.0 / max(1.0, bpm) * 4
    total = int(round(bars * seconds_per_bar * sample_rate))
    mix, used = _mix_tracks(spec.get("tracks") or [], sample_rate, seconds_per_bar, total)

    mix = _normalise(np.tanh(mix * 0.9), peak=0.94)
    target = write_audio(out_path, mix, sample_rate, bit_depth=bit_depth, container=container)
    return {
        "schema_version": 1,
        "mode": "mixdown",
        "path": str(target),
        "bpm": bpm,
        "bars": bars,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "container": container,
        "duration_ms": _duration_ms(mix.shape[0], sample_rate),
        "channels": mix.shape[1],
        "clip_count": len(used),
        "tracks": used,
        "warnings": [],
    }


def _mix_tracks(
    tracks: "list[dict]",
    sample_rate: int,
    seconds_per_bar: float,
    total: int,
) -> "tuple[np.ndarray, list[dict]]":
    """Sum the clips of `tracks` onto one buffer, honouring position, length and gains."""
    mix = np.zeros((max(1, total), 2), dtype=np.float32)
    used: "list[dict]" = []
    for track in tracks:
        track_gain = float(track.get("gain", 1.0))
        if track_gain <= 0:
            continue
        for clip in track.get("clips") or []:
            path = clip.get("path")
            if not path:
                continue
            try:
                samples, rate = read_audio(path)
            except Exception:
                continue
            if samples.ndim == 1:
                samples = np.stack([samples, samples], axis=1)
            samples = _resample(samples, rate, sample_rate)
            start = int(round(float(clip.get("start", 0)) * seconds_per_bar * sample_rate))
            if start >= mix.shape[0]:
                continue
            length = samples.shape[0]
            if clip.get("bars") is not None:
                # honour the clip's length on the grid: trim, or pad with silence
                wanted = int(round(float(clip["bars"]) * seconds_per_bar * sample_rate))
                length = wanted
                if samples.shape[0] > wanted:
                    samples = samples[:wanted]
                elif samples.shape[0] < wanted:
                    samples = _pad_to(samples, wanted)
            end = min(mix.shape[0], start + length)
            if end <= start:
                continue
            mix[start:end] += samples[: end - start] * track_gain * float(clip.get("gain", 1.0))
            used.append({"part": track.get("name", "track"), "path": str(path)})
    return mix, used


def export_stems(spec: dict) -> dict:
    """Bounce every track of the arrangement to its own file.

    One file per studio track, in the requested rate / depth / container, so the parts can be taken
    into a DAW or handed to a mastering stage separately.
    """
    out_dir = spec.get("out_dir") or spec.get("out_path")
    if not out_dir:
        raise ValueError("export_stems needs an out_dir")
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    base = str(spec.get("name") or "stem").strip() or "stem"
    bpm = float(spec.get("bpm", 120))
    bars = float(spec.get("bars", 8))
    sample_rate = int(spec.get("sample_rate", SAMPLE_RATE))
    bit_depth = int(spec.get("bit_depth", 16))
    container = str(spec.get("container", "wav")).lower()
    if container not in ("wav", "aiff"):
        raise ValueError("container must be wav or aiff (lossy formats need an encoder)")
    seconds_per_bar = 60.0 / max(1.0, bpm) * 4
    total = int(round(bars * seconds_per_bar * sample_rate))

    stems: "list[dict]" = []
    for index, track in enumerate(spec.get("tracks") or []):
        name = str(track.get("name") or f"track{index + 1}")
        mix, used = _mix_tracks([track], sample_rate, seconds_per_bar, total)
        if not used:
            continue
        samples = _normalise(np.tanh(mix * 0.9), peak=0.94)
        safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in name)
        target = directory / f"{base}-{index + 1:02d}-{safe}.{container}"
        write_audio(target, samples, sample_rate, bit_depth=bit_depth, container=container)
        stems.append(
            {
                "name": name,
                "path": str(target),
                "duration_ms": _duration_ms(samples.shape[0], sample_rate),
                "channels": 2,
                "clip_count": len(used),
            }
        )
    if not stems:
        raise ValueError("no playable clips in this arrangement")
    return {
        "schema_version": 1,
        "mode": "stem-export",
        "out_dir": str(directory),
        "bpm": bpm,
        "bars": bars,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "container": container,
        "stem_count": len(stems),
        "stems": stems,
        "warnings": [],
    }


def _vlq(value: int) -> bytes:
    """MIDI variable-length quantity."""
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_midi_multitrack(parts: "list[dict]", path: "str | Path", *, bpm: float = 120.0) -> Path:
    """Write a Type-1 MIDI file: one track per part, notes in milliseconds.

    `parts` entries are `{"name": str, "notes": [{"midi", "start_ms", "end_ms", "velocity"?}]}`.
    """
    ticks_per_beat = 480
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    micros_per_beat = int(round(60_000_000 / max(1.0, bpm)))

    def to_ticks(ms: float) -> int:
        return int(round(ms / 60000.0 * bpm * ticks_per_beat))

    chunks: "list[bytes]" = []
    # track 0: tempo + time signature, so the file lands on the right grid in any DAW
    conductor = bytearray()
    conductor += _vlq(0) + b"\xff\x51\x03" + struct.pack(">I", micros_per_beat)[1:]
    conductor += _vlq(0) + b"\xff\x58\x04" + bytes([4, 2, 24, 8])
    conductor += _vlq(0) + b"\xff\x2f\x00"
    chunks.append(bytes(conductor))

    for part in parts:
        events: "list[tuple[int, int, bytes]]" = []
        for order, note in enumerate(part.get("notes") or []):
            midi = max(0, min(127, int(note.get("midi", 60))))
            velocity = max(1, min(127, int(note.get("velocity", 100))))
            start = to_ticks(float(note.get("start_ms", 0)))
            end = max(start + 1, to_ticks(float(note.get("end_ms", 0))))
            events.append((start, 1, bytes([0x90, midi, velocity])))
            events.append((end, 0, bytes([0x80, midi, 64])))
        events.sort(key=lambda item: (item[0], item[1]))
        track = bytearray()
        name = str(part.get("name") or "part").encode("utf-8")[:127]
        track += _vlq(0) + b"\xff\x03" + _vlq(len(name)) + name
        previous = 0
        for tick, _order, payload in events:
            track += _vlq(tick - previous)
            previous = tick
            track += payload
        track += _vlq(0) + b"\xff\x2f\x00"
        chunks.append(bytes(track))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ticks_per_beat)
    body = b"".join(b"MTrk" + struct.pack(">I", len(chunk)) + chunk for chunk in chunks)
    target.write_bytes(header + body)
    return target


def read_midi_notes(path: "str | Path") -> "tuple[list[dict], float | None]":
    """Read every note of a MIDI file as `{midi, start_ms, end_ms}`, plus its tempo if it has one."""
    data = Path(path).expanduser().read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    _format, track_count, division = struct.unpack(">HHH", data[8:14])
    ticks_per_beat = division or 480
    offset = 14
    tempo: float | None = None
    notes: "list[dict]" = []
    for _ in range(track_count):
        if data[offset : offset + 4] != b"MTrk":
            break
        size = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        end = offset + 8 + size
        index = offset + 8
        tick = 0
        status = 0
        sounding: "dict[int, tuple[int, int]]" = {}

        def clock(ticks: int) -> float:
            bpm = tempo or 120.0
            return ticks / ticks_per_beat * (60_000.0 / bpm)

        while index < end:
            delta, index = _read_vlq(data, index)
            tick += delta
            byte = data[index]
            if byte == 0xFF:
                index += 1
                kind = data[index]
                index += 1
                length, index = _read_vlq(data, index)
                payload = data[index : index + length]
                index += length
                if kind == 0x51 and length == 3:
                    tempo = 60_000_000 / int.from_bytes(payload, "big")
                continue
            if byte in (0xF0, 0xF7):
                index += 1
                length, index = _read_vlq(data, index)
                index += length
                continue
            if byte & 0x80:
                status = byte
                index += 1
            command = status & 0xF0
            if command in (0x80, 0x90):
                note = data[index]
                velocity = data[index + 1]
                index += 2
                if command == 0x90 and velocity > 0:
                    sounding[note] = (tick, velocity)
                else:
                    started = sounding.pop(note, None)
                    if started is not None:
                        notes.append(
                            {
                                "midi": note,
                                "velocity": started[1],
                                "start_ms": clock(started[0]),
                                "end_ms": clock(tick),
                            }
                        )
                continue
            if command in (0xA0, 0xB0, 0xE0):
                index += 2
                continue
            if command in (0xC0, 0xD0):
                index += 1
                continue
            break
        offset = end
    notes.sort(key=lambda note: note["start_ms"])
    return notes, tempo


def _read_vlq(data: bytes, index: int) -> "tuple[int, int]":
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def export_midi(spec: dict) -> dict:
    """Write the arrangement as MIDI: one track per studio track, notes placed on the grid.

    A clip contributes notes either directly (`notes`, milliseconds relative to the clip — what the
    hum-to-MIDI flow attaches) or by pointing at a `.mid` file, which is parsed. Audio-only clips are
    reported back so the UI can say what could not be notated.
    """
    out_path = spec.get("out_path")
    if not out_path:
        raise ValueError("export_midi needs an out_path")
    bpm = float(spec.get("bpm", 120))
    seconds_per_bar = 60.0 / max(1.0, bpm) * 4
    parts: "list[dict]" = []
    skipped: "list[str]" = []
    for index, track in enumerate(spec.get("tracks") or []):
        name = str(track.get("name") or f"track{index + 1}")
        notes: "list[dict]" = []
        for clip in track.get("clips") or []:
            offset_ms = float(clip.get("start", 0)) * seconds_per_bar * 1000
            clip_notes = clip.get("notes")
            if clip_notes:
                for note in clip_notes:
                    notes.append(
                        {
                            "midi": int(note.get("midi", 60)),
                            "start_ms": offset_ms + float(note.get("start_ms", 0)),
                            "end_ms": offset_ms + float(note.get("end_ms", 0)),
                            "velocity": int(note.get("velocity", 100)),
                        }
                    )
                continue
            path = str(clip.get("path") or "")
            if path.lower().endswith((".mid", ".midi")):
                try:
                    parsed, _tempo = read_midi_notes(path)
                except Exception:
                    skipped.append(f"{name}: {Path(path).name} (unreadable MIDI)")
                    continue
                for note in parsed:
                    notes.append(
                        {
                            "midi": note["midi"],
                            "start_ms": offset_ms + note["start_ms"],
                            "end_ms": offset_ms + note["end_ms"],
                            "velocity": note.get("velocity", 100),
                        }
                    )
                continue
            if path:
                skipped.append(f"{name}: {Path(path).name} (audio — run hum_to_midi first)")
        if notes:
            parts.append({"name": name, "notes": notes})
    if not parts:
        raise ValueError("nothing to notate: no clip carries notes or MIDI")
    target = write_midi_multitrack(parts, out_path, bpm=bpm)
    return {
        "schema_version": 1,
        "mode": "midi-export",
        "path": str(target),
        "bpm": bpm,
        "bars": float(spec.get("bars", 8)),
        "track_count": len(parts),
        "note_count": sum(len(part["notes"]) for part in parts),
        "tracks": [{"name": part["name"], "notes": len(part["notes"])} for part in parts],
        "skipped": skipped,
        "warnings": skipped[:3],
    }


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
