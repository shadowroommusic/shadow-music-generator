"""Offline synthesis: turn a musical spec into real audio.

This is the local, model-free renderer behind `render_part` / `render_song`. The agent writes the
music — patterns, notes, tempo — and these functions turn it into 16-bit WAV files that the studio can
play. A model-backed backend (YuE2, a cloud API, …) can be dropped in behind the same interface later.

Everything is numpy + the standard library; no audio packages are required.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
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


#: Containers an export can ask for, in the order the UI lists them.
#:
#: `wav`/`aiff` are written here and always work. `flac`/`alac`/`m4a` (AAC) work with either the
#: encoder macOS ships (`afconvert`) or ffmpeg. `mp3`/`ogg`/`opus` need ffmpeg — CoreAudio has no MP3
#: encoder, so offering one without it would be a lie rather than a feature.
CONTAINERS = ("wav", "aiff", "flac", "alac", "m4a", "mp3", "ogg", "opus")
#: Containers that lose information, with the default bitrate each encoder gets.
LOSSY = {"m4a": 256000, "mp3": 320000, "ogg": 192000, "opus": 128000}
#: Default ffmpeg/libav codec per container.
FFMPEG_CODECS = {
    "flac": ["-c:a", "flac"],
    "alac": ["-c:a", "alac"],
    "m4a": ["-c:a", "aac"],
    "mp3": ["-c:a", "libmp3lame"],
    "ogg": ["-c:a", "libvorbis"],
    "opus": ["-c:a", "libopus"],
}
#: Default CoreAudio data format per container (`afconvert -d`).
AFCONVERT_FORMATS = {"flac": "flac", "alac": "alac", "m4a": "aac"}
#: Containers whose extension differs from their name.
EXTENSIONS = {"alac": "m4a"}


def find_ffmpeg() -> "str | None":
    """ffmpeg from PATH, from the ShadowRoom `bin`, or from the imageio-ffmpeg wheel.

    ShadowRoom keeps one static ffmpeg at `<home>/bin/ffmpeg` (a symlink to the imageio-ffmpeg
    binary), which every plugin gets first on PATH; the import is only a fallback for a process that
    started without that PATH.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    home = Path(os.environ.get("SHADOWROOM_HOME") or Path.home() / "Documents" / "ShadowRoom")
    staged = home / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if staged.exists():
        return str(staged)
    try:
        import imageio_ffmpeg  # type: ignore

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def find_afconvert() -> "str | None":
    """macOS's own converter, when this is a Mac."""
    return shutil.which("afconvert")


def container_capabilities() -> "list[dict]":
    """What this machine can actually write, for the export dialog and for error messages."""
    ffmpeg = find_ffmpeg()
    afconvert = find_afconvert()
    out: "list[dict]" = []
    for name in CONTAINERS:
        if name in ("wav", "aiff"):
            available, via = True, "built-in"
        elif ffmpeg is not None:
            available, via = True, "ffmpeg"
        elif name in AFCONVERT_FORMATS and afconvert is not None:
            available, via = True, "afconvert"
        else:
            available, via = False, "ffmpeg"
        out.append(
            {
                "id": name,
                "extension": EXTENSIONS.get(name, name),
                "lossy": name in LOSSY,
                "available": available,
                "via": via,
                "default_bitrate": LOSSY.get(name),
            }
        )
    return out


def _unsupported_message(container: str) -> str:
    if container == "mp3":
        return "MP3 needs an encoder: install ffmpeg (macOS has no MP3 encoder of its own)"
    return f"{container} needs ffmpeg on this machine"


def _write_container(
    path: "str | Path",
    samples: np.ndarray,
    rate: int,
    bit_depth: int,
    container: str,
    bitrate: object = None,
) -> Path:
    """Write a bounce in the requested container, staged through a temporary PCM file when needed.

    Every compressed format goes through ffmpeg when it exists (one code path, every codec); macOS's
    afconvert covers flac/alac/aac when it does not. A format the machine cannot write raises with a
    sentence that says how to enable it — never a silent downgrade.
    """
    container = container.lower()
    if container not in CONTAINERS:
        raise ValueError(f"container must be one of {', '.join(CONTAINERS)}")
    target = Path(path).expanduser()
    if container in ("wav", "aiff"):
        return write_audio(target, samples, rate, bit_depth=bit_depth, container=container)
    wanted = int(bitrate) if isinstance(bitrate, (int, float)) and bitrate else LOSSY.get(container)
    ffmpeg = find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="shadow-export-") as tmp:
        staged = write_audio(Path(tmp) / "bounce.wav", samples, rate, bit_depth=24 if bit_depth == 24 else 16, container="wav")
        if ffmpeg is not None:
            argv = [ffmpeg, "-v", "error", "-y", "-i", str(staged), *FFMPEG_CODECS[container]]
            if wanted and container in LOSSY:
                argv += ["-b:a", str(wanted)]
            argv.append(str(target))
            result = subprocess.run(argv, capture_output=True, text=True)
            if result.returncode == 0 and target.exists():
                return target
            detail = (result.stderr or "ffmpeg failed").strip().splitlines()
            raise ValueError(f"ffmpeg could not write {container}: {detail[-1] if detail else 'unknown error'}")
        afconvert = find_afconvert()
        if afconvert is not None and container in AFCONVERT_FORMATS:
            argv = [afconvert, "-f", "flac" if container == "flac" else "m4af", "-d", AFCONVERT_FORMATS[container], str(staged), str(target)]
            if container == "m4a":
                argv = [afconvert, "-f", "m4af", "-d", "aac", "-b", str(wanted or 256000), str(staged), str(target)]
            result = subprocess.run(argv, capture_output=True, text=True)
            if result.returncode == 0 and target.exists():
                return target
            detail = (result.stderr or "afconvert failed").strip().splitlines()
            raise ValueError(f"afconvert could not write {container}: {detail[-1] if detail else 'unknown error'}")
    raise ValueError(_unsupported_message(container))


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
    """Read a PCM WAV as float frames: `(N,)` for mono, `(N, channels)` for interleaved.

    Python's `wave` module only understands the plain PCM headers; anything with a
    `WAVE_FORMAT_EXTENSIBLE` header (format tag 0xFFFE) — which is what afconvert and most DAWs
    write — falls through to `_read_riff_audio` below.
    """
    try:
        return _read_wave_module(path)
    except Exception:
        return _read_riff_audio(path)


def _read_wave_module(path: "str | Path") -> "tuple[np.ndarray, int]":
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


def _read_riff_audio(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Minimal RIFF/WAVE reader: PCM 8/16/24/32-bit and 32-bit float, extensible headers included.

    Enough to read what `afconvert` decodes (it writes the extensible header) and what DAWs export,
    without depending on a decoder library.
    """
    raw = Path(path).expanduser().read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    offset = 12
    fmt: "tuple[int, int, int, int] | None" = None
    payload = b""
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        size = struct.unpack("<I", raw[offset + 4 : offset + 8])[0]
        body = raw[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]  # the sub-format's own tag
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            payload = body
        offset += 8 + size + (size % 2)
    if fmt is None or payload == b"":
        raise ValueError(f"no audio data in {path}")
    tag, channels, rate, bits = fmt
    if tag == 3:
        data = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 8:
        data = (np.frombuffer(payload, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        data = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        packed = np.frombuffer(payload, dtype=np.uint8)
        packed = packed[: (packed.size // 3) * 3].reshape(-1, 3)
        values = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        data = np.where(values >= 1 << 23, values - (1 << 24), values).astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 32:
        data = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV format tag {tag} ({bits}-bit) in {path}")
    if channels > 1:
        data = data[: (data.size // channels) * channels].reshape(-1, channels)
    return data, rate


def _decode_command() -> "tuple[str, ...] | None":
    """The external decoder whose arguments we know, or `None` when there is none.

    macOS ships `afconvert` (CoreAudio: mp3, m4a/aac, flac, aiff, caf…); ffmpeg is the same thing on
    a Linux or Windows box. Neither is a dependency — the plugin just stops being able to read those
    formats, and says so, when both are missing.
    """
    if (found := shutil.which("ffmpeg")) is not None:
        return (found, "-v", "error", "-y", "-i")
    if (found := shutil.which("afconvert")) is not None:
        return (found, "-f", "WAVE", "-d", "LEI16", "-o")
    return None


def read_audio_any(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Read any clip the studio can hold: WAV directly, everything else through a decoder.

    The mixdown used to skip anything it could not read with `wave`, so an imported mp3 simply was
    not in the "export the song" result — silently. Now the read either works (CoreAudio/ffmpeg) or
    raises, and the caller reports which clip it could not use.
    """
    target = Path(path).expanduser()
    if target.suffix.lower() in (".wav", ".wave", ""):
        try:
            return read_audio(target)
        except Exception:
            pass
    command = _decode_command()
    if command is None:
        raise ValueError(
            f"cannot read {target.suffix or 'this file'} — install ffmpeg, or export it as WAV first"
        )
    with tempfile.TemporaryDirectory(prefix="shadow-audio-") as tmp:
        decoded = Path(tmp) / "decoded.wav"
        if command[0].endswith("afconvert"):
            argv = [*command, str(decoded), str(target)]
        else:
            argv = [*command, str(target), str(decoded)]
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0 or not decoded.exists():
            detail = (result.stderr or result.stdout or "decoder failed").strip().splitlines()
            raise ValueError(f"could not decode {target.name}: {detail[-1] if detail else 'unknown error'}")
        return read_audio(decoded)


def _encode_lossy(source: Path, target: Path, *, bitrate: int = 256000) -> Path:
    """Encode a PCM file to AAC in an `.m4a`, through afconvert (macOS) or ffmpeg.

    The project still ships no encoder of its own — but macOS has one built in, so refusing to write
    an m4a when the machine can would be pretending rather than keeping a promise.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        argv = [ffmpeg, "-v", "error", "-y", "-i", str(source), "-c:a", "aac", "-b:a", str(bitrate), str(target)]
        result = subprocess.run(argv, capture_output=True, text=True)
    elif (afconvert := shutil.which("afconvert")) is not None:
        argv = [afconvert, "-f", "m4af", "-d", "aac", "-b", str(bitrate), str(source), str(target)]
        result = subprocess.run(argv, capture_output=True, text=True)
    else:
        raise ValueError("m4a needs an AAC encoder: install ffmpeg (afconvert ships with macOS)")
    if result.returncode != 0 or not target.exists():
        detail = (result.stderr or result.stdout or "encoder failed").strip().splitlines()
        raise ValueError(detail[-1] if detail else "the AAC encoder failed")
    return target


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

#: General-MIDI pitches for those voices: what a rendered beat looks like as notes.
DRUM_MIDI = {"kick": 36, "snare": 38, "clap": 39, "hat": 42, "openhat": 46}


def _stretch_notes(spec: dict, bars: float) -> "list[dict]":
    """What one stretch of a part plays, in beats from the stretch's own start.

    This is the note view of the same spec the renderer plays, repeats included — drums tile their
    step rows, pitched parts tile their written notes only when `repeat` asks for it.
    """
    if float(spec.get("gain", 1.0)) <= 0 or spec.get("silent"):
        return []
    out: "list[dict]" = []
    if str(spec.get("part", "")) == "drums":
        steps = int(spec.get("steps", 16))
        total_steps = int(round(bars * steps))
        # Swing is part of the plan (it is applied to the audio too); humanize is not, because it is
        # random per hit and belongs to the performance rather than to the notation.
        swing = float(spec.get("swing", 0.0))
        for voice, row in (spec.get("pattern") or {}).items():
            pitch = DRUM_MIDI.get(voice)
            if pitch is None or not isinstance(row, str):
                continue
            block = row.replace("|", "").replace(" ", "")
            if block == "":
                continue
            for repeat in range(0, max(1, -(-total_steps // len(block)))):
                for index, symbol in enumerate(block):
                    if symbol not in "xXoO":
                        continue
                    step = repeat * len(block) + index
                    if step >= total_steps:
                        continue
                    out.append(
                        {
                            "midi": pitch,
                            "start_beats": (step + (swing if step % 4 == 2 else 0.0)) * 4.0 / steps,
                            "length_beats": 4.0 / steps / 2,
                            "velocity": 112 if symbol in "xX" else 96,
                        }
                    )
        return out

    written = spec.get("notes") or []
    if not written:
        return []
    written_end = max(float(note.get("start", 0)) + float(note.get("length", 1)) for note in written)
    block_beats = max(4.0, 4.0 * math.ceil(written_end / 4.0))
    repeat = spec.get("repeat")
    copies = 1
    if repeat:
        wanted = int(repeat) if isinstance(repeat, (int, float)) and not isinstance(repeat, bool) else 0
        copies = wanted if wanted > 0 else max(1, int(math.ceil(bars * 4.0 / block_beats)))
    limit = bars * 4.0
    for copy in range(copies):
        for note in written:
            start = float(note.get("start", 0)) + copy * block_beats
            if start >= limit:
                continue
            out.append(
                {
                    "midi": int(note.get("midi", 60)),
                    "start_beats": start,
                    "length_beats": float(note.get("length", 1)),
                    "velocity": int(note.get("velocity", 100)),
                }
            )
    return out


def song_midi_parts(spec: dict, bpm: float, bars: float) -> "list[dict]":
    """The whole song as MIDI parts: every part, every segment, placed on the grid in milliseconds.

    The renderer and this share one reading of the spec, so the MIDI that comes out of
    `render_song(midi_path=…)` is the arrangement that was rendered — drums on MIDI's drum channel
    (10), pitched parts as written.
    """
    parts: "list[dict]" = []
    for index, part in enumerate(spec.get("parts") or []):
        name = str(part.get("part", f"part{index + 1}"))
        stretches: "list[tuple[float, dict, float]]" = []
        segments = part.get("segments")
        if isinstance(segments, list) and segments:
            start = 0.0
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                length = float(segment.get("bars", 1))
                stretches.append((start, {**part, **segment}, length))
                start += length
        else:
            stretches.append((0.0, part, bars))
        notes: "list[dict]" = []
        for start_bar, stretch, length in stretches:
            for note in _stretch_notes(stretch, length):
                begin = (start_bar * 4.0 + note["start_beats"]) * 60_000.0 / bpm
                notes.append(
                    {
                        "midi": note["midi"],
                        "start_ms": begin,
                        "end_ms": begin + note["length_beats"] * 60_000.0 / bpm,
                        "velocity": note.get("velocity", 100),
                    }
                )
        notes.sort(key=lambda entry: entry["start_ms"])
        parts.append({"name": name, "channel": 9 if name == "drums" else 0, "notes": notes})
    return parts


def _render_drums(spec: dict, beats: float, rate: int, rng: np.random.Generator) -> np.ndarray:
    """A block of step rows, tiled across the part's bars.

    One 16-step row is one bar, a 32-step row is two, so the row's own length decides the block and
    the block repeats until the part is full. Without the repeat a 4-bar beat was one bar of audio
    followed by three bars of silence — measured on the stems the workbench had already rendered
    (`night-drive-drums.wav`: bar 1 = 0.108 RMS, bars 2–8 = 0.000).
    """
    pattern = spec.get("pattern") or {}
    steps = int(spec.get("steps", 16))
    bars = int(spec.get("bars", 1))
    beat_samples = rate * _seconds_per_beat(spec.get("bpm", 120))
    step_samples = beat_samples / (steps / 4)  # 16 steps = one bar of 4 beats
    swing = float(spec.get("swing", 0.0))
    humanize = float(spec.get("humanize_ms", 0.0)) / 1000 * rate
    total = int(round(beat_samples * beats))
    total_steps = int(round(beats * (steps / 4)))
    out = np.zeros(total)
    for voice, row in pattern.items():
        factory = DRUM_VOICES.get(voice)
        if factory is None or not isinstance(row, str):
            continue
        hit = factory(rate, rng) * float(spec.get("gain", 1.0))
        block = row.replace("|", "").replace(" ", "")
        if block == "":
            continue
        for repeat in range(0, max(1, -(-total_steps // len(block)))):
            for index, symbol in enumerate(block):
                if symbol not in "xXoO":
                    continue
                step = repeat * len(block) + index
                jitter = rng.uniform(-humanize, humanize) if humanize > 0 else 0.0
                position = int(round(step * step_samples + _swing_offset(step, swing, step_samples) + jitter))
                position = max(0, position)
                length = min(hit.size, total - position)
                if length > 0:
                    out[position : position + length] += hit[:length]
    return out


def _render_notes(spec: dict, beats: float, rate: int) -> np.ndarray:
    """Pitched part: notes with midi number, start (beats) and length (beats).

    `repeat` tiles the written notes: `true` repeats them until the part is full, a number repeats
    them that many times. One bar of bass then covers an eight-bar part without the writer (or the
    model) spelling the same notes out eight times.
    """
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
    block_beats = 4.0
    if notes:
        written = max(float(note.get("start", 0)) + float(note.get("length", 1)) for note in notes)
        block_beats = max(4.0, 4.0 * math.ceil(written / 4.0))
    repeat = spec.get("repeat")
    copies = 1
    if repeat and notes:
        wanted = int(repeat) if isinstance(repeat, (int, float)) and not isinstance(repeat, bool) else 0
        copies = wanted if wanted > 0 else max(1, int(math.ceil(beats / block_beats)))
    for copy in range(copies):
        offset = copy * block_beats
        for note in notes:
            _place_note(out, note, offset, beat_samples, total, wave_kind, cutoff, attack, release, gain, vibrato, rate)
    return out


def _place_note(
    out: np.ndarray,
    note: dict,
    offset_beats: float,
    beat_samples: float,
    total: int,
    wave_kind: str,
    cutoff: float,
    attack: float,
    release: float,
    gain: float,
    vibrato: float,
    rate: int,
) -> None:
    """Add one note (shifted by `offset_beats`) to a pitched part's buffer."""
    midi = float(note.get("midi", 60))
    start = int(round((float(note.get("start", 0)) + offset_beats) * beat_samples))
    length = int(round(float(note.get("length", 1)) * beat_samples))
    length = max(1, min(length, total - start))
    if length <= 1 or start >= total:
        return
    time = np.arange(length) / rate
    frequency = np.full(length, midi_to_hz(midi))
    if vibrato > 0:
        frequency = frequency * (1 + vibrato * np.sin(2 * np.pi * 5.0 * time))
    voice = _oscillator(wave_kind, frequency)
    if wave_kind == "saw" and cutoff < 1.0:
        voice = _lowpass(voice, cutoff)
    envelope = _envelope(length, rate, attack, 0.05, 0.8, release)
    out[start : start + length] += voice * envelope * gain * float(note.get("gain", 1.0))


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

    An arrangement is a part made of `segments`: each segment is a stretch of bars rendered with the
    part's settings as defaults and the segment's keys overriding them, and the stretches are
    concatenated — so an intro can be hats only, the drop can add kick and clap, and a breakdown can
    be `{"bars": 8, "gain": 0}` (silence). `sections` names the timeline (`intro` / `drop` / …) and
    sets the total length when `bars` is not given.
    """
    out_path = spec.get("out_path")
    if not out_path:
        raise ValueError("render_song needs an out_path")
    song = Path(out_path).expanduser()
    song.parent.mkdir(parents=True, exist_ok=True)
    bpm = float(spec.get("bpm", 120))
    sections = _describe_sections(spec.get("sections") or [])
    bars = float(spec.get("bars") or (sum(entry["bars"] for entry in sections) if sections else 4))
    parts = spec.get("parts") or []
    if not parts:
        raise ValueError("render_song needs at least one part")
    total_samples = int(round(bars * 4 * _seconds_per_beat(bpm) * SAMPLE_RATE))

    rendered: "list[dict]" = []
    mix: np.ndarray | None = None
    for index, part in enumerate(parts):
        kind = str(part.get("part", f"part{index + 1}"))
        samples, segments = _render_part_arrangement(part, bpm, bars, index)
        # Every part is exactly the song long: a part that only enters later is padded, so the mix
        # and the stems line up with the timeline the sections describe.
        samples = _fit(samples, total_samples)
        samples = _normalise(samples, peak=0.85)
        target = Path(part["out_path"]).expanduser() if part.get("out_path") else song.with_name(f"{song.stem}-{kind}.wav")
        write_audio(target, samples, SAMPLE_RATE)
        rendered.append(
            {
                "part": kind,
                "path": str(target),
                "duration_ms": _duration_ms(samples.shape[0], SAMPLE_RATE),
                "bars": bars,
                "gain": float(part.get("gain", 1.0)),
                "pan": float(part.get("pan", 0.0)),
                "segments": segments,
            }
        )
        contribution = samples * float(part.get("gain", 1.0))
        mix = contribution if mix is None else _pad_to(mix, contribution.shape[0]) + _pad_to(contribution, mix.shape[0])

    assert mix is not None
    mix = np.tanh(mix * 0.9)  # soft clip instead of hard clipping
    mix = _normalise(mix, peak=0.92)
    write_audio(song, mix, SAMPLE_RATE)
    # The same arrangement as notes, when asked for: the MIDI is not transcribed from the audio, it
    # is the plan that was played, so it lines up with the stems exactly.
    midi_target: "Path | None" = None
    if spec.get("midi_path"):
        midi_target = write_midi_multitrack(
            song_midi_parts(spec, bpm, bars),
            str(spec["midi_path"]),
            bpm=bpm,
        )
    return {
        "schema_version": 1,
        "mode": "render-song",
        "path": str(song),
        "midi_path": str(midi_target) if midi_target is not None else None,
        "bpm": bpm,
        "bars": bars,
        "duration_ms": _duration_ms(mix.shape[0], SAMPLE_RATE),
        "channels": 2,
        "sample_rate": SAMPLE_RATE,
        "parts": rendered,
        "sections": sections,
        "notes": [note_name(int(note.get("midi", 60))) for part in parts for note in (part.get("notes") or [])][:64],
        "warnings": [],
    }


def _describe_sections(sections: "list[dict]") -> "list[dict]":
    """Name and place the song's sections: `[{name, bars, start_bar}]`, in timeline order."""
    described: "list[dict]" = []
    start = 0.0
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        length = float(section.get("bars", 0))
        described.append(
            {
                "name": str(section.get("name") or f"section {index + 1}"),
                "bars": length,
                "start_bar": start,
            }
        )
        start += length
    return described


def _fit(signal: np.ndarray, size: int) -> np.ndarray:
    """Trim or pad a rendered part so it is exactly `size` frames long."""
    frames = _as_frames(signal)
    if frames.shape[0] == size:
        return frames
    if frames.shape[0] > size:
        return frames[:size]
    return _pad_to(frames, size)


def _render_part_arrangement(
    part: dict,
    bpm: float,
    bars: float,
    index: int,
) -> "tuple[np.ndarray, list[dict]]":
    """Render one part of the song: a plain loop, or the chain of segments an arrangement is made of.

    A part without `segments` keeps the old behaviour exactly — one block of `bars`. With `segments`,
    each entry is rendered on its own and they are concatenated, which is what turns "a loop" into
    "a song": the same part plays different things per section, and `gain: 0` (or `silent: true`)
    leaves a stretch empty.
    """
    segments = part.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        payload = {**part, "bpm": bpm, "bars": bars, "seed": part.get("seed", 7 + index)}
        return renders_part(payload), []

    blocks: "list[np.ndarray]" = []
    described: "list[dict]" = []
    start = 0.0
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        length = float(segment.get("bars", 1))
        silent = bool(segment.get("silent")) or float(segment.get("gain", part.get("gain", 1.0))) == 0.0
        frames = int(round(length * 4 * _seconds_per_beat(bpm) * SAMPLE_RATE))
        if silent:
            blocks.append(np.zeros((max(0, frames), 2), dtype=np.float32))
        else:
            payload = {
                **part,
                **segment,
                "bpm": bpm,
                "bars": length,
                "seed": segment.get("seed", part.get("seed", 7 + index)) + position,
            }
            payload.pop("segments", None)
            blocks.append(renders_part(payload))
        described.append(
            {
                "bars": length,
                "start_bar": start,
                "silent": silent,
                "part": str(segment.get("part", part.get("part", ""))),
            }
        )
        start += length
    if not blocks:
        payload = {**part, "bpm": bpm, "bars": bars, "seed": part.get("seed", 7 + index)}
        payload.pop("segments", None)
        return renders_part(payload), []
    return np.concatenate(blocks, axis=0), described


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
    if container not in CONTAINERS:
        raise ValueError(f"container must be one of {', '.join(CONTAINERS)}")
    seconds_per_bar = 60.0 / max(1.0, bpm) * 4
    total = int(round(bars * seconds_per_bar * sample_rate))
    mix, used, warnings = _mix_tracks(spec.get("tracks") or [], sample_rate, seconds_per_bar, total)

    mix = _normalise(np.tanh(mix * 0.9), peak=0.94)
    target = _write_container(out_path, mix, sample_rate, bit_depth, container, spec.get("bitrate"))
    return {
        "schema_version": 1,
        "mode": "mixdown",
        "path": str(target),
        "bpm": bpm,
        "bars": bars,
        "sample_rate": sample_rate,
        "bit_depth": None if container in LOSSY else bit_depth,
        "bitrate": LOSSY.get(container) if container in LOSSY else None,
        "container": container,
        "duration_ms": _duration_ms(mix.shape[0], sample_rate),
        "channels": mix.shape[1],
        "clip_count": len(used),
        "tracks": used,
        "warnings": warnings,
    }


def _mix_tracks(
    tracks: "list[dict]",
    sample_rate: int,
    seconds_per_bar: float,
    total: int,
) -> "tuple[np.ndarray, list[dict], list[str]]":
    """Sum the clips of `tracks` onto one buffer, honouring position, length and gains.

    A clip the decoder cannot read is *reported*, never dropped in silence: an arrangement that
    exports without the mp3 the user imported would look successful and be wrong.
    """
    mix = np.zeros((max(1, total), 2), dtype=np.float32)
    used: "list[dict]" = []
    warnings: "list[str]" = []
    for track in tracks:
        track_gain = float(track.get("gain", 1.0))
        if track_gain <= 0:
            continue
        for clip in track.get("clips") or []:
            path = clip.get("path")
            if not path:
                continue
            try:
                samples, rate = read_audio_any(path)
            except Exception as exc:
                warnings.append(f"{Path(str(path)).name}: {exc}")
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
    return mix, used, warnings


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
    if container not in CONTAINERS:
        raise ValueError(f"container must be one of {', '.join(CONTAINERS)}")
    seconds_per_bar = 60.0 / max(1.0, bpm) * 4
    total = int(round(bars * seconds_per_bar * sample_rate))

    stems: "list[dict]" = []
    warnings: "list[str]" = []
    for index, track in enumerate(spec.get("tracks") or []):
        name = str(track.get("name") or f"track{index + 1}")
        mix, used, track_warnings = _mix_tracks([track], sample_rate, seconds_per_bar, total)
        warnings.extend(track_warnings)
        if not used:
            continue
        samples = _normalise(np.tanh(mix * 0.9), peak=0.94)
        safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in name)
        target = directory / f"{base}-{index + 1:02d}-{safe}.{EXTENSIONS.get(container, container)}"
        _write_container(target, samples, sample_rate, bit_depth, container, spec.get("bitrate"))
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
        "bit_depth": None if container in LOSSY else bit_depth,
        "bitrate": LOSSY.get(container) if container in LOSSY else None,
        "container": container,
        "stem_count": len(stems),
        "stems": stems,
        "warnings": warnings,
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

    `parts` entries are `{"name": str, "channel"?: int, "notes": [{"midi", "start_ms", "end_ms",
    "velocity"?}]}`. A channel of 9 (the tenth) is MIDI's drum channel, which is where a rendered
    drum part belongs so the file opens as a kit instead of a piano.
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
        channel = max(0, min(15, int(part.get("channel", 0))))
        for order, note in enumerate(part.get("notes") or []):
            midi = max(0, min(127, int(note.get("midi", 60))))
            velocity = max(1, min(127, int(note.get("velocity", 100))))
            start = to_ticks(float(note.get("start_ms", 0)))
            end = max(start + 1, to_ticks(float(note.get("end_ms", 0))))
            events.append((start, 1, bytes([0x90 | channel, midi, velocity])))
            events.append((end, 0, bytes([0x80 | channel, midi, 64])))
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
