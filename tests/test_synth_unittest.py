import struct
import sys
from collections import Counter
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_music_generator.synth import (
    NOTE_NAMES,
    _encode_lossy,
    export_midi,
    export_stems,
    mix_arrangement,
    note_name,
    read_audio,
    read_audio_any,
    read_midi_notes,
    render_part,
    render_song,
    write_audio,
    write_midi_multitrack,
)


def decoder_available() -> bool:
    """Whether this machine can decode/encode AAC — those tests skip cleanly when it cannot."""
    import shutil

    return shutil.which("ffmpeg") is not None or shutil.which("afconvert") is not None

SKETCH = {
    "bpm": 128,
    "bars": 2,
    "parts": [
        {
            "part": "drums",
            "pattern": {"kick": "x...x...x...x...", "hat": "..x...x...x...x.", "clap": "....x.......x..."},
        },
        {
            "part": "bass",
            "wave": "saw",
            "cutoff": 0.35,
            "notes": [
                {"midi": 36, "start": 0, "length": 1},
                {"midi": 39, "start": 2, "length": 1},
                {"midi": 43, "start": 3, "length": 1},
            ],
        },
        {
            "part": "chords",
            "wave": "triangle",
            "attack": 0.1,
            "notes": [{"midi": midi, "start": 0, "length": 4} for midi in (60, 63, 67)],
        },
    ],
}


#: One readable clip for the arrangement tests (written once, at import time).
CLIP_WAV = str(Path(tempfile.gettempdir()) / "shadow-test-clip.wav")
if not Path(CLIP_WAV).exists():
    write_audio(Path(CLIP_WAV), (np.sin(np.arange(22050) / 25.0) * 0.5).astype("float32"), 44100)


def read(path: str) -> "tuple[np.ndarray, int]":
    """Mono view of a file (stems are stereo now; level checks do not care)."""
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return (data.reshape(-1, channels).mean(axis=1) if channels > 1 else data), rate


def _onset_ms(samples: "np.ndarray", rate: int, window_ms: int = 10) -> "list[int]":
    """Tiny RMS-flux onset finder, so the swing test does not depend on another repo."""
    step = int(rate * window_ms / 1000)
    count = samples.size // step
    energy = np.sqrt(np.mean(samples[: count * step].reshape(count, step) ** 2, axis=1))
    flux = np.maximum(np.diff(energy, prepend=energy[:1]), 0.0)
    gate = 0.25 * float(flux.max() or 0.0)
    hits: "list[int]" = []
    last = -10
    for index in range(1, count - 1):
        if flux[index] < gate or flux[index] < flux[index - 1] or flux[index] < flux[index + 1]:
            continue
        if index - last < 3:
            continue
        hits.append(int(round(index * window_ms)))
        last = index
    return hits


def read_stereo(path: str) -> "tuple[np.ndarray, int]":
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, channels), rate


class SynthTests(unittest.TestCase):
    def test_a_drum_pattern_covers_every_bar_of_the_part(self):
        """The bug that made every rendered beat one bar of audio: rows were placed once.

        Measured on the workbench's own stems before the fix (`night-drive-drums.wav`, 8 bars):
        bar 1 = 0.108 RMS, bars 2–8 = 0.000.
        """
        with tempfile.TemporaryDirectory() as tmp:
            result = render_part(
                {
                    "out_path": str(Path(tmp) / "beat.wav"),
                    "part": "drums",
                    "bpm": 120,
                    "bars": 4,
                    "pattern": {"kick": "x...x...x...x...", "hat": "..x...x...x...x."},
                }
            )
            samples, rate = read(result["path"])
            per_bar = samples.size // 4
            levels = [
                float(np.sqrt(np.mean(samples[index * per_bar : (index + 1) * per_bar] ** 2)))
                for index in range(4)
            ]
            for index, level in enumerate(levels, start=1):
                self.assertGreater(level, 0.05, f"bar {index} is silent: {levels}")
            self.assertLess(max(levels) - min(levels), 0.02, f"bars differ: {levels}")

    def test_a_multi_bar_row_tiles_as_its_own_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_part(
                {
                    "out_path": str(Path(tmp) / "two-bar.wav"),
                    "part": "drums",
                    "bpm": 120,
                    "bars": 4,
                    "steps": 32,
                    "pattern": {
                        # a two-bar block: a kick on every beat, an extra hit only in bar 2
                        "kick": "x...x...x...x..." + "x...x...x..x.x..",
                    },
                }
            )
            samples, _ = read(result["path"])
            per_bar = samples.size // 4
            levels = [
                float(np.sqrt(np.mean(samples[index * per_bar : (index + 1) * per_bar] ** 2)))
                for index in range(4)
            ]
            self.assertGreater(min(levels), 0.02, f"a bar of the block is empty: {levels}")
            # bars 1/3 came from the same half of the block, bars 2/4 from the other
            self.assertAlmostEqual(levels[0], levels[2], delta=0.01)
            self.assertAlmostEqual(levels[1], levels[3], delta=0.01)

    def test_written_notes_repeat_when_asked(self):
        notes = [{"midi": 33, "start": 0, "length": 0.75}, {"midi": 36, "start": 3, "length": 0.5}]
        with tempfile.TemporaryDirectory() as tmp:
            once = render_part(
                {"out_path": str(Path(tmp) / "once.wav"), "part": "bass", "bpm": 120, "bars": 4, "notes": notes}
            )
            twice = render_part(
                {
                    "out_path": str(Path(tmp) / "twice.wav"),
                    "part": "bass",
                    "bpm": 120,
                    "bars": 4,
                    "repeat": True,
                    "notes": notes,
                }
            )
            plain, _ = read(once["path"])
            repeated, _ = read(twice["path"])
            per_bar = plain.size // 4
            third = float(np.sqrt(np.mean(plain[2 * per_bar : 3 * per_bar] ** 2)))
            self.assertLess(third, 0.01, "without repeat the third bar should hold what was written")
            for index in range(4):
                level = float(np.sqrt(np.mean(repeated[index * per_bar : (index + 1) * per_bar] ** 2)))
                self.assertGreater(level, 0.02, f"bar {index + 1} of the repeat is empty")

    def test_render_song_writes_a_mix_and_one_stem_per_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_song({**SKETCH, "out_path": str(Path(tmp) / "sketch.wav")})
            self.assertEqual([part["part"] for part in result["parts"]], ["drums", "bass", "chords"])
            mix, rate = read(result["path"])
            # 2 bars of 4/4 at 128 BPM = 8 beats = 3.75 s
            self.assertEqual(rate, 44100)
            self.assertAlmostEqual(mix.size / rate, 3.75, places=2)
            self.assertGreater(float(np.sqrt(np.mean(mix**2))), 0.05)
            self.assertLessEqual(float(np.abs(mix).max()), 1.0)
            for part in result["parts"]:
                stem, _ = read(part["path"])
                self.assertEqual(stem.size, mix.size)
                self.assertGreater(float(np.abs(stem).max()), 0.1)

    def test_parts_are_distinct_and_mixed_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_song({**SKETCH, "out_path": str(Path(tmp) / "sketch.wav")})
            stems = {part["part"]: read(part["path"])[0] for part in result["parts"]}
            self.assertFalse(np.allclose(stems["drums"], stems["bass"]))
            mix, _ = read(result["path"])
            # the mix carries energy from every stem
            for name, stem in stems.items():
                correlation = float(np.dot(mix, stem) / (np.linalg.norm(mix) * np.linalg.norm(stem) + 1e-9))
                self.assertGreater(correlation, 0.1, f"{name} is missing from the mix")

    def test_sections_and_segments_render_a_song_with_dynamics(self):
        """The shape a natural-language song request maps to: intro → drop → break → drop 2."""
        arrangement = {
            "out_path": "song.wav",
            "bpm": 124,
            "sections": [
                {"name": "intro", "bars": 2},
                {"name": "drop", "bars": 4},
                {"name": "break", "bars": 2},
                {"name": "drop 2", "bars": 2},
            ],
            "parts": [
                {
                    "part": "drums",
                    "segments": [
                        {"bars": 2, "pattern": {"hat": "..x...x...x...x."}},
                        {
                            "bars": 4,
                            "pattern": {
                                "kick": "x...x...x...x...",
                                "clap": "....x.......x...",
                                "hat": "..x...x...x...x.",
                            },
                        },
                        {"bars": 2, "gain": 0},
                        {
                            "bars": 2,
                            "pattern": {
                                "kick": "x...x...x...x...",
                                "clap": "....x.......x...",
                                "hat": "..x...x...x...x.",
                            },
                        },
                    ],
                },
                {
                    "part": "bass",
                    "wave": "saw",
                    "repeat": True,
                    "segments": [
                        {"bars": 2, "gain": 0},
                        {"bars": 4, "notes": [{"midi": 33, "start": 0, "length": 0.75}]},
                        {"bars": 2, "gain": 0},
                        {"bars": 2, "notes": [{"midi": 33, "start": 0, "length": 0.75}]},
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = render_song({**arrangement, "out_path": str(Path(tmp) / "song.wav")})
            # the sections set the song's length: 2 + 4 + 2 + 2 bars
            self.assertEqual([entry["name"] for entry in result["sections"]], ["intro", "drop", "break", "drop 2"])
            self.assertEqual([entry["start_bar"] for entry in result["sections"]], [0, 2, 6, 8])
            self.assertEqual(result["bars"], 10)
            # 10 bars of 4/4 at 124 BPM
            self.assertAlmostEqual(result["duration_ms"] / 1000, 10 * 4 * 60 / 124, places=2)
            for part in result["parts"]:
                self.assertEqual(len(part["segments"]), 4)
                self.assertEqual(part["duration_ms"], result["duration_ms"])

            mix, rate = read(result["path"])
            per_bar = mix.size // 10
            level = lambda index: float(np.sqrt(np.mean(mix[index * per_bar : (index + 1) * per_bar] ** 2)))
            intro = sum(level(index) for index in range(0, 2)) / 2
            drop = sum(level(index) for index in range(2, 6)) / 4
            silence = max(level(index) for index in range(6, 8))
            second = sum(level(index) for index in range(8, 10)) / 2
            self.assertGreater(intro, 0.005, "the intro should be there, just quieter")
            self.assertGreater(drop, intro * 3, f"the drop ({drop}) must lift the intro ({intro})")
            self.assertLess(silence, 0.001, f"the break should be empty, got {silence}")
            self.assertAlmostEqual(second, drop, delta=0.05)
            # the stems line up with the mix, so the Studio's meters agree with the timeline
            for part in result["parts"]:
                stem, _ = read(part["path"])
                self.assertEqual(stem.size, mix.size)
            drums, _ = read(result["parts"][0]["path"])
            drums_break = max(float(np.sqrt(np.mean(drums[index * per_bar : (index + 1) * per_bar] ** 2))) for index in (6, 7))
            self.assertLess(drums_break, 0.001, "the drums are out for the break")

    def test_a_render_can_hand_back_its_own_midi(self):
        """`midi_path` writes the plan that was played — not a transcription of the audio."""
        arrangement = {
            "out_path": "song.wav",
            "midi_path": "song.mid",
            "bpm": 120,
            "sections": [{"name": "intro", "bars": 1}, {"name": "drop", "bars": 2}],
            "parts": [
                {
                    "part": "drums",
                    "segments": [
                        {"bars": 1, "pattern": {"hat": "..x...x...x...x."}},
                        {"bars": 2, "pattern": {"kick": "x...x...x...x...", "hat": "..x...x...x...x."}},
                    ],
                },
                {
                    "part": "bass",
                    "wave": "saw",
                    "repeat": True,
                    "segments": [
                        {"bars": 1, "gain": 0},
                        {"bars": 2, "notes": [{"midi": 33, "start": 0, "length": 1}]},
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = render_song(
                {
                    **arrangement,
                    "out_path": str(Path(tmp) / "song.wav"),
                    "midi_path": str(Path(tmp) / "song.mid"),
                }
            )
            self.assertIsNotNone(result["midi_path"])
            notes, tempo = read_midi_notes(result["midi_path"])
            self.assertAlmostEqual(tempo or 0, 120, places=1)
            bar_ms = 4 * 60_000 / 120
            drum_bars = Counter(
                int(note["start_ms"] // bar_ms) for note in notes if note["midi"] in (36, 42)
            )
            # intro: hats only (four per bar, on the offbeat sixteenths)
            self.assertEqual(drum_bars[0], 4)
            # drop: the same hats plus a kick on every beat
            self.assertEqual(drum_bars[1], 8)
            self.assertEqual(drum_bars[2], 8)
            # kick and hat are General-MIDI kit pieces, so the file opens as a kit
            self.assertIn(36, [note["midi"] for note in notes])
            self.assertIn(42, [note["midi"] for note in notes])
            # the bass only plays from the drop, one note per bar
            self.assertEqual(sum(1 for note in notes if note["midi"] == 33), 2)
            self.assertTrue(all(note["start_ms"] >= bar_ms - 1 for note in notes if note["midi"] == 33))

    def test_read_audio_handles_extensible_wav_headers(self):
        """afconvert and DAWs write `WAVE_FORMAT_EXTENSIBLE`, which Python's `wave` refuses."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain.wav"
            write_audio(plain, np.sin(np.arange(4410) / 20.0).astype(np.float32) * 0.5, 44100)
            raw = plain.read_bytes()
            self.assertEqual(raw[12:16], b"fmt ")
            fmt_size = struct.unpack("<I", raw[16:20])[0]
            body = raw[20 : 20 + fmt_size]
            channels = struct.unpack("<H", body[2:4])[0]
            rate = struct.unpack("<I", body[4:8])[0]
            bits = struct.unpack("<H", body[14:16])[0]
            extended = (
                struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
                + struct.pack("<H", 22)  # cbSize
                + struct.pack("<HI", bits, 1)  # wValidBitsPerSample, dwChannelMask
                # SubFormat: KSDATAFORMAT_SUBTYPE_PCM, 00000001-0000-0010-8000-00aa00389b71
                + struct.pack("<IHH", 1, 0, 0x0010)
                + bytes([0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71])
            )
            rest = raw[20 + fmt_size :]
            rebuilt = (
                b"RIFF"
                + struct.pack("<I", len(raw))
                + b"WAVE"
                + b"fmt "
                + struct.pack("<I", len(extended))
                + extended
                + rest
            )
            target = Path(tmp) / "extensible.wav"
            target.write_bytes(rebuilt)
            with self.assertRaises(Exception):
                wave.open(str(target), "rb")
            samples, read_rate = read_audio(target)
            self.assertEqual(read_rate, 44100)
            self.assertGreater(float(np.abs(samples).max()), 0.4)

    def test_a_clip_that_cannot_be_read_is_reported_not_dropped(self):
        spec = {
            "out_path": "mix.wav",
            "bpm": 120,
            "bars": 2,
            "tracks": [
                {"name": "ok", "clips": [{"path": CLIP_WAV, "start": 0, "bars": 2}]},
                {"name": "missing", "clips": [{"path": "/tmp/definitely-not-here-9f3a.flac", "start": 0, "bars": 2}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = mix_arrangement({**spec, "out_path": str(Path(tmp) / "mix.wav")})
            self.assertEqual(result["clip_count"], 1)
            self.assertTrue(result["warnings"], "the unreadable clip has to be reported")
            self.assertIn("definitely-not-here", result["warnings"][0])

    @unittest.skipUnless(decoder_available(), "needs ffmpeg or afconvert")
    def test_a_non_wav_clip_is_decoded_into_the_mix(self):
        """An imported mp3/m4a has to be in the mixdown, not silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tone.wav"
            write_audio(source, np.sin(np.arange(44100) / 30.0).astype(np.float32) * 0.6, 44100)
            compressed = _encode_lossy(source, Path(tmp) / "tone.m4a", bitrate=192000)
            result = mix_arrangement(
                {
                    "out_path": str(Path(tmp) / "mix.wav"),
                    "bpm": 120,
                    "bars": 1,
                    "tracks": [{"name": "imported", "clips": [{"path": str(compressed), "start": 0, "bars": 1}]}],
                }
            )
            self.assertEqual(result["clip_count"], 1, result["warnings"])
            mix, _ = read_audio(result["path"])
            self.assertGreater(float(np.sqrt(np.mean(mix**2))), 0.05)

    @unittest.skipUnless(decoder_available(), "needs ffmpeg or afconvert")
    def test_m4a_export_writes_a_playable_aac_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tone.wav"
            write_audio(source, np.sin(np.arange(44100) / 30.0).astype(np.float32) * 0.6, 44100)
            result = mix_arrangement(
                {
                    "out_path": str(Path(tmp) / "mix.m4a"),
                    "bpm": 120,
                    "bars": 1,
                    "container": "m4a",
                    "tracks": [{"name": "tone", "clips": [{"path": str(source), "start": 0, "bars": 1}]}],
                }
            )
            self.assertTrue(result["path"].endswith(".m4a"))
            self.assertIsNone(result["bit_depth"], "AAC has no bit depth")
            decoded, rate = read_audio_any(result["path"])
            self.assertEqual(rate, 44100)
            self.assertGreater(float(np.sqrt(np.mean(decoded**2))), 0.05)

    def test_render_part_reports_bars_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = render_part(
                {
                    "out_path": str(Path(tmp) / "hats.wav"),
                    "part": "drums",
                    "bpm": 120,
                    "bars": 4,
                    "pattern": {"hat": "x.x.x.x.x.x.x.x."},
                }
            )
            samples, rate = read(result["path"])
            self.assertEqual(result["bars"], 4)
            self.assertAlmostEqual(samples.size / rate, 8.0, places=2)
            self.assertEqual(result["duration_ms"], 8000)

    def test_empty_spec_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render_song({"out_path": str(Path(tmp) / "x.wav"), "parts": []})
            with self.assertRaises(ValueError):
                render_part({"part": "bass", "notes": []})

    def test_note_names(self):
        self.assertEqual(note_name(60), "C4")
        self.assertEqual(note_name(36), "C2")
        self.assertEqual(NOTE_NAMES[69 % 12], "A")


class ExportTests(unittest.TestCase):
    def test_stems_write_one_file_per_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            song = render_song({**SKETCH, "out_path": str(Path(tmp) / "sketch.wav")})
            result = export_stems(
                {
                    "out_dir": str(Path(tmp) / "stems"),
                    "name": "sketch",
                    "bpm": SKETCH["bpm"],
                    "bars": SKETCH["bars"],
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "tracks": [
                        {"name": part["part"], "clips": [{"path": part["path"], "start": 0}]}
                        for part in song["parts"]
                    ],
                }
            )
            self.assertEqual(result["stem_count"], len(song["parts"]))
            self.assertEqual(result["sample_rate"], 48000)
            for stem in result["stems"]:
                samples, rate = read(stem["path"])
                self.assertEqual(rate, 48000)
                self.assertGreater(float(np.abs(samples).max()), 0.1)
                self.assertEqual(stem["clip_count"], 1)

    def test_stems_skip_empty_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_stems(
                    {
                        "out_dir": str(Path(tmp) / "stems"),
                        "tracks": [{"name": "空", "clips": []}],
                    }
                )

    def test_midi_export_round_trips_through_the_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_midi(
                {
                    "out_path": str(Path(tmp) / "parts.mid"),
                    "bpm": 120,
                    "bars": 4,
                    "tracks": [
                        {
                            "name": "bass",
                            "clips": [
                                {
                                    "start": 1,
                                    "notes": [
                                        {"midi": 36, "start_ms": 0, "end_ms": 400},
                                        {"midi": 43, "start_ms": 500, "end_ms": 900},
                                    ],
                                }
                            ],
                        },
                        {
                            "name": "lead",
                            "clips": [{"start": 0, "notes": [{"midi": 72, "start_ms": 200, "end_ms": 600}]}],
                        },
                    ],
                }
            )
            self.assertEqual(result["track_count"], 2)
            self.assertEqual(result["note_count"], 3)
            notes, tempo = read_midi_notes(result["path"])
            self.assertAlmostEqual(tempo or 0, 120.0, places=1)
            # one bar at 120 BPM = 2000 ms, so the bass notes land there
            self.assertEqual([note["midi"] for note in notes], [72, 36, 43])
            self.assertAlmostEqual(notes[0]["start_ms"], 200, delta=5)
            self.assertAlmostEqual(notes[1]["start_ms"], 2000, delta=5)
            self.assertAlmostEqual(notes[2]["start_ms"], 2500, delta=5)

    def test_midi_export_reports_audio_only_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            song = render_song({**SKETCH, "out_path": str(Path(tmp) / "sketch.wav")})
            result = export_midi(
                {
                    "out_path": str(Path(tmp) / "mixed.mid"),
                    "bpm": 120,
                    "tracks": [
                        {"name": "drums", "clips": [{"path": song["parts"][0]["path"], "start": 0}]},
                        {"name": "lead", "clips": [{"notes": [{"midi": 60, "start_ms": 0, "end_ms": 300}]}]},
                    ],
                }
            )
            self.assertEqual(result["track_count"], 1)
            self.assertTrue(any("audio" in entry for entry in result["skipped"]))

    def test_multitrack_midi_is_type_one_with_a_tempo_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = write_midi_multitrack(
                [{"name": "a", "notes": [{"midi": 60, "start_ms": 0, "end_ms": 500}]}],
                Path(tmp) / "one.mid",
                bpm=124,
            )
            data = target.read_bytes()
            self.assertEqual(data[:4], b"MThd")
            self.assertEqual(struct.unpack(">HHH", data[8:14]), (1, 2, 480))  # type 1, tempo + 1 part
            _notes, tempo = read_midi_notes(target)
            self.assertAlmostEqual(tempo or 0, 124.0, places=1)


class SpatialTests(unittest.TestCase):
    """Pan, reverb, delay, swing and humanize — the difference between a demo and a mix."""

    def test_pan_places_a_part_in_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = render_part(
                {"out_path": str(Path(tmp) / "left.wav"), "part": "bass", "bpm": 120, "bars": 1,
                 "wave": "sine", "pan": -1.0, "reverb": 0.0,
                 "notes": [{"midi": 40, "start": 0, "length": 2}]}
            )
            right = render_part(
                {"out_path": str(Path(tmp) / "right.wav"), "part": "bass", "bpm": 120, "bars": 1,
                 "wave": "sine", "pan": 1.0, "reverb": 0.0,
                 "notes": [{"midi": 40, "start": 0, "length": 2}]}
            )
            left_frames, _ = read_stereo(left["path"])
            right_frames, _ = read_stereo(right["path"])
            self.assertGreater(float(np.abs(left_frames[:, 0]).mean()), 10 * float(np.abs(left_frames[:, 1]).mean()))
            self.assertGreater(float(np.abs(right_frames[:, 1]).mean()), 10 * float(np.abs(right_frames[:, 0]).mean()))
            self.assertEqual(left["channels"], 2)

    def test_reverb_adds_a_tail_after_the_last_note(self):
        spec = {
            "part": "chords",
            "bpm": 120,
            "bars": 1,
            "wave": "triangle",
            "notes": [{"midi": 60, "start": 0, "length": 1}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dry = render_part({"out_path": str(Path(tmp) / "dry.wav"), **spec, "reverb": 0.0})
            wet = render_part({"out_path": str(Path(tmp) / "wet.wav"), **spec, "reverb": 0.5, "reverb_decay": 1.4})
            dry_frames, rate = read_stereo(dry["path"])
            wet_frames, _ = read_stereo(wet["path"])
            # the note lasts 0.5 s; look at the 0.8 s after it, where only reverb can be
            window = slice(int(0.6 * rate), int(1.3 * rate))
            dry_tail = float(np.sqrt(np.mean(dry_frames[window] ** 2)))
            wet_tail = float(np.sqrt(np.mean(wet_frames[window] ** 2)))
            self.assertLess(dry_tail, 1e-4)
            self.assertGreater(wet_tail, 20 * max(dry_tail, 1e-6))

    def test_swing_delays_the_offbeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            straight = render_part(
                {"out_path": str(Path(tmp) / "straight.wav"), "part": "drums", "bpm": 120, "bars": 1,
                 "pattern": {"hat": "..x...x...x...x."}, "swing": 0.0, "reverb": 0.0}
            )
            swung = render_part(
                {"out_path": str(Path(tmp) / "swung.wav"), "part": "drums", "bpm": 120, "bars": 1,
                 "pattern": {"hat": "..x...x...x...x."}, "swing": 0.5, "reverb": 0.0}
            )
            straight_samples, rate = read(straight["path"])
            swung_samples, _ = read(swung["path"])
            straight_hits = _onset_ms(straight_samples, rate)
            swung_hits = _onset_ms(swung_samples, rate)
            self.assertEqual(len(straight_hits), len(swung_hits))
            # hat hits sit on the offbeats (steps 2, 6, 10, 14 → 250/750/1250/1750 ms at 120 BPM)
            self.assertAlmostEqual(straight_hits[0], 250, delta=25)
            # …and swing pushes exactly those offbeats later
            self.assertGreater(swung_hits[0], straight_hits[0] + 30)

    def test_song_mix_is_stereo_and_decorrelated(self):
        with tempfile.TemporaryDirectory() as tmp:
            song = render_song(
                {
                    "out_path": str(Path(tmp) / "wide.wav"),
                    "bpm": 124,
                    "bars": 2,
                    "parts": [
                        {"part": "drums", "pan": -0.2, "reverb": 0.2,
                         "pattern": {"kick": "x...x...x...x...", "hat": "..x...x...x...x."}},
                        {"part": "bass", "pan": 0.15, "wave": "saw", "cutoff": 0.35,
                         "notes": [{"midi": 36, "start": 0, "length": 1}]},
                    ],
                }
            )
            frames, _ = read_stereo(song["path"])
            self.assertEqual(frames.shape[1], 2)
            correlation = float(np.corrcoef(frames[:, 0], frames[:, 1])[0, 1])
            self.assertLess(correlation, 0.999)
            self.assertGreater(float(np.abs(frames).max()), 0.3)


if __name__ == "__main__":
    unittest.main()
