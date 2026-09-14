import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_music_generator.synth import NOTE_NAMES, note_name, render_part, render_song

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


def read(path: str) -> "tuple[np.ndarray, int]":
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0, rate


class SynthTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
