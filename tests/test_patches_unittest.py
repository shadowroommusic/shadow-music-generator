"""The patch vocabulary for re-rendering one part, pinned to exact values.

A model chooses *which* knob to turn ("heavier" → `gain_db: +3`); this module decides what that means
to the renderer, so the numbers are asserted rather than described.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_music_generator.patches import REGENERATE_KEYS, apply_part_patch
from shadow_music_generator.synth import regenerate_part


def drums() -> dict:
    return {
        "part": "drums",
        "pattern": {"kick": "x...x...x...x...", "hat": "..x...x...x...x."},
        "swing": 0.1,
        "humanize_ms": 4.0,
        "gain": 1.0,
    }


def bass() -> dict:
    return {
        "part": "bass",
        "wave": "saw",
        "cutoff": 0.4,
        "notes": [
            {"midi": 33, "start": 0, "length": 1},
            {"midi": 36, "start": 2, "length": 1},
        ],
    }


class PatchTests(unittest.TestCase):
    def test_gain_is_decibels_turned_into_a_multiplier(self):
        patched, described = apply_part_patch(drums(), {"gain_db": 3})
        self.assertAlmostEqual(patched["gain"], 1.4125, places=3)
        self.assertEqual(described, ["音量 +3.0 dB"])
        quiet, _ = apply_part_patch(drums(), {"gain_db": -6})
        self.assertAlmostEqual(quiet["gain"], 0.5012, places=3)

    def test_swing_and_humanise_accumulate_and_clamp(self):
        patched, _ = apply_part_patch(drums(), {"swing": 0.05, "humanize_ms": 10})
        self.assertAlmostEqual(patched["swing"], 0.15)
        self.assertAlmostEqual(patched["humanize_ms"], 14.0)
        capped, _ = apply_part_patch(drums(), {"swing": 5, "humanize_ms": 100})
        self.assertLessEqual(capped["swing"], 0.4)
        self.assertLessEqual(capped["humanize_ms"], 40.0)

    def test_transpose_moves_every_note_and_clamps(self):
        patched, described = apply_part_patch(bass(), {"transpose": 12})
        self.assertEqual([note["midi"] for note in patched["notes"]], [45, 48])
        self.assertEqual(described, ["移调 +12 半音"])
        high = {"part": "lead", "notes": [{"midi": 126, "start": 0, "length": 1}]}
        self.assertEqual(apply_part_patch(high, {"transpose": 24})[0]["notes"][0]["midi"], 127)

    def test_velocity_scale_multiplies_note_gains(self):
        patched, described = apply_part_patch(bass(), {"velocity_scale": 1.5})
        self.assertEqual([note["gain"] for note in patched["notes"]], [1.5, 1.5])
        self.assertEqual(described, ["力度 ×1.5"])

    def test_pattern_patches_merge_and_remove_voices(self):
        patched, _ = apply_part_patch(drums(), {"pattern": {"clap": "....x.......x..."}})
        self.assertIn("clap", patched["pattern"])
        self.assertIn("kick", patched["pattern"])
        trimmed, described = apply_part_patch(patched, {"remove_voices": ["hat"]})
        self.assertNotIn("hat", trimmed["pattern"])
        self.assertEqual(described, ["去掉 hat"])

    def test_a_pattern_patch_needs_a_drum_part(self):
        with self.assertRaises(ValueError):
            apply_part_patch(bass(), {"pattern": {"kick": "x..."}})

    def test_transposing_a_drum_part_is_refused(self):
        with self.assertRaises(ValueError):
            apply_part_patch(drums(), {"transpose": 2})

    def test_unknown_keys_and_empty_patches_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            apply_part_patch(drums(), {"make_it_better": 1})
        self.assertIn("make_it_better", str(caught.exception))
        with self.assertRaises(ValueError):
            apply_part_patch(drums(), {})
        self.assertEqual("cutoff" in REGENERATE_KEYS, True)

    def test_the_original_spec_is_never_mutated(self):
        original = drums()
        apply_part_patch(original, {"gain_db": 6, "remove_voices": ["hat"]})
        self.assertEqual(original["gain"], 1.0)
        self.assertIn("hat", original["pattern"])


class RegenerateTests(unittest.TestCase):
    def test_regenerating_one_part_writes_a_new_file_and_reports_the_plan(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            first = regenerate_part(
                {
                    "out_path": str(Path(tmp) / "drums-a.wav"),
                    "bpm": 122,
                    "bars": 4,
                    "part": drums(),
                    "patch": {"gain_db": 0},
                }
            )
            second = regenerate_part(
                {
                    "out_path": str(Path(tmp) / "drums-b.wav"),
                    "bpm": 122,
                    "bars": 4,
                    "part": drums(),
                    "patch": {"gain_db": 6},
                }
            )
            self.assertEqual(first["spec"]["gain"], 1.0)
            self.assertGreater(second["spec"]["gain"], first["spec"]["gain"])
            self.assertEqual(second["diff_summary"], "音量 +6.0 dB")
            self.assertEqual(second["bars"], 4)
            # the same plan renders the same audio: a re-render is reproducible
            again = regenerate_part(
                {
                    "out_path": str(Path(tmp) / "drums-c.wav"),
                    "bpm": 122,
                    "bars": 4,
                    "part": drums(),
                    "patch": {"gain_db": 6},
                }
            )
            from shadow_music_generator.synth import read_audio

            a, _ = read_audio(second["path"])
            b, _ = read_audio(again["path"])
            self.assertTrue(np.allclose(a, b), "the same spec must render the same audio")

    def test_a_gain_patch_is_actually_audible(self):
        """Stems are normalised, so the patch has to ride on top of that normalisation.

        Measured before this rule existed: +5 dB came back 1.048× louder (the normaliser had eaten it),
        i.e. "heavier" changed nothing a user could hear.
        """
        import tempfile

        from shadow_music_generator.synth import read_audio

        def rms(path: str) -> float:
            data, _ = read_audio(path)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            return float(np.sqrt(np.mean(mono**2)))

        with tempfile.TemporaryDirectory() as tmp:
            base = regenerate_part(
                {"out_path": f"{tmp}/base.wav", "bpm": 124, "bars": 4, "part": drums(), "patch": {"swing": 0.0}}
            )
            loud = regenerate_part(
                {"out_path": f"{tmp}/loud.wav", "bpm": 124, "bars": 4, "part": drums(), "patch": {"gain_db": 5}}
            )
            quiet = regenerate_part(
                {"out_path": f"{tmp}/quiet.wav", "bpm": 124, "bars": 4, "part": drums(), "patch": {"gain_db": -5}}
            )
            self.assertGreater(rms(loud["path"]) / rms(base["path"]), 1.3)
            self.assertLess(rms(quiet["path"]) / rms(base["path"]), 0.7)
            self.assertAlmostEqual(loud["gain_delta"], 1.7783, places=3)

    def test_a_missing_part_spec_is_refused(self):
        with self.assertRaises(ValueError):
            regenerate_part({"out_path": "/tmp/x.wav", "patch": {"gain_db": 1}})


if __name__ == "__main__":
    unittest.main()
