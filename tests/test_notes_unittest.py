"""The deterministic note transforms, measured against hand-checked values.

Every operation here runs on behalf of an agent: the model decides *what* to do ("raise it two
semitones"), this module decides what every note becomes. So the tests are about exact numbers, not
plausibility.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_music_generator.notes import SCALES, scale_pitches, transform_notes


def notes() -> "list[dict]":
    return [
        {"id": "n1", "pitch": 60, "start_ms": 130, "end_ms": 400, "velocity": 100},
        {"id": "n2", "pitch": 63, "start_ms": 520, "end_ms": 900, "velocity": 90},
    ]


class ScaleTests(unittest.TestCase):
    def test_scale_pitches_follow_the_root(self):
        self.assertEqual(scale_pitches("C", "major"), {0, 2, 4, 5, 7, 9, 11})
        self.assertEqual(scale_pitches("A", "minor"), {9, 11, 0, 2, 4, 5, 7})
        # enharmonic spellings are the same key
        self.assertEqual(scale_pitches("Bb", "major"), scale_pitches("A#", "major"))
        self.assertIn("minor_pentatonic", SCALES)

    def test_unknown_keys_are_refused(self):
        with self.assertRaises(ValueError):
            scale_pitches("H", "major")
        with self.assertRaises(ValueError):
            scale_pitches("C", "lydian_dominant")


class TransformTests(unittest.TestCase):
    def test_transpose_moves_every_selected_note(self):
        result = transform_notes({"notes": notes(), "operations": [{"op": "transpose", "semitones": 2}]})
        self.assertEqual([note["pitch"] for note in result["notes"]], [62, 65])
        self.assertEqual(result["diff_summary"], "升高 2 个半音")
        self.assertEqual(result["affected"], ["n1", "n2"])
        # timing and velocity are untouched by a transpose
        self.assertEqual([note["start_ms"] for note in result["notes"]], [130, 520])
        self.assertEqual([note["velocity"] for note in result["notes"]], [100, 90])

    def test_transpose_clamps_at_the_midi_edges(self):
        result = transform_notes(
            {
                "notes": [{"id": "a", "pitch": 126, "start_ms": 0, "end_ms": 100}],
                "operations": [{"op": "transpose", "semitones": 24}],
            }
        )
        self.assertEqual(result["notes"][0]["pitch"], 127)

    def test_only_named_notes_are_touched(self):
        result = transform_notes(
            {
                "notes": notes(),
                "operations": [{"op": "transpose", "semitones": -12, "note_ids": ["n2"]}],
            }
        )
        self.assertEqual([note["pitch"] for note in result["notes"]], [60, 51])
        self.assertEqual(result["affected"], ["n2"])
        self.assertEqual(result["affected_count"], 1)

    def test_quantise_snaps_both_ends_and_keeps_a_length(self):
        result = transform_notes({"notes": notes(), "operations": [{"op": "quantise", "grid_ms": 125}]})
        self.assertEqual(
            [(note["start_ms"], note["end_ms"]) for note in result["notes"]],
            [(125, 375), (500, 875)],
        )
        self.assertEqual(result["diff_summary"], "量化到网格（125 ms）")

    def test_partial_quantise_moves_towards_the_grid(self):
        result = transform_notes(
            {
                "notes": [{"id": "a", "pitch": 60, "start_ms": 100, "end_ms": 300}],
                "operations": [{"op": "quantise", "grid_ms": 200, "strength": 0.5}],
            }
        )
        # 100 is exactly between 0 and 200; half strength lands it half way to the grid line
        self.assertEqual(result["notes"][0]["start_ms"], 50)
        self.assertIn("50%", result["diff_summary"])

    def test_scale_lock_moves_the_nearest_semitone(self):
        # C# (61) and D# (63) are not in C major: 61 → 62, 63 → 64
        result = transform_notes(
            {
                "notes": [
                    {"id": "a", "pitch": 61, "start_ms": 0, "end_ms": 100},
                    {"id": "b", "pitch": 63, "start_ms": 100, "end_ms": 200},
                    {"id": "c", "pitch": 60, "start_ms": 200, "end_ms": 300},
                ],
                "operations": [{"op": "set_scale", "root": "C", "scale": "major"}],
            }
        )
        self.assertEqual([note["pitch"] for note in result["notes"]], [62, 64, 60])
        self.assertEqual(result["diff_summary"], "音阶约束到 C major")

    def test_humanise_is_reproducible_with_a_seed(self):
        spec = {
            "notes": notes(),
            "operations": [{"op": "humanise", "timing_ms": 20, "velocity": 10}],
            "seed": 42,
        }
        first = transform_notes(spec)
        second = transform_notes(spec)
        self.assertEqual(first["notes"], second["notes"])
        different = transform_notes({**spec, "seed": 43})
        self.assertNotEqual(first["notes"], different["notes"])

    def test_shift_keeps_note_lengths_and_never_goes_negative(self):
        result = transform_notes({"notes": notes(), "operations": [{"op": "shift", "ms": -200}]})
        first = result["notes"][0]
        # the start clamps at zero and the note keeps its duration — a shift never changes lengths
        self.assertEqual(first["start_ms"], 0)
        self.assertEqual(first["end_ms"] - first["start_ms"], 270)
        self.assertEqual(result["diff_summary"], "前移 200 ms")

    def test_velocity_scales_and_clamps(self):
        result = transform_notes(
            {"notes": notes(), "operations": [{"op": "velocity", "scale": 0.5, "add": -1000}]}
        )
        self.assertEqual([note["velocity"] for note in result["notes"]], [1, 1])

    def test_operations_chain_in_order_and_the_report_says_so(self):
        result = transform_notes(
            {
                "notes": notes(),
                "operations": [
                    {"op": "transpose", "semitones": 12},
                    {"op": "quantise", "grid_ms": 250},
                ],
            }
        )
        self.assertEqual(result["diff_summary"], "升高 12 个半音；量化到网格（250 ms）")
        self.assertEqual([note["pitch"] for note in result["notes"]], [72, 75])

    def test_the_clip_and_revision_come_back_untouched(self):
        result = transform_notes(
            {
                "clip_id": "clip_042",
                "revision": "r7",
                "notes": notes(),
                "operations": [{"op": "transpose", "semitones": 1}],
            }
        )
        self.assertEqual(result["clip_id"], "clip_042")
        self.assertEqual(result["revision"], "r7")

    def test_nonsense_is_refused_with_a_readable_message(self):
        with self.assertRaises(ValueError):
            transform_notes({"notes": notes(), "operations": []})
        with self.assertRaises(ValueError):
            transform_notes({"notes": notes(), "operations": [{"op": "make_it_better"}]})


if __name__ == "__main__":
    unittest.main()
