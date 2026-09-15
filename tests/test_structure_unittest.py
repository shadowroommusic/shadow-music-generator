"""Arrangement surgery: insert, repeat, delete — pinned to exact positions.

These are the numbers a user sees on the timeline, so the tests assert them rather than describing
the intent. The arrangement used throughout is the one the workbench sends: sections and the clips
that sit on them.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_music_generator.structure import arrange


def summary() -> dict:
    return {
        "bars": 16,
        "sections": [
            {"id": "a", "name": "intro", "start": 0, "bars": 4},
            {"id": "b", "name": "drop", "start": 4, "bars": 8},
            {"id": "c", "name": "outro", "start": 12, "bars": 4},
        ],
        "tracks": [
            {
                "name": "鼓",
                "clips": [
                    {"id": "d1", "start": 0, "length": 4},
                    {"id": "d2", "start": 4, "length": 8},
                    {"id": "d3", "start": 12, "length": 4},
                ],
            },
            {"name": "贝斯", "clips": [{"id": "b1", "start": 4, "length": 8}]},
        ],
    }


class ArrangeTests(unittest.TestCase):
    def test_insert_bars_pushes_the_arrangement_right(self):
        result = arrange(
            {**summary(), "operations": [{"op": "insert_bars", "at_bar": 12, "count": 8, "name": "breakdown"}]}
        )
        self.assertEqual(result["bars"], 24)
        self.assertEqual(
            [(section["name"], section["start"], section["bars"]) for section in result["sections"]],
            [("intro", 0, 4), ("drop", 4, 8), ("breakdown", 12, 8), ("outro", 20, 4)],
        )
        drums = [clip for clip in result["clips"] if clip["track"] == "鼓"]
        self.assertEqual([clip["start"] for clip in drums], [0, 4, 20])
        bass = [clip for clip in result["clips"] if clip["track"] == "贝斯"]
        self.assertEqual([clip["start"] for clip in bass], [4])
        self.assertEqual(result["diff_summary"], "在第 13 小节插入 8 小节")

    def test_inserting_inside_a_section_grows_it(self):
        result = arrange({**summary(), "operations": [{"op": "insert_bars", "at_bar": 8, "count": 4}]})
        self.assertEqual(
            [(section["name"], section["start"], section["bars"]) for section in result["sections"]],
            [("intro", 0, 4), ("drop", 4, 12), ("outro", 16, 4)],
        )
        self.assertEqual(result["bars"], 20)

    def test_duplicate_section_copies_clips_and_points_at_their_source(self):
        result = arrange({**summary(), "operations": [{"op": "duplicate_section", "section": "drop"}]})
        self.assertEqual(result["bars"], 24)
        drums = [clip for clip in result["clips"] if clip["track"] == "鼓"]
        self.assertEqual([clip["start"] for clip in drums], [0, 4, 12, 20])
        copy = next(clip for clip in drums if clip["start"] == 12)
        self.assertEqual(copy["copy_of"], "d2", "the UI clones the source clip, audio file and all")
        self.assertEqual(
            [section["name"] for section in result["sections"]],
            ["intro", "drop", "drop", "outro"],
        )
        self.assertEqual(result["diff_summary"], "复制「drop」段落（2 个片段，8 小节）")

    def test_repeat_last_section_needs_no_name(self):
        result = arrange({**summary(), "operations": [{"op": "repeat_last_section"}]})
        self.assertEqual(result["bars"], 20)
        self.assertEqual([section["start"] for section in result["sections"]], [0, 4, 12, 16])
        self.assertEqual(
            [clip["start"] for clip in result["clips"] if clip["track"] == "鼓"],
            [0, 4, 12, 16],
        )

    def test_remove_section_deletes_its_clips_and_pulls_the_rest_left(self):
        result = arrange({**summary(), "operations": [{"op": "remove_section", "section": "DROP"}]})
        self.assertEqual(result["bars"], 8)
        self.assertEqual([(s["name"], s["start"]) for s in result["sections"]], [("intro", 0), ("outro", 4)])
        self.assertEqual([clip["start"] for clip in result["clips"] if clip["track"] == "鼓"], [0, 4])
        self.assertEqual([clip for clip in result["clips"] if clip["track"] == "贝斯"], [])
        self.assertEqual(result["diff_summary"], "删除「drop」段落（2 个片段）")

    def test_operations_chain_in_order(self):
        result = arrange(
            {
                **summary(),
                "operations": [
                    {"op": "insert_bars", "at_bar": 16, "count": 8, "name": "breakdown"},
                    {"op": "duplicate_section", "section": "breakdown"},
                ],
            }
        )
        self.assertEqual(result["bars"], 32)
        self.assertEqual(
            [(section["name"], section["start"]) for section in result["sections"]],
            [("intro", 0), ("drop", 4), ("outro", 12), ("breakdown", 16), ("breakdown", 24)],
        )
        self.assertIn("插入 8 小节", result["diff_summary"])
        self.assertIn("复制「breakdown」", result["diff_summary"])

    def test_bad_input_is_refused_with_a_readable_message(self):
        with self.assertRaises(ValueError):
            arrange({**summary(), "operations": []})
        with self.assertRaises(ValueError):
            arrange({**summary(), "operations": [{"op": "make_it_longer"}]})
        with self.assertRaises(ValueError) as caught:
            arrange({**summary(), "operations": [{"op": "duplicate_section", "section": "chorus"}]})
        self.assertIn("chorus", str(caught.exception))

    def test_the_revision_comes_back_untouched(self):
        result = arrange(
            {**summary(), "revision": "r9", "operations": [{"op": "insert_bars", "at_bar": 0, "count": 4}]}
        )
        self.assertEqual(result["revision"], "r9")
        self.assertEqual(result["clips"][0]["start"], 4, "everything moved right of the insertion")


if __name__ == "__main__":
    unittest.main()
