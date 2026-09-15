"""Arrangement surgery: inserting bars, repeating a section, cutting one out.

The workbench owns the arrangement; this module is the arithmetic an agent must not do in its head.
It is handed a *summary* of the current arrangement (sections and the clips that sit on them) plus a
list of intentions, and returns what the new arrangement looks like — where every clip ends up, which
clips are copies of which, which clips' audio no longer matches the structure, and a sentence saying
what happened.

Sections are **labels, not containers**: moving a clip never moves them, which is what makes the
section strip trustworthy, while these operations move them deliberately.
"""
from __future__ import annotations

__all__ = ["arrange", "OPERATIONS"]

#: The intentions this module understands.
OPERATIONS = ("insert_bars", "duplicate_section", "remove_section", "repeat_last_section")


def _sections(payload: dict) -> "list[dict]":
    out: "list[dict]" = []
    for index, section in enumerate(payload.get("sections") or []):
        out.append(
            {
                "id": str(section.get("id") or f"sec-{index}"),
                "name": str(section.get("name") or f"section {index + 1}"),
                "start": float(section.get("start", 0)),
                "bars": max(1.0, float(section.get("bars", 4))),
            }
        )
    return sorted(out, key=lambda entry: entry["start"])


def _clips(payload: dict) -> "list[dict]":
    out: "list[dict]" = []
    for track_index, track in enumerate(payload.get("tracks") or []):
        for clip in track.get("clips") or []:
            out.append(
                {
                    "id": str(clip.get("id")),
                    "track": str(track.get("name") or f"track{track_index + 1}"),
                    "start": float(clip.get("start", 0)),
                    "length": max(0.25, float(clip.get("length", 4))),
                }
            )
    return out


def _shift(clips: "list[dict]", sections: "list[dict]", at: float, count: float) -> None:
    """Push everything at or after `at` to the right; grow the section `at` lands inside."""
    for clip in clips:
        if clip["start"] >= at:
            clip["start"] += count
    for section in sections:
        if section["start"] >= at:
            section["start"] += count
        elif section["start"] + section["bars"] > at:
            section["bars"] += count


def arrange(payload: dict) -> dict:
    """Apply the operations to one arrangement summary.

    `payload` is `{bars, sections, tracks, operations, revision?}`; each operation is
    `{op: insert_bars, at_bar, count, name?}` or `{op: duplicate_section|remove_section, section}` or
    `{op: repeat_last_section}`. Section names are matched case-insensitively.
    """
    bars = float(payload.get("bars", 0))
    sections = _sections(payload)
    clips = _clips(payload)
    operations = payload.get("operations") or []
    if not isinstance(operations, list) or not operations:
        raise ValueError("arrange needs at least one operation")
    described: "list[str]" = []

    def find(needle: str) -> "dict | None":
        wanted = str(needle).strip().lower()
        for section in sections:
            if section["id"].lower() == wanted or section["name"].lower() == wanted:
                return section
        return None

    for operation in operations:
        kind = str(operation.get("op") or "").strip().lower()
        if kind not in OPERATIONS:
            raise ValueError(f"unknown operation {kind or '(missing)'}; use {', '.join(OPERATIONS)}")

        if kind == "insert_bars":
            at = max(0.0, float(operation.get("at_bar", 0)))
            count = max(1.0, float(operation.get("count", 8)))
            _shift(clips, sections, at, count)
            inside = [s for s in sections if s["start"] < at and s["start"] + s["bars"] > at]
            if not inside:
                sections.append(
                    {
                        "id": f"sec-{int(at)}-{int(count)}-{len(sections)}",
                        "name": str(operation.get("name") or f"段落 {len(sections) + 1}"),
                        "start": at,
                        "bars": count,
                    }
                )
            bars += count
            described.append(f"在第 {int(at) + 1} 小节插入 {int(count)} 小节")
            continue

        if kind == "repeat_last_section":
            if not sections:
                raise ValueError("this arrangement has no sections to repeat")
            target = max(sections, key=lambda entry: entry["start"])
        else:
            name = operation.get("section")
            if name is None:
                raise ValueError(f"{kind} needs a section name or id")
            target = find(str(name))
            if target is None:
                raise ValueError(f"no section called {name!r}")

        start = target["start"]
        end = start + target["bars"]

        if kind in ("duplicate_section", "repeat_last_section"):
            _shift(clips, sections, end, target["bars"])
            copies: "list[dict]" = []
            for clip in [entry for entry in clips if start <= entry["start"] < end]:
                copy = {
                    "id": f"{clip['id']}-x{len(copies)}",
                    "track": clip["track"],
                    "start": clip["start"] + target["bars"],
                    "length": clip["length"],
                    "copy_of": clip["id"],
                }
                clips.append(copy)
                copies.append(copy)
            sections.append(
                {
                    "id": f"{target['id']}-x",
                    "name": target["name"],
                    "start": end,
                    "bars": target["bars"],
                }
            )
            bars += target["bars"]
            described.append(f"复制「{target['name']}」段落（{len(copies)} 个片段，{int(target['bars'])} 小节）")
            continue

        # remove_section
        kept = [clip for clip in clips if not (start <= clip["start"] < end)]
        removed = len(clips) - len(kept)
        clips[:] = kept
        sections.remove(target)
        for clip in clips:
            if clip["start"] >= end:
                clip["start"] -= target["bars"]
        for section in sections:
            if section["start"] >= end:
                section["start"] -= target["bars"]
        bars = max(1.0, bars - target["bars"])
        described.append(f"删除「{target['name']}」段落（{removed} 个片段）")

    sections.sort(key=lambda entry: entry["start"])
    clips.sort(key=lambda entry: (entry["start"], entry["track"]))
    return {
        "schema_version": 1,
        "mode": "arrange",
        "revision": payload.get("revision"),
        "bars": bars,
        "sections": sections,
        "clips": clips,
        "diff_summary": "；".join(described),
        "warnings": [],
    }
