# Agent notes — ShadowRoom

Repo conventions and the current roadmap, for anyone (human or agent) picking this up later.

## Repo map

One MCP plugin = one repository.

These MCPs are the **tool layer**: each one is useful on its own (any MCP-capable agent can drive
it), and together they are the toolkit that `producer-tools` bundles into one app and
`shadow-producers` drives as an agent.

| Repo | What it is | Status |
| --- | --- | --- |
| `rekordbox-serato-bridge` | Rekordbox ⇄ Serato library conversion (cues, loops, colors, playlists) | published; real-machine tested |
| `audio-analysis-dedupe` | Audio analysis + duplicate detection | published; not yet user-tested |
| `set-planner` | Set timing + next-track suggestions | published; not yet user-tested |
| `shadow-music-generator` | Queue for local AI music generation pipelines | published; not yet user-tested |
| `producer-tools` | the app that collects these tools into one product | scaffolding (UI reference written) |
| `shadow-producers` (planned) | the AI agent | not started |

## House rules

- **License:** AGPL-3.0 everywhere (see `LICENSE`).
- **README:** user-facing only — features / requirements / install / configuration / tools / usage /
  safety / troubleshooting / license. The **intro is bilingual** (one English sentence + the same
  sentence in Chinese), and so is the GitHub repository description; the rest of `README.md` stays
  English with the full Chinese text in `README.zh-CN.md`. Implementation notes belong in
  `docs/internals.md`. See `CONTRIBUTING.md`.
- **Commits:** `feat(scope):`, `fix(scope):`, `docs(scope):`, `chore(scope):`; author
  `shadowroommusic <shadowroommusic@users.noreply.github.com>` (set per repo via
  `git config user.name` / `user.email`).
- **No guessing about vendor formats.** Only behaviour that was measured or verified on a real
  machine goes into code, docs or tests.
- **Real-machine verification** (Rekordbox, Serato, the actual hardware) is required before calling a
  conversion feature done.

## Decisions

- **2026-09-14 — UI/frontend reference.** `producer-tools` follows the UI/frontend patterns of
  DeepSeek Harness (pnpm workspaces, React + Vite + CSS Modules, token-based theming, plugin-style
  `ui-*` packages with slots and docks, per-package i18n) with our own `--sr-*` tokens and brand
  colours. The agent core is the Codex harness (bundled `codex` binary + `app-server` protocol).
  Details: `producer-tools/docs/ui-reference.md`.

- **2026-09-14 — harness evaluation (Codex vs DeepSeek).** The Codex harness (`openai/codex`,
  Apache-2.0) **can** be used inside Shadow Producers: bundle the `codex` binary and drive it in
  `app-server` mode through its protocol/control socket (`codex-rs/app-server*` crates,
  `codex-app-server-protocol` is published on crates.io; the same pattern the Codex desktop app
  uses). Linking the Rust crates directly would require vendoring the workspace, because
  `codex-core` / `codex-app-server` are **not** published. Codex natively speaks MCP, so the four
  ShadowRoom plugins drop straight in. `deepseek-ai/deepseek-harness` (MIT, everything-is-a-plugin)
  works as a reference or alternative, but is a developer preview with breaking changes.
  Reference checkouts live in `~/Documents/ShadowRoom/_reference/{codex,deepseek-harness}`.

- **2026-09-14 — generator backend (option B).** `shadow-music-generator` stays backend-less for
  now: it is used as a job record + request manager (prompts, lyrics, parameters, license notes,
  dry-run by default). The real provider wiring happens later inside `producer-tools`. The YuE2
  adapter in `src/shadow_music_generator/adapters/` is kept as ready groundwork for whenever a
  machine or API is chosen — YuE2 itself needs Linux + an NVIDIA GPU (BF16, 24 GB), so it normally
  runs remotely (local GPU box, ssh, or a cloud API).

## Roadmap

1. Publish each MCP as its own public repository. ✅ *(done)*
2. **Test and optimise every MCP and the software gradually** — plugin by plugin, on the real
   machine, revisiting behaviour, edge cases and UX as we go.
3. Build `producer-tools`: the application that collects these MCPs into one product.
4. Build `shadow-producers`: the AI agent repository.
5. Open-source both when they are ready.

## Local environment (maintainer machine)

- Working copies live in `~/Documents/ShadowRoom/<repo>`; those are the git repos that push to
  GitHub.
- GitHub: `https://github.com/shadowroommusic/*`, pushed over SSH (`~/.ssh/id_ed25519`, key titled
  "shadowroom-automation (Codex)"). 2FA is enabled on the account.
- The Codex plugin marketplace named `shadowroom` currently points at an older monorepo checkout;
  the repositories above are the source of truth.

- **2026-09-15 — the synth arranges: sections, segments, and MIDI from the plan.** `render_song`
  takes `sections` (`intro` / `drop` / …) that name the timeline and set the length, and every part
  can be a chain of `segments` — each one a stretch of bars with its own pattern/notes/gain, so a
  breakdown is `{"bars": 8, "gain": 0}`. Every part is fitted to the song's length, so stems, mix and
  MIDI line up. `midi_path` writes the arrangement as a Type-1 MIDI file (drums on channel 10, swing
  included, humanised timing not) — not a transcription. Verified end to end through the workbench:
  "a 12-bar deep house, intro 4 / drop 6 / outro 2, and its MIDI too" → stems + mix with intro 0.098
  / drop 0.252 / outro 0.105 RMS and a 148-note MIDI at the song's tempo.

- **2026-09-15 — drum rows tile, and that was a real bug.** `_render_drums` placed a step row once, so
  every drum part longer than one bar was one bar of audio followed by silence — measured on the
  stems the workbench had already rendered (`night-drive-drums.wav`: bar 1 = 0.108 RMS, bars 2–8 =
  0.000). Rows now repeat across the part (16 steps = one bar, 32 = a two-bar block), and pitched
  parts gained `repeat` for the same reason: a written bar can cover an eight-bar part.
