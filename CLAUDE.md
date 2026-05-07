# CLAUDE.md — Digital twin project context

This file is the standing brief for any Claude Code session in this repo. Read top to bottom on first message of a session.

## Who's who

- **User: Bukola.** PhD student in Sam Oliveira's lab. Owns the digital-twin track of the Partaker work. Always speak to her, not Sam.
- **PI: Sam Oliveira.** Owns repo `SamOliveiraLab/Viva-munk` (origin remote).
- **Framework author: Eran Agmon** (NOT Iran — speech-to-text mis-spells). His `vivarium-collective/Viva-munk` is the upstream remote.
- **Lab-mates referenced:** Nona (biofilm structure, hydrodynamics — separate paper), Rutuja, Harnish, Tolu, Hassan, Jana, Ore.

## What this project is

A digital twin of E. coli growth in a mother-chamber chip, seeded from real Partaker tracking data. Three-stage arc per Sam's framing:

1. **Imitation** — simulator seeded from real initial cell positions/sizes/angles. (`partaker_to_vivamunk.py` already does this — reads frame 0 of a Partaker `cell_history_*.csv`, builds Viva-munk agents, runs forward.)
2. **Digital twin proper** — predict from frame 0 only, quantify divergence from reality.
3. **Rule attribution** — add candidate rules one at a time until simulation matches reality.

**Scope guardrail:** This paper is **mother-chamber single-cell dynamics**, NOT biofilm structure.

**OUT of scope (Nona's lane):** EPS quantification, biofilm 3D structure metrics, "structure predicts behavior" framing.

**IN scope — Sam's actual rule menu** (verbatim from his meeting): hydrodynamics, cell-to-cell communication / signaling, intracellular rules, population-driven rules, boundary / crowdness rules. He categorized them as **intercellular, intracellular, or environmental**.

**Plus dataset-specific candidates:** growth rate variability, division asymmetry, mother-cell aging, IPTG response (post-frame-28), channel flushing geometry.

## The full story — paper arc in plain terms

Two movies side by side: **real** (microscope) and **sim** (digital twin started from frame 0 only). Both get measured for the same metrics:

- **Density** — where cells cluster
- **Motility** — how cells move
- **Physical constraints** — walls, flow direction, crowdness
- **Cell count, total area, mean position** (the basic time-series)

Compute a **goodness-of-fit** — single number summarizing how close sim is to real. Then iterate: **add one rule → re-run → re-measure → fit improves?** Keep the rules that close the gap. The rules that close the gap = the biology that governs the real experiment. **That's the paper.**

Sam's pitch line: "give me one frame, I'll give you the rest of the experiment."

## External collaborators

- **COMSOL collaborator** — supplies hydrodynamics simulation data for the chip geometry. Bukola will receive COMSOL data and feed it as the "environment" rule (Stage 3, hydrodynamics layer). Don't wait for COMSOL data to ship Stages 1–2.

## Slide deck deadline

Bukola is presenting **tomorrow** (slides at minimum). Tonight's job is producing the four artifacts (`sim.gif`, `real.gif`, `real_vs_sim.gif`, `divergence.png`) so slides 1–3 are real. Slides 4–6 (rules table, final match, what-this-unlocks) are framed as the roadmap, not yet results.

End deliverable is **six slides**:
1. Pitch ("one frame in, 8 hours out")
2. Real vs simulated, side-by-side video
3. Drift graph (cell count, total area, mean position over time)
4. Rules attribution table
5. Final match
6. What this unlocks

Every step should map to filling one of those slides.

## The dataset (`1_5_lauren_replicate_1`)

Stitched in Partaker from two consecutive runs:

| Stitched frames | Source file | Media |
|---|---|---|
| 0–27 (~28 frames) | `1-SR_1_5_6hPre-C_PlainM9_TS_MC1` | M9 (no IPTG) |
| 28–259 (~232 frames) | `2-SR_1_5_6hPre-C_1mM_IPTG_TS_MC1` | M9 + 1 mM IPTG |

**Bukola's analysis window:** frames **0–130** of the stitched series.

**Critical biology fact: media switch at frame 28** (IPTG arrival). Mark frame 28 on every timeline output. Don't expect Stage-1/2 simulation rules to predict IPTG-driven changes — that's Stage 3 territory.

**Parameters:**
- Pixel size: **0.0645 µm/px** (100× Nikon)
- Frame interval: **5 min = 300 s**
- Frame 0 has **213 cells** (some have NaN morphology; loader skips those)

## CSVs and what they're for

Both live at `/Volumes/SAM1/server_workspace_backup/` (server's view; Bukola's Mac sees the same SAM1 mount):

- **`cell_history_amby.csv`** — per-cell biographies, list-columns. Format consumed by `partaker_to_vivamunk.py` (frame 0 → simulator seed).
- **`enhanced_tracking_data_all_tracks.csv`** — per-cell-per-frame with lineage. Best source for **rendering the real movie** (`real.gif`).

## Current state (as of 2026-05-06)

- Stage 1 redo in progress on the server `ma9010209f4lcwn` (Tailscale). Goal: produce `sim.gif`, `real.gif`, `real_vs_sim.gif`, `divergence.png` in `out/digital_twin_2026-04-29/`.
- Approach: smoke test with first 50 cells from frame 0, then scale to all 213.
- Bug fixed and pushed: NaN guard in `load_partaker_cells` (Partaker emits NaN for some cells; size filter let them through and Chipmunk crashed). See commit `9c034bf`.
- **Open blocker:** server's default Python is 3.10, but modern `process-bigraph` requires ≥3.11. Need a python3.11+ on the server (check `which python3.11 python3.12`, `conda env list`).
- Local pinned versions that work: `process-bigraph==1.1.4`, `bigraph-schema==1.1.3`, `bigraph-viz==1.0.4`.

## Path map (matters!)

| Where | Repo path | Data path |
|---|---|---|
| Bukola's Mac (`AuspiciousAmby`) | `/Volumes/Extreme SSD/Partaker-results/DigitalTwin/Viva-munk` | `/Volumes/SAM1/server_workspace_backup/` |
| Lab server (`ma9010209f4lcwn`, user `smdoliveira`) | `/Volumes/Server_Data/Synced-Microscope-Images(Lab pc)/Bukola/Viva-munk` | `/Volumes/SAM1/server_workspace_backup/` |

## Working rules with Bukola

- **Short responses.** "no long talk pls" was an explicit ask.
- **Plain commit messages — never add a `Co-Authored-By: Claude` (or any AI-coauthor) trailer.** Apply to all commits in this repo.
- **Anchor every step to the 3-stage arc and the 6-slide deliverable.**
- Match her language: she's casual ("buddy", lowercase). Mirror that, don't be stiff.
- When she asks "what should we do," propose, don't dump a long plan. One concrete next step + the why.
- Confirm risky actions (push, force-push, deleting branches). She trusts judgment but should still see the action before it happens.

## How `partaker_to_vivamunk.py` works (one paragraph)

CLI: `python partaker_to_vivamunk.py path/to/cell_history.csv [--pixel_size 0.0645] [--frame_interval 300] [--sim_time 28800]`. Reads frame 0 only (`if int(row['start_time']) != 0: continue`), filters out Artifacts/NaN/out-of-range cells, builds Viva-munk segment agents, attaches `grow_divide`, runs `run_experiment` which writes a GIF + bigraph viz + SQLite history under `out/`. To run for the full 130-frame data window, pass `--sim_time 39000` (130 × 300 s).

## What NOT to add to this file

- Architecture diagrams or file walkthroughs (read the code).
- Resolved bugs or fix recipes (`git log` is canonical).
- Per-conversation TODOs (use tasks).
- AI memoir.
