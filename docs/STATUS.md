# Current project status

Updated 2026-10-28. Experiments are complete through **Stage 23**; Stage 24 is proposed and has not been implemented.

## Current models

| Role | Model / checkpoint | Use |
| --- | --- | --- |
| Default transcription workflow | Offline CNN-GRU, `outputs/cnn_gru/best-inference.pt` | Local standard-tuning guitar audio to timed string/fret notes and MIDI; uses future context. |
| Selected research candidate | Stage 22 `uniform_ce650`, `outputs/balance22/models/uniform_ce/final.pt` | Selected on fixed development scores; preserved through Stage 23 with no default promotion. |
| Weighted treatment | `outputs/balance22/models/balanced_ce/final.pt` | Tested class weighting; not selected. |
| Pretrained references | MT3 inference adapter and Basic Pitch | Stage 1 baseline comparisons. |

The offline model has 935,081 parameters and centered CQT features. The causal models have 931,361 parameters, duplicated past-only 128 ms STFT planes and a carried unidirectional GRU state. Different data, timing and training histories prevent treating all stages as one model leaderboard. File-PCM replay does not measure microphone-to-speaker latency.

## Latest completed experiment

Stage 23 locked all four checkpoint identities, the fixed decoder, checkpoint normalization and the exact Stage 22 development-selection record before reading test audio or annotations. It evaluated the existing twelve progression-3 recordings with **zero training, optimizer instantiations or normalization updates**: 48 scored twelve-second replays plus four one-second test-audio warmups. It performed no test-driven model reselection or calibration.

| Locked model | Development state F1 | Test state F1 | Development onset F1 | Test onset F1 |
| --- | ---: | ---: | ---: | ---: |
| Parent500 | 0.0828 | 0.0291 | 0.1148 | 0.0724 |
| Original loss650 | 0.1066 | 0.0513 | 0.1445 | 0.0892 |
| Uniform CE650, selected on development | 0.1203 | 0.0578 | 0.1493 | 0.0832 |
| Weighted CE650 | 0.1084 | 0.0528 | 0.1310 | 0.0785 |

These F1 values average clips within each split. Uniform CE's test state advantage over original loss is **+0.00659**, while its note-onset difference is **-0.00601**. The state advantage persists; the onset advantage does not. Absolute scores fall substantially from development. The copied development selection remains `uniform_ce650` regardless of test rankings.

The test IDs are disjoint from this training/development split, but the recordings had earlier historical evaluations and all six players occur across splits. This is not a fresh independent-player or electric-guitar generalization result. Test-supported fret positions, musical content and diagnostic denominators differ from development; this comparison does not identify a cause for the decline.

Open the [Stage 23 audio review](../outputs/holdout23/review.html), [findings](../outputs/holdout23/findings.md) or [verification record](../outputs/holdout23/verification.json). The [tracked snapshot](../experiments/stage23.snapshot.json) records **1,418 artifact/code/input bindings**.

## What the evidence supports

The project supports local research on isolated standard-tuning guitar, inspectable string/fret notes, controlled acoustic-model comparisons, constrained fingering and rhythm/notation drafts. It does not establish reliable transcription of arbitrary songs or electric-guitar recording conditions.

Training uses 48 progression-1 recordings. Recent development experiments repeatedly use twelve progression-2 recordings and an earlier decoder calibration. Stage 22 spent 450 new updates total, 150 per matched arm, and selected uniform active-fret CE by development state F1 then onset F1. Its training-only cap4 amendment and all original evidence remain preserved. See the [Stage 22 trial](../outputs/balance22/review.html), [experiment index](../experiments/README.md) and [detailed guide](experiment-guide.md) for the lineage.

Stage 20 found that all 64 development GT positions had prior supervision; Stage 21 found only a small conditional-fret gain over fixed frequency priors. Those oracle-GT-active diagnostics use different denominators from end-to-end transcription and cannot establish deployable accuracy.

## Verification and housekeeping

The full suite passed **688 tests**, including 18 new Stage 23 tests, with one existing `pretty_midi/pkg_resources` deprecation warning. Tests and browser QA ran separately on this 8 GB host. Verification adds no acoustic experiment training; existing synthetic optimizer test fixtures remain part of the suite.

The [numeric audit](../outputs/holdout23/audit.json) reproduces all twelve target caches from original annotations with exact shape/dtype/array bytes, checks four unchanged checkpoint files and the byte-identical copied selection, and reconstructs the 48 saved cases without new neural inference. It checks **7,181 exact note/MIDI attacks** and **14,362 events**, including frame/head scores, official note/string scores, source replays and exports.

Browser QA checks 48 model/clip views, 240 export links, 48 exact JSON downloads, 48 playback-preserving model cycles, completion-gated logs and 14,362 three-clock event markers. It checks 136 EOF tags and active playback switching. Desktop/mobile screenshots were inspected. The browser caught a template substitution error; the repaired renderer passed without changing model outputs or scores.

Processing/audio excludes EOF flush cost while virtual completion includes it. Identical graphs cannot establish a speed benefit from the loss; matched-delay subsets omit misses. No capture-hardware latency was measured.

The full test run verified Stage 22's 1,128 bindings before and after tests. The final read-only integrity check covers all 23 indexed stages and Stage 23's 1,418 bindings. Historical producers and artifacts retain their exact bytes. The current model record still points to Stage 22 development, because Stage 23 makes no new model selection.

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
```

The [verification guide](VERIFICATION.md) explains integrity-only mode, local asset requirements and the fresh-environment runner. Run browser QA and the full suite separately on this machine. Use the project checker for routine read-only checks; Stage 23 rejects report regeneration after verification and repeat inference after its protocol is created.

Earlier housekeeping established local Git, byte-preserving attributes, indexed documentation and a fresh isolated-environment CPU transcription smoke check. That check installed 39 cached packages offline and produced 268 timed notes from 12 seconds, with matching MIDI/JSON. It used the workspace Python and local assets; it was not an accuracy or clean-machine portability result. The [housekeeping receipt](../experiments/housekeeping-checks.json) describes commit `466f77b`; its helper hashes belong to that version.

## Next research step

The [Stage 24 proposal](../outputs/holdout23/next-experiment.json) adds an explicit research inference CLI for the locked `uniform_ce650` checkpoint. A local audio file would produce timed JSON/CSV/MIDI and a simple string/fret text view using the frozen PCM frontend and decoder. The existing offline CLI remains the default.

This engineering step allows zero training, no calibration search, no reselection or model-quality promotion, and no meter-prediction claim. Parity checks use an existing development excerpt and synthetic boundary fixtures; progression-3 scores must not tune the implementation. It has not been implemented.

## Preserving the workspace

Git tracks source, configs, tests, small reference fixtures, documentation and compact records. Ignored `data/`, `models/` and `outputs/` need a separate backup. Historical package/output paths and exact source bytes support archived hashes; do not mass-format or move them during cleanup.

`.cache/python` contains the interpreter used by `.venv`; preserve it. `.venv-tf-check` supports the recorded MT3 frontend fixture workflow. Identify disposable temporary files individually before cleanup. The selected research checkpoint is a causal two-plane `multires17.model.CausalCNNGRU`, separate from the default offline model's CLI.
