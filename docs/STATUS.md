# Current project status

Updated 2026-10-28. Experiments are complete through **Stage 19**; Stage 20 is proposed and has not run.

## Current models

| Role | Model / saved checkpoint | Use |
| --- | --- | --- |
| Default transcription workflow | Offline CNN-GRU, `outputs/cnn_gru/best-inference.pt` | Local guitar audio to timed string/fret notes and MIDI; uses future context. |
| Latest selected research candidate | Stage 19 `control500`, `outputs/fret19/models/control/final.pt` | Causal two-plane CNN-GRU research comparison; selection applies within Stage 19. |
| Latest tested treatment | `outputs/fret19/models/conditional_fret/final.pt` | Added conditional-fret loss; not adopted. |
| Pretrained references | MT3 inference adapter and Basic Pitch | Baseline comparisons documented in Stage 1. |

The offline model has 935,081 parameters and centered CQT features. The latest causal models have 931,361 parameters, duplicated past-only 128 ms STFT feature planes and a carried unidirectional GRU state. Their different data, timing, training histories and evaluation settings prevent treating scores across all stages as one model leaderboard.

The causal branch processes file PCM packets. It is not a measured microphone-to-speaker application. The project includes experiments for constrained guitar fingering, rhythm/chord drafts, editable MusicXML and controlled guitar/accompaniment mixtures; their individual limits remain in the [detailed guide](experiment-guide.md).

## Latest completed experiment

Stage 19 restored the same 250-update parent into two arms. Each received 250 new updates over five additional sampled-crop epochs, reaching 500 lifetime updates. The stage spent **500 new optimizer updates total**. The one shared parent's 250 historical updates are counted once.

| Reused-development measurement | Original loss / control | Extra conditional-fret loss |
| --- | ---: | ---: |
| Argmax exact string/fret frame F1, primary | 0.0828 | 0.0749 |
| Note-onset F1 | 0.1148 | 0.1136 |
| Best active fret on reference-active frames | 18.66% | 19.55% |
| Correct fret when reference and prediction are active | 18.79% | 19.44% |

The auxiliary term improved the targeted fret-choice measurements but lowered the primary frame score and slightly lowered note scores. The declared rule selected **`control500`**. The offline default remains unchanged.

Read the local [findings](../outputs/fret19/findings.md), [interactive review](../outputs/fret19/review.html) or [verification record](../outputs/fret19/verification.json). These files are ignored by Git and are present in the recorded workspace. A compact [tracked verification snapshot](../experiments/stage19.snapshot.json) preserves their hashes.

## What the evidence supports

- Local, reproducible research workflows on isolated standard-tuning guitar, with notes, string/fret positions and inspectable exports.
- Controlled comparisons of small acoustic models, decoding choices, supervision and feature representations under limited training budgets.
- Explicit separation of source estimates, packet availability and measured virtual processing completion where a stage actually measured that clock.

Recent stages repeatedly use the same twelve progression-2 development recordings. Training uses 48 progression-1 recordings, and players occur across splits. Earlier progression-3 and external-electric results are historical evaluations, not a fresh selection set for another round of tuning. Current results do not establish reliable transcription of arbitrary songs, instruments or electric-guitar recording conditions.

## Verification and housekeeping

The last recorded full test run, on 2026-10-22, passed **565 tests** with one existing dependency deprecation warning. Stage 19 also records 36 audited model/clip cases, twelve bit-identical parent probability reproductions, exact exports and checks of all 36 browser views. See the [test report](../outputs/fret19/test-results.json) and [audit](../outputs/fret19/audit.json).

On 2026-10-28, this documentation cleanup rechecked all **304 artifact/code bindings** from the tracked Stage 19 snapshot. Historical producer modules and experiment artifacts remain in place. The documentation checks do not constitute a new inference benchmark or a fresh full test run.

| Housekeeping item | Status |
| --- | --- |
| Local Git baseline | Complete: initial commit `895eed6`, branch `main`. |
| Preserve file bytes across Git checkouts | Complete: `.gitattributes` disables automatic line-ending conversion. |
| Concise README, current status and experiment index | Complete in this documentation cleanup. |
| One project-wide verification command | Planned; current commands remain listed below. |
| Fresh isolated-environment transcription smoke run | Planned; no new portability claim yet. |
| Stage 20 coverage diagnosis | Proposed; not executed. |

Current checks in the existing workspace:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache\uv'
uv pip check --python .venv\Scripts\python.exe
```

The full historical test suite uses local checkpoint/data assets in addition to source and fixtures. Stage-specific reproduction commands are in the [detailed guide](experiment-guide.md). Some report/audit commands rewrite their output records, so they are different from a read-only file-integrity check.

## Next research step

The [Stage 20 proposal](../outputs/fret19/next-experiment.json) is a **zero-training, zero-neural-inference coverage audit**. It will compare available training labels with actual supervised crop exposure and relate rare/unseen `(string, fret)` labels to saved errors.

Available source frames, repeated supervised presentations and unique supervised source-frame/string identities will be counted separately. Parent and child histories differ and must use their own recorded crop schedules. The proposal makes no coverage finding yet and does not select another model. Independent recordings remain necessary for a later generalization claim.

## Preserving the workspace

Git tracks source, configs, tests, small reference fixtures, documentation and compact experiment records. The ignored `data/`, `models/` and `outputs/` directories need a separate backup to preserve the completed research. The [experiment index](../experiments/README.md) records their local paths.

Keep historical stage packages and artifact paths in place: their imports, receipts and hashes support reproducibility. `.gitattributes` preserves existing source bytes; automatic formatting or line-ending rewrites can invalidate archived hashes.

`.cache/python` contains the interpreter used by `.venv` on this workspace. Preserve that runtime. The separate `.venv-tf-check` environment supports the recorded MT3 frontend fixture workflow. Disposable cache or temporary files should be identified individually before any disk cleanup.
