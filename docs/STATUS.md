# Current project status

Updated 2026-10-28. Experiments are complete through **Stage 21**; Stage 22 is proposed and has not trained.

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

Stage 21 compared saved best-active fret decisions with fixed training-frequency priors, using **zero new training updates and zero neural inference calls**. Every predictor uses the same 22,219 GT-active string-frames and 64 supported positions, with each model's own recorded training history.

| Control500 diagnostic | Saved model | Primary presentation-count majority prior | Unique-count majority prior, secondary |
| --- | ---: | ---: | ---: |
| Frame-weighted best-active fret accuracy | 18.66% | 16.62% | 18.03% |
| Equal-supported-position accuracy | 12.89% | 9.38% | 9.38% |

The model chooses its training-modal fret on **60.64%** of active GT frames. Its accuracy exceeds the independence expectation from its own output/GT marginals by **3.53 percentage points**. This shows some alignment beyond those frequency marginals alongside concentrated predictions. It does not establish a causal mechanism or justify selecting another model.

The primary frame-weighted gain is 2.04 percentage points; the control beats that prior on seven clips, ties on two and loses on three. All declared priors remain visible: presentation, unique and available counts, with deterministic majority and analytic distribution views, plus uniform25. The primary source was fixed before examining these results. GT-active conditioning excludes the activity decision, so these are oracle-support diagnostics rather than end-to-end accuracy.

Open the [prior/confusion review](../outputs/prior21/review.html), [findings](../outputs/prior21/findings.md) or [verification record](../outputs/prior21/verification.json). The [tracked Stage 21 snapshot](../experiments/stage21.snapshot.json) preserves 693 artifact/code/input bindings.

## Previous coverage diagnosis

Stage 20 counted the actual supervised exposure and related it to saved errors, with **zero new training updates and zero neural inference calls**. It examined 48 training target caches, the recorded crop schedules and all 36 model/clip validation cases.

| Training history | Available active source frames | Repeated active presentations | Unique supervised string-frame identities |
| --- | ---: | ---: | ---: |
| Parent, epochs 1-5 | 174,071 | 242,592 | 127,390 |
| Each child, epochs 1-10 | 174,071 | 476,406 | 154,681 |

All **64 ground-truth positions** used by validation had already received supervision; the unseen bin has no active frames or reference notes. For `control500` argmax, the 1000+ presentation bin contains 21,386 of 22,219 active GT string-frames. It accounts for **95.81% of active-frame errors**, with an **89.07% within-bin error rate**. The remaining 833 active frames in lower-exposure bins all fail exact-state prediction.

This points toward difficulty predicting supported positions. Repeated frames can still represent few independent examples, and the bin comparison does not establish causation. Each model uses its own history; the children's shared exposure is counted once per history. Pitch-versus-position diagnostics retain their empty validation cohorts rather than inventing error rates.

Open the [coverage review](../outputs/coverage20/review.html), [findings](../outputs/coverage20/findings.md) or [verification record](../outputs/coverage20/verification.json). The [tracked Stage 20 snapshot](../experiments/stage20.snapshot.json) preserves 658 artifact/code/input bindings, including the source experiment receipts and frozen target caches. No new model was selected.

## Latest trained comparison

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

The latest full test run, on 2026-10-28, passed **633 tests** with one existing dependency deprecation warning, including 27 new Stage 21 tests. See the [Stage 21 test record](../outputs/prior21/test-results.json).

Stage 21 independently reconstructed all three saved-model confusions with scalar comparisons and all 21 prior confusions with explicit expected-mass loops. It checked exact CSV values in 900 prior rows, 3,600 class rows, 144 string rows, 288 case rows and 38,400 supported-class confusion rows. Browser checks covered 24 model/predictor views, 576 control states, 264 export links, three exact confusion-JSON downloads and 36 source-case links. Desktop and mobile screenshots were inspected.

Stage 20's independent audit reconstructed both histories using delayed target arrays and crop masks, then compared them with the main source-range counting algorithm. It reproduced all original frame totals and official note matches, and checked exact values in 2,259 note rows, 300 coverage rows and 900 class-error rows. Its browser checks covered six model/policy views, 420 filter states, 54 export links and six exact JSON downloads; desktop and mobile screenshots were inspected.

Stage 19's original run passed 565 tests and records 36 audited model/clip cases, twelve bit-identical parent probability reproductions, exact exports and checks of all 36 browser views. Its original [test report](../outputs/fret19/test-results.json) and [audit](../outputs/fret19/audit.json) remain unchanged.

The full test run verified Stage 20's **658 artifact/code/input bindings** before and after the tests. The final integrity check covers all 21 indexed stages and the Stage 21 snapshot's **693 artifact/code/input bindings**. Historical research artifacts and producers remain byte-identical.

A newly created virtual environment installed **39 packages offline** from the local cache and ran the real offline transcription CLI on CPU. Its 12-second development clip produced 268 timed notes, with matching MIDI/JSON and valid string/fret intervals. Package imports were checked to come from that new environment. This is an environment/export smoke check using local assets and the workspace Python runtime, not an accuracy or clean-machine portability result.

That earlier [housekeeping receipt](../experiments/housekeeping-checks.json) belongs to commit `466f77b`; its helper-source hashes describe that version. The Stage 20 snapshot records the updated checker that also supports diagnosis snapshots while preserving the Stage 19 model selection.

| Housekeeping item | Status |
| --- | --- |
| Local Git baseline | Complete: initial commit `895eed6`, branch `main`. |
| Preserve file bytes across Git checkouts | Complete: `.gitattributes` disables automatic line-ending conversion. |
| Concise README, current status and experiment index | Complete in this documentation cleanup. |
| One project-wide verification command | Complete: saved hashes, indexed paths, dependency compatibility and full tests. |
| Fresh isolated-environment transcription smoke run | Complete: offline install, CPU CLI and MIDI/JSON contract checks. |
| Stage 20 coverage diagnosis | Complete: zero training/inference, independent counts, exports and browser checks. |
| Stage 21 training-prior diagnosis | Complete: zero training/inference, scalar reconstructions, full confusions and browser checks. |

Current checks in the existing workspace:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
```

The full historical test suite uses local checkpoint/data assets in addition to source and fixtures. The [verification guide](VERIFICATION.md) explains the command's scope, integrity-only mode and separate fresh-environment smoke runner. Stage-specific reproduction commands remain in the [detailed guide](experiment-guide.md). Some old report/audit commands rewrite their output records; the new checker reads their saved evidence directly.

## Next research step

The [Stage 22 proposal](../outputs/prior21/next-experiment.json) is a matched class-weighting trial capped at **450 new optimizer updates total**: three arms receive 150 updates over three shared sampled-crop epochs. All begin from the same complete saved control500 model/AdamW/AMP/RNG state and end at lifetime step650.

The arms are original loss, uniform active-only fret CE and the same auxiliary term with bounded inverse-square-root class weights from unique supervised training counts. Weighting is a hypothesis to test, not an established fix. Fixed final state/note scores and conditional fret diagnostics will expose tradeoffs. The trial has not trained, allows no independent-test inference and proposes no production promotion. Independent recordings remain necessary for a later generalization claim.

## Preserving the workspace

Git tracks source, configs, tests, small reference fixtures, documentation and compact experiment records. The ignored `data/`, `models/` and `outputs/` directories need a separate backup to preserve the completed research. The [experiment index](../experiments/README.md) records their local paths.

Keep historical stage packages and artifact paths in place: their imports, receipts and hashes support reproducibility. `.gitattributes` preserves existing source bytes; automatic formatting or line-ending rewrites can invalidate archived hashes.

`.cache/python` contains the interpreter used by `.venv` on this workspace. Preserve that runtime. The separate `.venv-tf-check` environment supports the recorded MT3 frontend fixture workflow. Disposable cache or temporary files should be identified individually before any disk cleanup.
