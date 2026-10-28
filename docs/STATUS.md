# Current project status

Updated 2026-10-28. Experiments are complete through **Stage 22**; Stage 23 is proposed and has not run.

## Current models

| Role | Model / checkpoint | Use |
| --- | --- | --- |
| Default transcription workflow | Offline CNN-GRU, `outputs/cnn_gru/best-inference.pt` | Local standard-tuning guitar audio to timed string/fret notes and MIDI; uses future context. |
| Latest selected research candidate | Stage 22 `uniform_ce650`, `outputs/balance22/models/uniform_ce/final.pt` | Fixed final development selection among three matched causal continuations; no default promotion. |
| Latest weighted treatment | `outputs/balance22/models/balanced_ce/final.pt` | Tested class weighting; not selected. |
| Pretrained references | MT3 inference adapter and Basic Pitch | Stage 1 baseline comparisons. |

The offline model has 935,081 parameters and centered CQT features. The causal models have 931,361 parameters, duplicated past-only 128ms STFT planes and a carried unidirectional GRU state. Different data, timing and training histories prevent treating all stages as one model leaderboard. File-PCM replay does not measure microphone-to-speaker latency.

## Latest completed experiment

Stage 22 restored the same complete `control500` model, AdamW moments/options/counters, AMP scaler and CPU/CUDA RNG into three arms. Each received **150 new updates over three shared sampled-crop epochs**, reaching lifetime step650. The trial spent **450 new updates total**; its one shared historical parent has500 updates, counted once. The decoder, frontend, normalization and learning rate0.0001 stayed fixed.

| Reused-development metric | Parent500 | Original loss650 | Uniform fret CE650 | Weighted fret CE650 |
| --- | ---: | ---: | ---: | ---: |
| Mean clip argmax string/fret state F1, primary | 0.0828 | 0.1066 | **0.1203** | 0.1084 |
| Mean clip note-onset F1 | 0.1148 | 0.1445 | **0.1493** | 0.1310 |
| Best fret on all active GT frames | 18.66% | 20.82% | 21.13% | 20.49% |
| Equal-supported-position fret accuracy | 12.89% | 12.49% | 13.00% | 12.98% |
| Fret accuracy when GT and prediction are active | 18.79% | 21.86% | 21.80% | 20.88% |

The fixed final rule selects **`uniform_ce650`** by state F1, then note-onset F1. Only the three step650 finals are eligible; the parent is a dose reference. Weighted CE lowers state F1 by0.01192 and onset F1 by0.01837 versus uniform CE, so this trial does not support adopting the weighting rule. Conditional/macro fret metrics use different denominators and do not override primary selection.

A **pre-training amendment** handles one training-available but historically unexposed position, string4/fret17:50 available active frames, zero unique supervised frames in the parent history and76 presentations in the fixed new crops. The original proposal would reject it. Before any optimizer update, the existing cap4 was assigned to such available/unexposed positions; labels absent from training retain zero weight and fail if active. No validation result, crop, arm, coefficient or budget changed. The [exact weight/amendment record](../outputs/balance22/class-weights.json) and initial preflight stop are preserved; the original proposal remains byte-identical.

Open the [Stage 22 audio review](../outputs/balance22/review.html), [findings](../outputs/balance22/findings.md) or [verification record](../outputs/balance22/verification.json). The [tracked snapshot](../experiments/stage22.snapshot.json) records **1,128 artifact/code/input bindings**.

## What the evidence supports

The project supports local research on isolated standard-tuning guitar, inspectable string/fret notes, controlled acoustic-model comparisons, constrained fingering and rhythm/notation drafts. It does not establish reliable transcription of arbitrary songs or electric-guitar recording conditions.

Recent stages repeatedly use the same twelve progression-2 development recordings and an earlier decoder calibration. Training uses48 progression-1 recordings, and players occur across splits. Stage20 found that all64 validation GT positions had prior supervision; Stage21 found only a small conditional-fret gain over fixed frequency priors. See the [experiment index](../experiments/README.md) for the complete lineage and the [detailed guide](experiment-guide.md) for earlier results.

## Verification and housekeeping

The latest full suite passed **670 tests**, including37 new Stage22 tests, with one existing dependency deprecation warning. An initial attempt overlapped browser QA and hit CPU allocator exhaustion/CUDA errors on this8GB host. The sequential rerun passed; [both attempts are retained](../outputs/balance22/test-results.json). Verification reruns added no acoustic experiment updates.

The [numeric audit](../outputs/balance22/audit.json) checks all34 real AdamW counters at650 per child, exact pre-update restoration snapshots, identical final paired RNG streams and independently reconstructed class exposure for epochs11-13:47,469 /43,499 /46,982 active presentations. It reproduces all48 cases, twelve bit-identical parent probability sets, **7,395 exact note/MIDI attacks** and **14,790 events**, without additional neural inference or optimizer updates.

Browser QA checks48 model/clip views,240 export links,48 exact JSON downloads,48 playback-preserving model cycles, completion-gated logs and14,790 three-clock event markers. EOF closures are tagged; active playback survives model switching. Desktop/mobile screenshots were inspected. Processing/audio excludes EOF flush cost while virtual completion includes it. Identical graphs cannot establish a speed benefit from the loss; matched-delay subsets omit misses. No capture-hardware latency was measured.

The full test run verified Stage21's693 bindings before and after tests. The final read-only check covers all22 indexed stages and Stage22's1,128 bindings. Historical research producers and artifacts retain their exact bytes.

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
```

The [verification guide](VERIFICATION.md) explains the integrity-only mode, local asset requirements and fresh-environment runner. Run browser QA and the full suite separately on this machine. Stage-specific audits/report commands remain in the detailed guide; some regenerate their own evidence, whereas the project checker reads saved evidence directly.

Earlier housekeeping established local Git, byte-preserving attributes, indexed documentation and a fresh isolated-environment CPU transcription smoke check. That check installed39 cached packages offline and produced268 timed notes from12 seconds, with matching MIDI/JSON. It used the workspace Python and local assets; it was not an accuracy or clean-machine portability result. The [housekeeping receipt](../experiments/housekeeping-checks.json) describes commit `466f77b`; its helper hashes belong to that version.

## Next research step

The [Stage 23 proposal](../outputs/balance22/next-experiment.json) locks all four model identities and the selected `uniform_ce650` checkpoint before evaluating the existing twelve progression-3 recordings. It allows **zero training**, no calibration, no test-driven model reselection and no production promotion.

Those recordings had earlier historical evaluations. They are held out from this training/development protocol, but are not an untouched independent dataset; players occur across splits. This proposed check cannot establish fresh electric-guitar generalization. It has not run, and Stage22 performed no test inference.

## Preserving the workspace

Git tracks source, configs, tests, small reference fixtures, documentation and compact records. Ignored `data/`, `models/` and `outputs/` need a separate backup. Historical package/output paths and exact source bytes support archived hashes; do not mass-format or move them during cleanup.

`.cache/python` contains the interpreter used by `.venv`; preserve it. `.venv-tf-check` supports the recorded MT3 frontend fixture workflow. Identify disposable temporary files individually before cleanup. The latest research checkpoint is a causal two-plane `multires17.model.CausalCNNGRU`, separate from the default offline model's CLI.
