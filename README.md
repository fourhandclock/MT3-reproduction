# Guitar transcription research

Low-compute guitar transcription experiments, completed through **Stage 21**. The default workflow is the offline CNN-GRU. The causal branch, constrained fingering, rhythm/notation and separation experiments are documented with their own controls and limits.

- [Current status](docs/STATUS.md): defaults, latest results, limitations and remaining work.
- [Experiment index](experiments/README.md): all 21 stages, code, configs and local reviews.
- [Detailed experiment guide](docs/experiment-guide.md): recorded commands and findings.
- [Research plan](research-plan.md): original questions and experiment sequence.

## Current snapshot

| Item | Status |
| --- | --- |
| Default transcription model | Offline CNN-GRU, `outputs/cnn_gru/best-inference.pt`. |
| Latest completed experiment | Stage 21: saved fret predictions versus training-frequency priors. |
| Selected Stage 19 research candidate | Original-loss `control500`; the added loss was not adopted. |
| Next research step | Proposed Stage 22: three matched class-weighting arms, 450 updates total; not trained. |
| Last recorded full test run | 633 passed, one existing dependency warning, 2026-10-28. |

On reference-active frames, Stage 21 found `control500` best-active fret accuracy of 18.66%, versus 16.62% for the fixed training-majority prior. Equal-supported-position accuracy is 12.89% versus 9.38%. The model chooses its training-modal fret on 60.64% of those frames. This zero-training, zero-neural-inference diagnosis uses reused development recordings and oracle activity support; its scores are not end-to-end transcription accuracy.

Open the local [Stage 21 prior/confusion review](outputs/prior21/review.html) or [findings](outputs/prior21/findings.md). The [Stage 20 coverage review](outputs/coverage20/review.html) explains exposure counts; the [Stage 19 review](outputs/fret19/review.html) retains audio, saved notes and their timing clocks. Review assets are local files, ignored by Git.

## Transcribe a local guitar clip

From the project root in the existing environment, replace `my-guitar.wav` with your audio path:

```powershell
.\.venv\Scripts\python.exe -m cnn_gru transcribe "my-guitar.wav" --device cpu --seconds 12 --output "outputs/manual-transcription/prediction.mid"
```

This writes `prediction.mid` and `prediction.json` with timed string/fret predictions. The model uses future context and standard six-string tuning. The [detailed guide](docs/experiment-guide.md#cnn-gru-training-and-inference) covers the recorded training/inference workflow, constrained tablature, rhythm and other experiments.

## Setup

The recorded Windows setup requires `uv` and `git`; it creates a project-local Python 3.10 environment and installs the pinned baseline dependencies:

```powershell
.\scripts\setup.ps1
```

Use `-CpuOnly` for the CPU PyTorch installation. Setup can download Python, packages and the pinned MT3 adapter. Rhythm/notation and separation have optional dependency files and instructions in the [guide](docs/experiment-guide.md). The smaller [fresh-environment smoke check](docs/VERIFICATION.md#fresh-isolated-environment-transcription) installs only the offline transcription dependencies and runs on CPU.

A source checkout does not include completed training checkpoints, datasets or experiment outputs. Restore those ignored assets from a separate backup to use the saved models and full historical verification suite. The [experiment records](experiments/README.md#verification-and-backups) describe the tracked hash snapshot.

## Verification

In the recorded workspace:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
```

This checks all 21 indexed stages' paths, the latest snapshot's 693 artifact/code/input bindings, checkpoint identities, installed dependencies and the full test suite. Historical evidence stays unchanged; new logs go under `tmp/project-check/`. See [verification instructions](docs/VERIFICATION.md) for the integrity-only mode and separate fresh-environment transcription check.

## Repository map

| Path | Purpose |
| --- | --- |
| [docs/](docs/STATUS.md) | Current status and detailed historical guide. |
| [experiments/](experiments/README.md) | Stage index, machine-readable registry and compact verification record. |
| [configs/](configs/) | Recorded experiment settings. |
| [tests/](tests/) | Tests and small reference fixtures. |
| [scripts/](scripts/) | Setup, audit helpers and browser checks. |
| Stage packages such as [cnn_gru/](cnn_gru/), [streaming/](streaming/) and [fret19/](fret19/) | Preserved implementations and imports for completed experiments. |
| `data/`, `models/`, `outputs/` | Local datasets, weights and results; ignored by Git. |
| `.venv/`, `.venv-tf-check/`, `.cache/`, `tmp/` | Local environments, runtime files, caches and temporary work. |

Historical code and output paths stay in place because saved results bind them to exact hashes. `.gitattributes` preserves existing file bytes across Git checkouts. `.cache/python` contains the Python runtime used by the current environment; preserve it during disk cleanup.
