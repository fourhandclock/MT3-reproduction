# Guitar transcription research

Low-compute guitar transcription experiments, completed through **Stage 23**. The default workflow is the offline CNN-GRU. The causal branch, constrained fingering, rhythm/notation and separation experiments are documented with their own controls and limits.

- [Current status](docs/STATUS.md): defaults, latest results, limitations and remaining work.
- [Experiment index](experiments/README.md): all 23 stages, code, configs and local reviews.
- [Detailed experiment guide](docs/experiment-guide.md): recorded commands and findings.
- [Research plan](research-plan.md): original questions and experiment sequence.

## Current snapshot

| Item | Status |
| --- | --- |
| Default transcription model | Offline CNN-GRU, `outputs/cnn_gru/best-inference.pt`. |
| Latest completed experiment | Stage 23: locked evaluation of four existing models on progression-3 recordings; zero training. |
| Selected research candidate | Stage 22 `uniform_ce650`; development selection, no default promotion. |
| Next implementation step | Proposed Stage 24: an explicit research inference CLI using the locked model; zero training. |
| Last recorded full test run | 688 passed, one existing dependency warning, 2026-10-28. |

Stage 23 preserves Stage 22's selected `uniform_ce650` checkpoint. Its test state/note-onset F1 is **0.0578 / 0.0832**, versus **0.0513 / 0.0892** for continued original loss: the state advantage persists, while the onset advantage does not. These existing recordings had historical evaluations and share players across splits. No test-driven reselection, calibration or default promotion occurred.

Open the local [Stage 23 audio review](outputs/holdout23/review.html) or [findings](outputs/holdout23/findings.md). The [Stage 22 development trial](outputs/balance22/review.html), [prior diagnosis](outputs/prior21/review.html) and [coverage audit](outputs/coverage20/review.html) retain the preceding evidence. Review assets are local files, ignored by Git.

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

This checks all 23 indexed stages' paths, the latest snapshot's 1,418 artifact/code/input bindings, checkpoint identities, installed dependencies and the full test suite. Historical evidence stays unchanged; new logs go under `tmp/project-check/`. Run browser QA separately from the full suite on this 8GB host. See [verification instructions](docs/VERIFICATION.md) for the integrity-only mode and separate fresh-environment transcription check.

## Repository map

| Path | Purpose |
| --- | --- |
| [docs/](docs/STATUS.md) | Current status and detailed historical guide. |
| [experiments/](experiments/README.md) | Stage index, machine-readable registry and compact verification record. |
| [configs/](configs/) | Recorded experiment settings. |
| [tests/](tests/) | Tests and small reference fixtures. |
| [scripts/](scripts/) | Setup, audit helpers and browser checks. |
| Stage packages such as [cnn_gru/](cnn_gru/), [streaming/](streaming/) and [holdout23/](holdout23/) | Preserved implementations and imports for completed experiments. |
| `data/`, `models/`, `outputs/` | Local datasets, weights and results; ignored by Git. |
| `.venv/`, `.venv-tf-check/`, `.cache/`, `tmp/` | Local environments, runtime files, caches and temporary work. |

Historical code and output paths stay in place because saved results bind them to exact hashes. `.gitattributes` preserves existing file bytes across Git checkouts. `.cache/python` contains the Python runtime used by the current environment; preserve it during disk cleanup.
