# Workspace verification

These checks preserve historical producer files and experiment outputs. New logs,
test scratch files and smoke predictions go under ignored `tmp/` directories.
They perform no acoustic training and do not regenerate stage reports or audits.

## One command for routine checks

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
```

The command checks:

- Source/config/report/review paths for all 21 indexed stages.
- The latest tracked snapshot's SHA256 and its exact-byte identity with the
  original local verification record.
- All 693 artifact/code/input bindings in the Stage 21 snapshot, the default checkpoint and
  all three checkpoints referenced by the hash-bound Stage 19 run record.
- Agreement between the indexed research selection and the saved run. A later
  diagnosis can preserve an earlier trained-model selection; its referenced
  model record must itself be hash-bound by the snapshot.
- Installed dependency compatibility with `uv pip check` and the full pytest suite.
- The same saved hashes again after the tests, to catch accidental changes.

Exit code 0 means every requested check passed. Exit code 1 means failure;
missing ignored assets fail explicitly. The printed `tmp/project-check/verify-*/`
directory contains `report.json`, dependency/test logs and `pytest.xml`.

For just the path/hash checks, without importing ML packages or running tests:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py --integrity-only
```

The shorter mode is labeled in its report. It does not imply that tests or
dependencies passed. A source-only checkout needs the ignored assets restored
from backup for either mode. The 693 bindings cover the latest snapshot, not every
file ever produced in all 21 stages. Existing historical audit/browser receipts
are checked as saved evidence; this command does not rerun the browser.

## Fresh isolated-environment transcription

```powershell
.\scripts\check_fresh_env.ps1
```

This creates a new Python 3.10 virtual environment under a unique
`tmp/fresh-transcription/run-*/env` directory. System/user site packages and
inherited Python import paths are excluded. Each inference package's import
location must be inside the new environment. The existing environment supplies
the base Python interpreter, and the source stays in this checkout.

Installation defaults to **offline**, using `.cache/uv`. It installs the recorded
`torch==2.7.1+cu126` wheel and the smaller [inference requirements](../requirements-inference.txt).
The CUDA-enabled wheel runs this smoke test on **CPU**; no GPU is required for the
transcription. An empty cache fails with installation logs. `-AllowDownloads`
explicitly enables package downloads if needed. A sandbox that blocks Windows
executable metadata updates may require permission for package installation.

The real `python -m cnn_gru transcribe` command processes the first 12 seconds of
the existing Rock2 development excerpt with the saved default checkpoint. The
check validates input/checkpoint hashes, nonempty output, finite note intervals,
standard-tuning string/fret/pitch identities, no overlapping frets on a string,
and exact agreement between independently parsed MIDI and JSON notes.

The run directory contains `smoke.json`, `prediction.mid`, `prediction.json`,
package installation/dependency logs and `requirements-resolved.txt`. Existing
model outputs are not overwritten. New environments are retained for inspection.
They are disposable independently of `.venv` and `.cache/python`.

This establishes a local environment/export smoke result using cached packages,
the workspace interpreter and local model/audio assets. It does not establish
clean-machine installation, a CPU-only PyTorch distribution, transcription
accuracy, fresh-data generalization or hardware capture latency. The full research
test suite continues to run in the original environment.
