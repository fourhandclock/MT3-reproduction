# Detailed experiment guide

This guide preserves the commands, controls and findings recorded through Stage 19. Use [current status](STATUS.md) for the present default and remaining work, or the [experiment index](../experiments/README.md) to find a stage.

The opening baseline sections below describe **Stage 1**. Each later section describes its own completed experiment, budget and data scope. A next-step proposal in an earlier section records what was planned then; a later stage may already have executed it. Reused validation results are development evidence, and timing scopes differ across stages.

The full recorded workspace includes ignored data, checkpoints and outputs. Those assets require a separate backup; cloning the source repository alone does not restore them. See [repository preservation notes](STATUS.md#preserving-the-workspace).

## Stage 1: pretrained baselines


- **MT3 reference-style frontend (`mt3-reference`, default):** the unofficial `kunato/mt3-pytorch` port, through `openmirlab/mt3-infer` at commit `d8cbd287470291dee335f84315c5a9b9ff6e6c8a`, with an independently implemented frontend following Google's magnitude/end-padding/mel-weight operations. Instrument filtering is disabled and the checkpoint hash is verified. This is not a full independently validated reproduction of Google's JAX model/decoder.
- **Unmodified port (`mt3-pytorch`, optional):** retains the wrapper's centered power spectrogram. Its two-excerpt results are retained under `outputs/smoke`; these frontend differences must not be interpreted as failures of the original MT3 model.
- **Basic Pitch:** Spotify's released 0.4.0 ONNX model, using CPU inference with four threads. Decoding defaults are retained; its minimum note length is 127.7 ms.
- **Data:** 12 deterministic GuitarSet microphone excerpts, spanning six players, five styles and comping/soloing. Selection uses progression 3, following MT3's published validation convention.
- **Budget:** first 12 seconds of each excerpt; note onsets and frames scored over the first 10 seconds, leaving two seconds of release context. No training, full dataset download, source separation, string/fret prediction or meter estimation.

GuitarSet appears in the models' research/training context. Exact Basic Pitch overlap is unresolved. This is a diagnostic baseline, not evidence of generalization to unseen electric guitars, nor a full paper benchmark. MT3 uses CUDA if available and Basic Pitch uses CPU: timing compares these deployments, not the architectures on equal hardware.

## Setup on Windows

The setup script keeps Python, packages and caches inside this workspace. It uses Python 3.10 because the selected Basic Pitch ONNX installation avoids TensorFlow on that version. It requires `uv` and `git`.

```powershell
.\scripts\setup.ps1
```

For a computer without a compatible NVIDIA driver, use `-CpuOnly`. The GPU PyTorch wheel is approximately 2.5 GiB; setup has a larger disk footprint than the model checkpoints. The actual audio subset is downloaded using ZIP byte ranges, rather than retrieving the approximately 657 MB complete microphone archive.

```powershell
.\.venv\Scripts\python.exe -m baseline doctor
.\.venv\Scripts\python.exe -m baseline prepare
.\.venv\Scripts\python.exe -m baseline run --limit 2 --output outputs/quickcheck
.\.venv\Scripts\python.exe -m baseline run
```

`prepare --data-only` and `prepare --models-only` can be run separately. Partial downloads require HTTP range support and internet access. Selected ZIP members are CRC32-validated, and individual SHA256 values are recorded. Full-archive checksums are recorded for provenance but cannot be validated without the complete archives.

For CPU inference, append `--device cpu`. To run one baseline, use `--models basic-pitch` or `--models mt3-reference`. Use `--models mt3-pytorch` only to inspect the unmodified port. Run models sequentially on this 8 GB RAM machine. Close memory-intensive applications if loading or preprocessing becomes slow; GPU VRAM and system RAM are separate budgets.

Completed predictions are reused when configuration, selected data, environment and pipeline source hashes match. Use another `--output` directory for a changed experiment, or `--overwrite` to regenerate selected outputs. Cached result files are not treated as new timing measurements.

## Results

Open `outputs/stage1/report.md` for the protocol, summary and error ranking. Open `outputs/stage1/review.html` in a browser for audio and overlaid reference/prediction intervals. No server or external assets are required. A plot's upper bars are reference notes and lower bars are predicted notes; hover for pitch and timestamps. The audio provides the same starting time as the plots.

Other outputs:

- `outputs/stage1/metrics.csv`: per-excerpt measurements.
- `outputs/stage1/summary.json`: macro/micro metrics and processing totals.
- `outputs/stage1/run.json`: configuration, software, model provenance and timing scope.
- `outputs/stage1/<model>/*.mid`: transcription MIDI files.
- `outputs/stage1/<model>/*.json`: reference notes, predictions, one-to-one matching pairs and errors.
- `outputs/stage1/basic-pitch/*.npz`: onset/note/contour probabilities.
- `data/manifest.json`: exact selected source IDs and file checksums.
- `models/mt3_pytorch/provenance.json`: checkpoint and configuration provenance.
- `outputs/hardware.json`: detected hardware and runtime versions.
- `requirements-resolved.txt`: installed-version audit; use the setup script for the local MT3 adapter source, not a PyPI replacement.

Regenerate the report without inference:

```powershell
.\.venv\Scripts\python.exe -m baseline report
```

## Evaluation details

Reference MIDI-note values are rounded to the closest semitone. String/fret labels are retained for later work but are not predicted here. Matching uses `mir_eval`: pitch tolerance 50 cents, onset tolerance 50 ms, offset tolerance `max(50 ms, 20% of reference duration)`. Frame scoring uses a 62.5 Hz grid and all non-drum pitches. No reference-based guitar-program filtering is applied to MT3 predictions.

The offset metric excludes reference notes whose release lies beyond the supplied audio; their onset-matched predictions are excluded from that metric as well. Count and inspect these censored notes before comparing with full-track published results. Later-context note onsets do not affect prefix note metrics, but sustained activity is included in frame metrics over the scored interval.

Warm processing measurements include file reads, resampling, acoustic features, inference, decoding and MIDI serialization. Model loading, a two-second warmup, dataset downloads and report/plot generation are excluded and documented. Processing/audio below 1 means faster than the duration of the supplied audio; it does not imply live streaming or establish onset latency.

The error report separates missed/extra notes and onset-matched offset errors. Octave and timing candidate counts are simple non-exclusive heuristics, so listening remains necessary. Ordinary note metrics collapse instrument labels; the report additionally preserves MT3's program counts and drum predictions.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests check metric boundaries, observed versus censored releases, empty predictions, frame-grid placement, string-label preservation and deterministic sampling across naming styles. Temporary test files are confined to `tmp/pytest-stage1`.

### Checking the frontend against Google's code

`tests/fixtures/mt3_frontend.npz` contains reference outputs generated by Google's actual `spectral_ops.py` under TensorFlow 2.15.1 on silence, noise, an impulse and a short GuitarSet signal. The source revision/hash and configuration are preserved beside the fixture. Ordinary tests use these saved outputs without importing TensorFlow. The verified settings include 512 mel bins from 20 Hz to **7,600 Hz**, not the 8,000 Hz Nyquist default of the unmodified port.

To regenerate them, use a separate CPU-only verification environment. It is not part of normal inference or training:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.cache\python'
uv venv --python 3.10 .venv-tf-check
uv pip install --python .venv-tf-check\Scripts\python.exe tensorflow==2.15.1 gin-config==0.5.0 numpy==1.26.4
.\.venv-tf-check\Scripts\python.exe scripts/make_frontend_fixture.py
```

The fixture generator uses the fixture's saved guitar samples when available, or `tmp/frontend-guitar.npy` when generating the fixture for the first time. Those samples are the first 4,096 samples of the selected Funk recording resampled to 16 kHz. Run the TensorFlow check separately from MT3 inference on a machine with limited RAM. The third-party model source is retained unchanged.

## CNN-GRU training and inference

The custom model is implemented in `cnn_gru/`, with settings in `configs/cnn_gru.json`. It has **935,081 parameters**: three frequency-pooling CNN blocks, a 128-dimensional projection and two bidirectional GRU layers with 128 units per direction. Heads predict six string/fret distributions, six onset probabilities, six offset probabilities and 49 auxiliary pitches.

Its input is a centered CQT with 216 bins (36 per octave), starting at E2, at 16 kHz with a 160-sample hop. Two-second crops contain 200 frames. Training and inference use the same train-only feature normalization. The recurrent layers remain float32 while CUDA AMP can accelerate the CNN/projection. This version uses future context and performs offline transcription.

Run the pipeline in the existing environment; no additional packages are required:

```powershell
.\.venv\Scripts\python.exe -m cnn_gru prepare
.\.venv\Scripts\python.exe -m cnn_gru cache
.\.venv\Scripts\python.exe -m cnn_gru train
.\.venv\Scripts\python.exe -m cnn_gru evaluate
```

The default completed run is in `outputs/cnn_gru`. A fresh training experiment needs its own `--output` directory; the existing checkpoint is protected against accidental overwrite. `train --resume` resumes an interrupted run only when its configuration, manifest and source revision match, retaining the global step cap. The completed run already spent its 500-step budget.

There are 48 training recordings from progression 1 and 12 validation recordings from progression 2. The stage-1 progression-3 excerpts are used for final evaluation only. Normalization and class-balancing statistics use training recordings only. Each epoch draws 400 random crops, making 100 updates at batch size 4; training stops at 5 epochs, 500 updates or early stopping. Players appear in all splits, so this does not measure unseen-player performance.

`best.pt` and `last.pt` preserve the original training run. `best-inference.pt` contains the same selected model weights with an updated decoder calibrated on validation only, after preventing release refinement from overlapping the next note on a string. The decoding repair added zero optimizer steps. For a fresh run, `evaluate` uses `best.pt` unless a validated `best-inference.pt` exists. The calibration command supports a decoder revision while preserving training provenance:

```powershell
.\.venv\Scripts\python.exe -m cnn_gru calibrate
.\.venv\Scripts\python.exe -m cnn_gru evaluate
```

Use the trained prototype on an audio file:

```powershell
.\.venv\Scripts\python.exe -m cnn_gru transcribe "path\to\guitar.wav" --seconds 30 --output outputs\my-guitar.mid
```

The MIDI has a track per string; the adjacent JSON contains onset/offset seconds, string indices, frets and pitches. The fixed tuning is standard six-string guitar, with frets 0-24. Note pitches follow open-string MIDI pitch plus fret. The model does not infer alternate tuning, score rhythms, bends, hand span or time signatures.

Read `outputs/cnn_gru/report.md` for training history and the comparison with pretrained models. Open `outputs/cnn_gru/evaluation/review.html` for audio/note plots, MIDI and timed tablature CSVs. `evaluation/run.json` records the selected checkpoint, decoder thresholds, software and source hashes. Processing-time measurements recompute CQT from audio; they do not use the training feature cache.

## Training budget

Your limited-compute preference is implemented in `configs/cnn_gru.json`: **5 initial epochs, at most 500 optimizer steps, batch size 4, 2-second crops, early-stopping patience 2, and no worker subprocesses initially on Windows**. The first run completed 5 epochs and 500 updates. Report it as a limited-budget result, and compare models under the same update budget before drawing architecture conclusions.

## Matched CNN-transformer comparison

Stage 3 replaces the GRU temporal module with four small bidirectional transformer encoder blocks. It reuses the existing dataset and CQT cache; no new downloads or dependencies are needed.

```powershell
.\.venv\Scripts\python.exe -m cnn_transformer train
.\.venv\Scripts\python.exe -m cnn_transformer compare
```

The completed experiment is under `outputs/cnn_transformer`. Training a fresh experiment requires a new `--output` directory; `--resume` preserves the global step budget after an interruption. The recorded transformer run already completed 5 epochs and 500 updates. `compare` loads the existing GRU and transformer checkpoints and adds no training updates.

The transformer has **937,897 parameters**, versus the GRU's **935,081**, a difference of about 0.3%. Both start with bit-identical CNN, projection and output-head weights. The shared module initialization order is preserved, and no trained GRU weights are transferred. The training recipe is the same: source recordings, sampled crops, gain augmentation, label targets, normalization, batch size, AdamW, cosine schedule, loss and validation-selection rule. The epoch loop is checked for AST equivalence.

The temporal transformer uses 128-dimensional inputs, four attention heads, feed-forward width 192, sinusoidal position encoding, pre-normalization and dropout 0.1. A learned output projection returns 256 dimensions to the same heads as the GRU. Feed-forward width is chosen to match parameter counts. Each encoder layer is initialized independently, instead of retaining identical cloned initial values.

Both temporal modules use float32; their shared CNN/projection/heads use the same CUDA AMP policy. Both are offline and use the same overlapping two-second windows. Neither uses an autoregressive note-token decoder. This compares temporal acoustic encoders, rather than reproducing MT3's encoder-decoder architecture.

Open `outputs/cnn_transformer/report.md` for the result and controls, and `outputs/cnn_transformer/comparison/review.html` for paired audio/note plots, MIDI and timed tab CSVs. The comparison measures both pipelines in the same process/device after warming the full 12-second window batch and MIDI output path. It also records three inference-only passes per excerpt and reports their median, separating network work from CQT and note decoding. These are offline throughput measurements, not live latency.

The histories, best/last transformer checkpoints, input/source hashes, shared-initialization audit and selected validation thresholds are preserved. One seed, one small transformer configuration and a 500-update budget do not establish a universal architecture ranking.

## Tone/effect robustness

Stage 4 uses the CNN-GRU selected in the matched comparison. It evaluates the frozen clean-trained model on stronger EQ, drive and an unseen combined chain, then compares it with one new from-scratch augmentation run under the same five-epoch / 500-update cap.

```powershell
.\.venv\Scripts\python.exe -m robustness prepare
.\.venv\Scripts\python.exe -m robustness external
.\.venv\Scripts\python.exe -m robustness baseline
.\.venv\Scripts\python.exe -m robustness train
.\.venv\Scripts\python.exe -m robustness evaluate
```

The completed trial is under `outputs/robustness`; `train` defaults to its `augmented` subdirectory. It preserves the original clean-model checkpoints. An existing trial is protected against accidental additional training; a new trial requires a new output path. The current run already spent its 500-update budget. Regenerate reports without inference or training using `python -m robustness report` in the project environment.

Training draws 50% clean, 25% mild EQ and 25% moderate-overdrive views of the same 48 training recordings. Source/crop/gain draws, labels, architecture, normalization, loss, optimizer and schedule are retained. An independent RNG selects the view. Validation uses the same twelve progression-2 recordings under clean/mild-EQ/moderate-drive conditions for both models' decoder calibration; the trial's checkpoint selection uses this mixed validation objective.

EQ and drive are rendered on the complete polyphonic waveform. Drive uses a four-times-oversampled tanh waveshaper and optional low-pass coloration. Zero-phase filters/centered resampling preserve sample count and introduce no fixed label shift. RMS is matched, with a peak ceiling when necessary. These are analytic offline simulations on acoustic recordings, not real amplifier models. They may change perceptual attack/release clarity even though performed-note labels remain fixed.

The separate external dataset is [EGSet12](https://zenodo.org/records/11406378), by Hegel Pedroza, Wallace Abreu, Ryan M. Corey and Iran R. Roman, licensed CC BY 4.0. Only its twelve WAV/JAMS pairs are downloaded, about 107 MB, and their published MD5 checksums are verified. No pretrained model from that record is loaded. EGSet12 is evaluation-only: no training, normalization, threshold selection or early stopping uses it.

The dataset contains one electric guitarist and recording setup, with amp/microphone capture and no additional effects. The first 12 seconds are supplied to the model and the first 10 seconds are scored. Provided note/string labels are retained unchanged; their timing alignment has not been independently audited. Primary 50 ms onset scores and a preset 100 ms timing diagnostic are reported separately.

Read `outputs/robustness/report.md` and open `outputs/robustness/evaluation/review.html` for a condition filter, processed audio and paired note plots. Full per-record MIDI, timed tablature CSV, note/error JSON, model/configuration hashes and validation criteria are retained. Stress evaluation reuses cached features and is not a deployment timing benchmark.

The first trial did not improve the synthetic-effect average, and real-electric onset F1 fell from 0.291 to 0.275. Retain the original clean-trained CNN-GRU as the current default. This is a negative result for this particular short-budget recipe, not evidence that augmentation is generally ineffective. A stronger transfer study needs independent real-electric training/validation material and fresh recording conditions.

## Constrained guitar fingering

Stage 5 converts fixed pitch/onset/offset events to consistent string/fret positions using a bounded beam search. It adds no model training. The default fretboard is standard six-string tuning, capo 0 and physical frets 0-24. Candidate positions preserve every input pitch and interval.

```powershell
.\.venv\Scripts\python.exe -m fingering select
.\.venv\Scripts\python.exe -m fingering evaluate
```

`select` chooses the acoustic-prior weight on progression-2 reference-note performances, using the frozen CNN-GRU's fret-state probabilities. Ground-truth strings are removed before search and used only to evaluate recovery. The declared span/finger limits remain fixed. `evaluate` compares lowest-fret, constrained pitch-only and acoustic-guided assignments on the progression-3 excerpts, using both reference-note events and predicted CNN-GRU events.

The search pins sustained notes to their assigned strings, prohibits simultaneous notes on one string, limits fretted-note span to five and uses a four-finger-group proxy that allows partial/full barres beneath higher fretted notes. Open or lower-fretted active strings block a barre. Soft costs penalize position travel, fast shifts, span and high frets, with a small open-string preference. These checks are simplified feasibility proxies, not a complete biomechanical guarantee; thumb use, picking/muting and advanced techniques are not modeled.

An unplaceable/pruned note is flagged with null string/fret fields, while its pitch and timing remain intact. It remains in pitch metrics and an unassigned MIDI lane. JSON is the exact interval source; MIDI is quantized, with separate lanes to preserve overlapping duplicate pitches. Alternative fingerings can be valid without matching the original guitarist's choice.

Use the acoustic model and constrained decoder on an audio file:

```powershell
.\.venv\Scripts\python.exe -m fingering transcribe "path\to\guitar.wav" --seconds 30 --output outputs\my-constrained-tab
```

For pitch-only retabulation of existing MIDI or a note JSON:

```powershell
.\.venv\Scripts\python.exe -m fingering retab "path\to\notes.mid" --output outputs\my-retab
```

Both commands produce note JSON, timed tab CSV and MIDI. The learned prior supports the acoustic model's standard tuning/capo 0; alternate tunings/capos can be specified in `configs/fingering.json` for pitch-only retabulation. Dense passages, unresolved notes and physically unusual positions need musician review.

Open `outputs/fingering/report.md` and `outputs/fingering/evaluation/review.html` for timed six-line tabs and audio. The first run gives 0.834 string-aware onset F1 when reference notes are provided. With predicted notes, string-aware F1 changes from 0.431 to 0.424, but the assigned positions meet the configured constraints and three notes are explicitly unresolved. Pitch onset F1 stays 0.522. This is a feasibility tradeoff, not a correction of transcription errors. No training updates are added.

## Rhythm, chord drafts and editable notation

Stage 6 adds a separate pretrained Beat This **small0** beat/downbeat model, a fixed chord-template baseline and deterministic score generation. It adds **zero training epochs or optimizer updates**. The 8.45 MB checkpoint has 2,099,960 parameters; models run sequentially on the existing laptop GPU.

Install the extra inference/validation dependencies in the project environment, then download the pinned checkpoint:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-rhythm.txt
.\.venv\Scripts\python.exe -m rhythm prepare
.\.venv\Scripts\python.exe -m rhythm evaluate
```

The completed run is in `outputs/rhythm`. Open `review.html` for twelve audio excerpts, four oracle/predicted note-and-beat combinations, clickable timed tabs, synchronized bar highlights, MusicXML/JSON/CSV downloads and beat/meter/chord correction controls. Read `report.md` and `findings.md` for the results. The viewer uses local audio and embedded score data; it needs no hosted service.

Beat This small0 was trained on GuitarSet. Its scores here are integration diagnostics, **not unseen-data rhythm performance**. All 24 validation/test excerpts have 4/4 reference labels. Test beat/downbeat F1 is 0.661/0.578 at a 70 ms tolerance, with about 50% provisional meter duration agreement. Beat F1 is 0.908 on comping and 0.415 on solos. Detected pulse counts do not uniquely identify notation denominators or compound meter; correction is part of the current workflow.

The lightweight chord baseline uses pitch-class durations from notes over two-beat context and Viterbi smoothing. It proposes major/minor and three seventh qualities plus silence. Reduced major/minor duration accuracy is 0.488 on reference notes/beats and 0.249 on predicted notes/beats, with average reference coverage 87.7%. GuitarSet's performed-chord labels partly depend on lead-sheet roots/boundaries and string-note annotations. An isolated guitar solo may not specify full-song harmony.

Transcribe an isolated guitar file through the frozen CNN-GRU, constrained fingering and rhythm stage:

```powershell
.\.venv\Scripts\python.exe -m rhythm transcribe "path\to\guitar.wav" --seconds 30 --output outputs\my-score
```

This produces `my-score.musicxml`, score JSON/CSV, `my-score-beats.json`, and separate `my-score-raw.json/.csv/.mid`. MusicXML has concert-pitch and six-line guitar-tab staves, independent sustained voices, rests, ties, meter and local tempo directions. Positions that could not be assigned remain in the pitch staff and JSON. Raw note seconds and pitches are retained separately from score quarters. The default standard tuning/capo 0 is inherited from the current acoustic model.

Quantization chooses a straight-sixteenth or eighth-triplet lattice per bar. Written durations approximate audible release; intended articulation, swing, arbitrary tuplets and key-aware enharmonic spelling are not inferred. Minimum durations, large timing shifts and new same-string overlaps are flagged. End-to-end raw onset F1 stays 0.522; snapped onset F1 is 0.452 against performed notes. The latter is a timing diagnostic, not engraved-score accuracy.

In the viewer, download **Raw note JSON for corrections** and **Download corrections** after editing. Regenerate a score from those files:

```powershell
.\.venv\Scripts\python.exe -m rhythm score "path\to\raw.json" --corrections "path\to\corrections.json" --output outputs\corrected-score
```

Correction JSON contains `beats`, `downbeats` (seconds), `duration_seconds`, optional `meter` with `numerator`, `denominator`, `pulse_quarters`, optional `subdivision` (4 or 3), and optional nonoverlapping `chords` with `start`, `end`, `label`. For 6/8 tracked as two dotted-quarter pulses, use numerator 6, denominator 8 and pulse_quarters 1.5. Chord labels use sharp roots C/C#/D/.../B and qualities `maj`, `min`, `7`, `maj7`, `min7`, or `N` for silence. Manual chord events replace the template proposals. Raw source attacks must lie inside the correction duration.

Regenerate reports or notation from saved results without inference/training, and validate against the official schema:

```powershell
.\.venv\Scripts\python.exe -m rhythm export
.\.venv\Scripts\python.exe -m rhythm report
.\.venv\Scripts\python.exe -m rhythm.validate --prepare-schema
.\.venv\Scripts\python.exe -m rhythm.validate
```

All sixty evaluation scores pass MusicXML 4.0 XSD validation, preserve the pitch/tick multiset of source attacks on the pitch staff, and conserve bar lengths across voices. Tests cover variable tempo, pickups, meter changes, compound-meter overrides, triplets, ties, short events, overlaps, silence and corrections. UI checks cover all 48 test excerpt/scenario combinations, seeking/highlighting and correction downloads. These establish conversion/integration behavior; an independent rhythm dataset and musician-reviewed written scores are still needed.

## Guitar separation and controlled mixtures

Stage 7 takes the mixtures branch first. It adds the official frozen **Demucs 4.0.1 htdemucs_6s** guitar separator before the existing frozen CNN-GRU. There are **no training updates or new threshold choices**. The checkpoint is 55.0 MB with 27,414,996 parameters. The recorded RTX 3050 run uses the original 7.8-second segment, shifts 0, overlap 0.25, float32 and one file at a time. GPU input chunks are processed while waveform accumulation stays on CPU; the separator is released before transcription starts.

Install/download, render the controlled inputs, then evaluate:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-mixtures.txt
.\.venv\Scripts\python.exe -m mixtures prepare
.\.venv\Scripts\python.exe -m mixtures evaluate
```

The completed experiment is in `outputs/mixtures`. Open `review.html` to compare gain-matched isolated guitar, mixtures and separated guitar on the same audio timeline. `report.md`, `metrics.csv`, `run.json` and `data-manifest.json` preserve the protocol, exact source identities and results. `python -m mixtures render` and `python -m mixtures separate` run those phases independently. Existing outputs reject changed source/configuration/renderer identities; use a fresh output directory for a new recipe. Separation caches verify checkpoint, wrapper, input and output hashes. `python -m mixtures report` regenerates the review from saved results without inference or training.

The twelve progression-3 excerpts are combined with deterministic procedural bass, keyboard/pad and percussion at guitar-to-accompaniment RMS ratios +6, 0 and -6 dB. Only the instructed beat/chord lead sheet guides rendering; original guitar note/string labels are not used to synthesize the backing and never enter model inference. A common peak scalar preserves the source ratio and guitar timing. Each ratio has an isolated-guitar control at that exact gain. Additional controls pass clean guitar through the separator and transcribe accompaniment alone. This creates 144 paired records across twelve fixed conditions; it is a controlled synthetic experiment, not a natural full-band test.

| Guitar / backing ratio | Direct mixture onset F1 | Separated guitar onset F1 |
| --- | ---: | ---: |
| +6 dB | 0.313 | 0.484 |
| 0 dB | 0.215 | 0.465 |
| -6 dB | 0.137 | 0.437 |

Clean-guitar raw onset F1 is 0.525 and clean-through-separator F1 is 0.522. This stage evaluates raw JSON intervals before MIDI tick rounding; the earlier 0.522 baseline evaluated a MIDI round trip. The inspected discrepancy decodes identical raw events from the original probability cache. All new conditions share the raw-event protocol. The raw CNN-GRU emits 535 scored-prefix false-positive notes on accompaniment alone, showing that specialization does not provide reliable instrument attribution by itself. This no-guitar count tests the raw transcriber, not the complete separator pipeline.

Warm separation takes roughly 0.75 seconds per twelve-second excerpt; separation plus fresh CQT/CNN/decoding has an offline processing/audio ratio around 0.071. Peak allocated GPU tensors are 619.4 MiB for the separator and 69.7 MiB for the acoustic phase. Model loading, synthesis, exported files, fingering beam search, rhythm and score conversion are recorded separately or excluded from this timing scope. The original bidirectional acoustic model and separator need future context, so these are throughput results rather than streaming latency.

Run an audio file through separation and constrained guitar transcription:

```powershell
.\.venv\Scripts\python.exe -m mixtures transcribe "path\to\song.wav" --seconds 30 --output outputs\song-tab
```

It exports a float WAV guitar stem, note JSON/CSV/MIDI and a run record. Input files may be mono or stereo; the separator preserves stereo evidence before returning a mono guitar stem to the existing acoustic model. Output stems are not independently peak-normalized or shifted to match labels. JSON preserves exact note seconds; MIDI uses ticks.

To also generate a MusicXML draft:

```powershell
.\.venv\Scripts\python.exe -m mixtures transcribe "path\to\song.wav" --seconds 30 --score --output outputs\song-tab
```

The optional rhythm model reads the original mixture; note/chord/tab inference uses the extracted guitar. Models run sequentially. Outputs add `song-tab-score.musicxml/.json/.csv` and beat JSON. A complete twelve-second example is saved under `outputs/mixtures/example/rock-mix*`, including MusicXML validated against the official schema.

The official guitar source is experimental and may contain several guitars, piano/bass leakage or missing attacks. The observed benefit on short synthetic acoustic mixtures does not establish real-song or electric-guitar accuracy. Real recordings with isolated guitar references, disjoint songs and musician-checked notes remain the next validation priority. Beat/meter/chord correction limitations from stage 6 still apply; source separation does not repair them automatically.

Use `python -m mixtures.audit` to verify saved waveform/source/label identities, complete unique condition coverage and note-export agreement. The first run's inference hashes remain recorded separately from later cache-validation checks. The checkpoint is loaded only from its verified official file; its legacy format requires explicit local class deserialization under PyTorch 2.7.

## Causal streaming research prototype

Stage 8 adds a separate **931,145-parameter causal CNN-GRU**, with a past-only log-frequency STFT frontend, temporal convolution caches, per-frame normalization, carried unidirectional GRU state and incremental note-on/note-off events. It leaves the existing offline pipeline as the default. The causal branch is a **file-replay prototype**, not an actual microphone application.

The total stage budget is 500 updates: an initial 100-update trial was discarded after target-grid/EOF edge-case fixes, then a corrected model started from scratch for four epochs / 400 updates. Its training plus validation took 255.9 seconds, with peak allocated GPU tensors 280.3 MiB. Post-training decoder calibration adds no optimizer updates and uses progression-2 validation only. The original/source-repaired histories and checkpoint hashes remain recorded separately. Further training is not automatically added to this run.

```powershell
.\.venv\Scripts\python.exe -m streaming prepare
.\.venv\Scripts\python.exe -m streaming train
.\.venv\Scripts\python.exe -m streaming calibrate
.\.venv\Scripts\python.exe -m streaming evaluate
```

No new dependencies or audio data are downloaded. Causal feature/target caches live under `data/streaming`; the selected checkpoints and history are under `outputs/streaming/model`. Existing training checkpoints are protected against accidental fresh training. Use a different `--output` root for an explicitly requested new experiment; this corrected recipe remains capped at 400 updates. `python -m streaming report` rebuilds the review from saved evaluation records without inference or training.

The completed result is **not accurate enough to replace the offline model**:

| Packet duration | Causal onset F1 | Matched onset delivery p50 / p95 | Matched onsets | Processing/audio |
| --- | ---: | ---: | ---: | ---: |
| 20 ms | 0.116 | 87.2 / 143.8 ms | 70 | 0.300 |
| 40 ms | 0.116 | 98.4 / 161.2 ms | 70 | 0.175 |
| 80 ms | 0.116 | 120.2 / 196.3 ms | 70 | 0.102 |

The frozen offline raw-event comparison gives onset F1 0.525. This is not an isolated architecture comparison: frontend, normalization, recurrence, decoding and final training count differ. Packet sizes preserve the same estimated note intervals in this sample, while larger packets increase delivery delay. Delay quantiles pool correctly matched events only; missed notes have no finite delivery time. Matched release counts are smaller, and EOF-generated releases are excluded from release-feedback latency. Read counts, missed/extra notes and F1 alongside the timing table.

Open `outputs/streaming/review.html` for audio, note estimates, source-to-delivery links, a delivery-clock event log and four-mode comparison. `report.md`, `evaluation/summary.json`, `evaluation/run.json` and per-example metrics preserve the measurements. The viewer distinguishes estimated source times from actual event availability in the replay clock.

Replay a file in packets and export timed tabs, MIDI and a separate event ledger:

```powershell
.\.venv\Scripts\python.exe -m streaming replay "path\to\isolated-guitar.wav" --seconds 30 --packet-ms 20 --output outputs\causal-guitar
```

The command first loads/resamples the file offline, then feeds mono 16 kHz PCM to the streaming consumer. For application integration, `streaming.inference.PCMStream(model, normalization, config, parameters, device)` accepts new **mono float32 16 kHz samples** through `push(samples)`, returning immutable event copies. `finish()` marks remaining releases as censored EOF events. Start a fresh `PCMStream` for each recording. Native capture, causal rate conversion, audio-driver/UI scheduling, separator, global fingering search, meter and notation are outside this causal loop.

The standalone replay command does not explicitly warm the model; its run metadata records that startup behavior. The comparative benchmark warms each packet shape first. JSON is the exact source-interval record. MIDI projects to 480 ticks per quarter at 120 BPM and gives each published attack at least one tick; nine adjustments across the three causal packet modes prevent three underlying tiny intervals from disappearing. Extra MIDI lanes preserve quantized same-pitch overlaps. These projection changes do not alter scored JSON notes or event timing.

Each feature frame uses only the 2048 samples before its right edge (128 ms of past audio), with a 160-sample / 10 ms hop and no end padding. Temporal convolutions consume only past frames. Group normalization folds each time frame into a separate sample, preventing the original model's normalization from averaging over future time. Global mean/std statistics come only from training recordings. Inference uses float32 and carries two past frames per convolution plus GRU state between packets.

Note activity confirms across four frames, releases across two, and same-fret retriggers need a following confirmation frame. The 40 ms minimum-note setting guards retriggers; it is not a minimum duration for every emitted note. Source onsets may be backdated by the validation-selected 60 ms value, but delivery times remain unchanged. The recorded virtual delivery clock is `max(previous_finish, packet_arrival) + measured_processing_time`; it accounts for queue buildup during every packet, including intervals with no emitted notes. It is not measured microphone-to-speaker latency.

Tests cover future-prefix invariance, exact frontend chunk equivalence, model state equivalence/reset, gradient flow, right-edge target indexing, observed-boundary censoring, immutable events, EOF retriggers, partial hops, empty replay and latency censoring. The short training result suggests that acoustic/decoder quality needs improvement before native live capture work is useful; it does not identify one component as the sole cause of the gap.

`python -m streaming.audit` verifies all 48 examples, model-parent weight equality, paired incremental events, exact JSON/scored-interval agreement, source checksums and preserved MIDI attack counts. The completed viewer passed checks for 48 mode/excerpt combinations, 144 export links, 48 JSON downloads and source/delivery clock coordinates. At stage-8 completion, the full test suite passed 109 tests.

## Frozen streaming-model diagnosis

Stage 9 examines the stage-8 causal model with **zero new training epochs or optimizer updates**. Its weights, frontend, train-only feature normalization and recurrent context remain frozen. Validation head audits, reference-head decoder checks and a small decoder-only search identify why fast incremental events have poor note accuracy. This stage adds an experimental repair mode; it does not replace either the causal checkpoint or the default offline model.

The exact string/fret head is the main observed bottleneck. On progression-2 validation, its mean frame F1 is **0.0576** using raw argmax, or **0.0903** using the existing activity threshold of 0.35. Detecting whether a string is active is easier: activity F1 is **0.5421** under that threshold. These are different frame-level tasks, not assembled note F1. Predicted active-class runs have a median of eight frames and about 21% last fewer than four frames; reference runs have a median of 26 frames and none in this subset are shorter than four.

The oracle experiment crosses predicted/reference state and onset heads with both 0 and 60 ms source backdates. Each column below holds backdating, thresholds, four-frame confirmation and two-frame release confirmation fixed. Offset and auxiliary pitch heads are unchanged and ignored by the existing decoder.

| Validation heads | Note onset F1, 0 ms backdate | Note onset F1, 60 ms backdate |
| --- | ---: | ---: |
| Predicted state + predicted onset | 0.08372 | 0.11105 |
| Predicted state + reference onset | 0.11585 | 0.14276 |
| Reference state + predicted onset | 0.85825 | 0.24666 |
| Reference state + reference onset | 0.96121 | 0.35837 |

Reference heads are privileged diagnostic inputs; no reference value enters ordinary transcription. Fixed-backdate comparisons support prioritizing acoustic state/fret quality. Exact reference state also needs its own timing convention: the acoustic model's selected 60 ms correction harms an already aligned oracle grid. Even the zero-backdate oracle matches only **671 of 753** scored notes; its 82 misses are same-pitch repetitions. With reference state and predicted onsets, 236 of 359 adjacent same-pitch repetitions are missed. Retrigger evidence and guards still need attention. Oracle packet-arrival delays are zero-compute structural lower bounds, not measured model or microphone latency.

A context audit compares continuous 12-second inference with resetting convolution/GRU state every two seconds. Exact string/fret frame F1 changes from **0.0903 to 0.0854**, while note onset F1 stays near **0.111**. Segment scores do not show monotonic deterioration. This gives no strong evidence that retained-state drift alone explains the gap; the reset also changes convolution context, so it is not an isolated GRU test.

The decoder search evaluates 18 validation configurations: state-probability EMA weights 1, 0.5 and 0.25; confirmation lengths four and two; and source backdates 60, 0 and 120 ms. Thresholds and acoustic probabilities are fixed. Validation selects EMA **alpha 0.5**, four-frame confirmation and 60 ms backdate, improving mean onset F1 from **0.11105 to 0.11909**. On the twelve reused progression-3 excerpts, however, onset F1 changes from **0.11636 to 0.11487**, and onset-plus-offset F1 from 0.02824 to 0.02440. The repair is **not promoted**. These reused excerpts provide an engineering comparison, not a fresh generalization test, and no further tuning uses their outcome.

Run the decoder-only selection, comparison and saved report in the project environment:

```powershell
.\.venv\Scripts\python.exe -m diagnosis9 select
.\.venv\Scripts\python.exe -m diagnosis9 evaluate
.\.venv\Scripts\python.exe -m diagnosis9 report
.\.venv\Scripts\python.exe -m diagnosis9 audit --recompute
```

The default `--output` root is `outputs/diagnosis9`. `audit` checks saved provenance and exports; `audit --recompute` also verifies the selected and identity decoders against cached validation probabilities without training. `report` rebuilds the local review from saved results. Open `outputs/diagnosis9/review.html`, `report.md` and `findings.md`; `selection.json`, diagnostic JSONs and `evaluation/run.json` preserve the selected recipe, scores and source/input/checkpoint hashes.

Verification passes **159 tests**. The independent saved-result audit checks all 24 test cases, 1,620 exact JSON notes and preserved MIDI attacks, and 3,240 paired ledger events; it recomputes pitch/string/latency scores and the selected/identity validation recipes. Browser checks passed 24 views, 120 export links and 24 JSON downloads. A subsequent prose-only report refresh has identical interactive JavaScript and a separate visual check, recorded in `presentation-provenance.json`. The public repair replay also reproduces all 69 source-note intervals of the saved rock excerpt. These checks establish implementation consistency, separately from transcription accuracy.

The independent diagnostic commands are:

```powershell
.\.venv\Scripts\python.exe -m diagnosis9.head_audit
.\.venv\Scripts\python.exe -m diagnosis9.context_audit
.\.venv\Scripts\python.exe -m diagnosis9.oracles
.\.venv\Scripts\python.exe -m diagnosis9.oracles --hybrid
```

The head/context audits evaluate the frozen model on validation only. Oracle commands diagnose annotation-derived target heads and verified cached validation predictions; they perform no new acoustic inference or fitting. All diagnostic scores use the first ten seconds with twelve seconds of input and the existing note tolerances.

Replay another isolated-guitar file using either decoder:

```powershell
.\.venv\Scripts\python.exe -m diagnosis9 replay "path\to\isolated-guitar.wav" --mode strict --seconds 30 --packet-ms 20 --output outputs/diagnosis9/example --device auto
```

Use `--mode repair` to inspect the validation-selected EMA candidate. Both modes require a valid frozen selection; `--diagnostics-root` defaults to `outputs/diagnosis9`. The command exports note JSON, MIDI, timed string/fret CSV and event/timing records. It reads/resamples a local file before packet replay; it does not capture a microphone. Source timestamps and delivery timestamps remain separate, and EOF releases remain censored.

The next budgeted work should test causal frontend resolution, normalization and supervised label delay under matched controls, beginning with a tiny training-subset overfit sanity check. These are proposed new training experiments, not additional compute spent in stage 9. Retain the existing limited-compute cap and use new output roots; freeze validation choices before another independent recording test.

## Stage 10: acoustic sanity and target timing

Stage 10 completes a separate **500-update** acoustic experiment: 100 updates on one fixed four-crop training batch, followed by two fresh 200-update models. The sanity run has five reporting blocks; each matched model has five epochs of 160 sampled crops, batch size four. No new dependencies or data are required. Models and diagnostics live under `outputs/acoustic10`; all earlier checkpoints remain intact.

The sanity check passes on 2,197 active string/frame decisions across seven frets. Active-fret accuracy rises from 14.3% to **97.2%**, exact-state F1 from 0.095 to **0.910**, and state loss falls to 4.98% of its initial value. This is memorization of four fixed training crops with dropout/gain disabled. It establishes that this setup can learn those examples; it provides no generalization result and does not initialize either comparison model.

The matched models use identical fresh weights, crop/gain schedules, normalization, architecture, loss balancing and optimizer settings. One uses original targets; the other translates all four target heads by six frames (60 ms) on the full cached recording before cropping. Both share the same first-six-prediction-frame training mask. Delayed public note times subtract the declared delay once for attacks, retriggers and releases; availability and delivery clocks retain their actual values. The existing decoder thresholds/backdate are fixed, with no calibration.

Both fixed step-200 models are compared on twelve progression-2 **validation** excerpts. Head metrics cover the same original ten-second source span: immediate predictions 0:1000 versus delayed predictions 6:1006, against original targets, with the first six source frames ignored in both.

| Validation measurement | Immediate labels | 60 ms delayed labels |
| --- | ---: | ---: |
| Exact string/fret frame F1 | 0.0554 | 0.0604 |
| Note onset F1 | 0.0381 | 0.0526 |
| Onset+offset F1 | 0.0045 | 0.0248 |
| Correctly matched onset count | 35 | 77 |
| Matched onset delivery p50 / p95 | 96.7 / 144.3 ms | 119.0 / 191.7 ms |

The declared rule selects `delay60` as the development candidate by exact-state F1, then note-onset F1. The gain is small and absolute quality remains weak. It does **not** replace the existing causal reference or offline product default. No test audio or fresh recordings are evaluated; these same validation examples supply selection and the displayed scores. The two models have half the optimizer updates of the earlier 400-update causal run, so they are not an equal-budget replacement comparison with that model.

Recorded optimization phases total about 37.4 seconds, with peak allocated training GPU tensors of 288.3 MiB. Startup, source verification and validation replay are separate; allocated tensors are not total VRAM use. Warm replay processing/audio is about 0.296 for both models, including exact probability recording. Matched-event delays are conditional on detected notes, and one immediate-label match has a slightly negative signed delay within the 50 ms matching tolerance.

Open `outputs/acoustic10/review.html` for the paired audio, acoustic-head metrics, notes and source/delivery event ledger. `findings.md`, `sanity.json`, `selection.json`, `training-summary.json`, final model receipts and `validation/run.json` preserve the decision and budget. Saved probability files reproduce every exported source interval.

```powershell
.\.venv\Scripts\python.exe -m acoustic10 report
.\.venv\Scripts\python.exe -m acoustic10 audit
```

The initial command is `python -m acoustic10 run --device cuda` on a fresh output root. The completed root protects its training budget; `evaluate`, `report` and `audit` reuse saved weights without additional optimizer updates. Verification passes 225 tests, and the independent audit checks all 24 validation cases, 1,650 JSON notes/MIDI attacks, and 3,300 ledger events. Browser checks cover 24 paired views, 120 export links and 24 exact JSON downloads.

The next acoustic trial should examine past-only spectral resolution under a new matched budget. The current 2,048-point FFT has 7.8125 Hz spacing; interpolation to musical bins does not create finer resolution. Longer windows also change acoustic age and delivery delay, so timing must remain explicit. The passing tiny-batch check and small timing gain do not isolate spectral resolution, normalization, short-crop context or insufficient training as the sole cause.

## Stage 11: past-only spectral windows

Stage 11 compares **128 ms / 2,048-point** and **256 ms / 4,096-point** Hann analysis windows within **500 total new updates**, 250 per model over five epochs. Both retain 216 log-frequency bins, 16 kHz audio, a 10 ms hop and the same 931,145-parameter causal CNN-GRU. The previous training-only sanity evidence is verified and reused with zero new sanity updates. Existing source files and checkpoints are preserved.

The comparison holds both arms to the same 130 ms supervised delay, first-26-frame masks, crop/gain schedule, initialization, loss weights and fixed decoder. Each frontend gets normalization calculated from all 48 training recordings with the same original recipe. The short features and their normalization exactly reproduce the old cache; long features live separately under `data/spectral11`. Full-file and 20 ms packet feature extraction are byte-identical, with zero future lookahead.

Doubling the window changes FFT spacing from 7.8125 to 3.90625 Hz and increases the Hann peak's past age from 64 to 128 ms. It also changes temporal smearing and interaction with the shared target delay. This tests a **window-length tradeoff under fixed supervision**, rather than isolating frequency resolution or finding each frontend's best latency setting.

| Validation measurement | 128 ms window | 256 ms window |
| --- | ---: | ---: |
| Exact string/fret frame F1 | 0.0699 | 0.0607 |
| Note onset F1 | 0.0666 | 0.0286 |
| Auxiliary pitch frame F1 | 0.1215 | 0.1144 |
| Low-register pitch frame F1, MIDI 40–51 | 0.1156 | 0.0768 |
| Correctly matched onsets | 89 | 46 |

The declared exact-state-first rule selects `short128` as the development model. The longer window does not help under this limited budget and fixed timing contract; low absolute quality still prevents a product change. The twelve displayed progression-2 validation clips also supply selection, and no test recordings are inferred. These results cannot be compared as an equal-recipe improvement over stage 10, whose update count, target delay and masks differ.

Measured onset delivery p50/p95 is 185.8/268.8 ms for the short arm and 145.4/265.1 ms for the long arm, conditional on different sets of 89 and 46 matched attacks. The smaller long-arm median does not establish a general latency advantage. Both have zero future lookahead and shared source-clock compensation. Normalization includes startup frames; the 26-frame exclusions apply to training loss and source-aligned validation metrics. Low-band macro F1 includes zeros from three validation clips with no reference pitches in that band; pooled support is 4,991 pitch/frame positives.

Recorded training takes about 39.5 seconds across both arms; peak allocated GPU training tensors are 280.3/281.1 MiB, excluding other device allocations. Warm replay processing/audio is approximately 0.321/0.319, including exact probability recording and excluding offline file loading/resampling. Saved actual AdamW counters confirm 250 updates for every parameter state in each model.

Open `outputs/spectral11/review.html` for paired acoustic, note, audio and source/delivery comparisons. `findings.md`, `cache-audit.json`, `training-summary.json`, model receipts and `validation/run.json` preserve the controls and limitations. Model files are under `models/short128/final.pt` and `models/long256/final.pt`.

```powershell
.\.venv\Scripts\python.exe -m spectral11 report
.\.venv\Scripts\python.exe -m spectral11 audit
```

The initial command is `python -m spectral11 run --device cuda` on a fresh output root; the completed root protects its training budget. Saved weights support `evaluate`, `report` and `audit` without additional updates. Verification passes **279 tests**. Independent auditing reproduces all 24 validation cases, 2,238 exact source notes/MIDI attacks and 4,476 events. Browser QA checks 24 views, 120 exports and 24 exact JSON downloads.

The next separately budgeted acoustic variable is frame-local normalization, with the short window and the remaining training/timing settings held fixed. This experiment does not identify normalization, optimization or data variety as the sole cause of weak quality; fresh recordings remain necessary before a generalization or native-microphone claim.

## Stage 12: frame-local normalization groups

Stage 12 compares eight CNN normalization groups with one group, keeping the short window and the complete stage-11 training/timing recipe fixed. The one-group variant normalizes all CNN channels/frequency positions together within each frame. Both variants remain frame-local and batch-independent; neither uses future frames. Projection LayerNorm, feature normalization, model size and all other layers remain unchanged.

The exact stage-11 `short128` checkpoint supplies the eight-group control, saving a duplicate training run. The one-group model trains from the same fresh seed-42 tensors for **250 new updates** over five epochs, with identical 1,000 crop/gain descriptors. Both compared models have 250 lifetime training updates; the control's 250 updates are historical, and this stage uses only 250 new updates against its 500-update cap. No new sanity training or warmstart from the sanity model occurs.

All twelve current control probability sets are bit-identical to stage 11. Source notes, head decisions and note/string scores reproduce exactly, while processing/delivery clocks are measured again. Original weights, optimizer state and receipts remain untouched; `models/group8/reference.json` points to the historical binary. The treatment is under `models/group1/final.pt` and requires the `normalization12.model.CausalCNNGRU` wrapper, which reads `model.cnn_norm_groups` when loading it.

| Validation measurement | Eight groups | One group |
| --- | ---: | ---: |
| Exact string/fret frame F1 | 0.0699 | 0.0721 |
| Note onset F1 | 0.0666 | 0.0779 |
| String-aware note onset F1 | 0.0513 | 0.0632 |
| Auxiliary pitch frame F1 | 0.1215 | 0.1268 |
| Low-register pitch frame F1, MIDI 40–51 | 0.1156 | 0.1223 |

The declared rule selects `group1` for development, with gains of only 0.00213 state F1 and 0.01127 onset F1. These are one-seed, validation-only results from the same twelve selection clips, not independent-test evidence. Accuracy remains weak, so the existing offline default and prior causal references remain unchanged.

The 128 ms/2,048-point frontend, frozen training-only per-bin statistics, 130 ms target delay, 26-frame masks and fixed decoder are shared. Head metrics cover the same original source span; source timestamp compensation never advances measured delivery. Group count is the only model operation changed, and both variants retain all 931,145 parameter tensors and the same initialization RNG consumption. Reusing the historical control changes training chronology; fresh paired inference timings avoid reusing its old runtime measurements.

Treatment training takes about 24.3 seconds, with 280.1 MiB peak allocated GPU tensors. The control receives zero new training seconds; its original 20.6-second training phase belongs to stage 11. Current warm processing/audio is approximately 0.325/0.326. Matched onset delivery p50/p95 is 185.2/268.9 ms for 89 control matches and 191.3/252.6 ms for 91 treatment matches; these different conditional samples do not establish a speed/latency ranking.

Open `outputs/normalization12/review.html` and `findings.md` for the comparison. `input-audit.json`, `protocol.json`, `training-summary.json`, raw historical reference and final treatment receipt preserve reuse and update accounting.

```powershell
.\.venv\Scripts\python.exe -m normalization12 report
.\.venv\Scripts\python.exe -m normalization12 audit
```

The initial command is `python -m normalization12 run --device cuda` on a fresh root. The completed root protects its training budget; `evaluate`, `report` and `audit` reuse saved weights. All **334 tests** pass. Independent auditing reproduces 24 validation cases, 2,419 exact JSON notes/MIDI attacks and 4,838 ledger events, including the historical control. Browser verification covers all paired views and saved exports.

The next separately budgeted question is optimization dose: whether additional matched updates improve the existing variants. Preserve both parent checkpoints and compare a documented continuation recipe, keeping data, normalization, features and timing fixed. The small normalization gain does not explain the full acoustic gap; new recordings remain necessary for a generalization or native-microphone claim.

## Stage 13: additional optimization dose

Stage 13 preserves both 250-update parent models, restores their weights **and AdamW moments/counters**, and gives each 250 additional updates. The stage spends **500 new updates total**; each model reaches 500 lifetime updates after five additional sampled-crop epochs. The parents remain immutable references. No new sanity training, source data or dependencies are added.

The continuation holds the stored final learning rate at **0.0001**, without a scheduler restart. Historical random-generator and AMP-scaler states were absent, so both arms use a declared fresh Torch seed 43 and scaler initialization 1024; data retains seed 42 and uses deterministic epochs 6–10. This is a new continuation recipe, not a bit-exact reconstruction of uninterrupted training. Before the first update, each `resume-state.pt` records restored tensors and optimizer state for independent comparison with its parent. The final checkpoints now also save scaler and RNG states.

Features, per-bin statistics, grouping, labels, masks, decoder, loss, dropout and the shared new crop/gain schedule remain fixed. A matched 1,000-crop continuation adds both gradient updates and sampled exposure. Its effect cannot identify optimizer moments, learning rate or insufficient training as the sole cause of the acoustic gap.

| Validation measurement | Eight groups · 250 | Eight groups · 500 | One group · 250 | One group · 500 |
| --- | ---: | ---: | ---: | ---: |
| Exact string/fret frame F1 | 0.0699 | 0.0779 | 0.0721 | 0.0764 |
| Note onset F1 | 0.0666 | 0.1012 | 0.0779 | 0.0915 |
| Auxiliary pitch frame F1 | 0.1215 | 0.1674 | 0.1268 | 0.1561 |
| Low-register pitch frame F1 | 0.1156 | 0.0947 | 0.1223 | 0.1054 |

Continuation improves each arm's primary and note-onset scores, but the low-register diagnostic declines in both. The declared final-only rule selects `group8_500` from the two 500-update candidates; the 250-update parents are dose references, not eligible winners. This changes the within-budget development ranking from stage 12 and does not establish a universal normalization ranking. Accuracy remains weak, and no product default changes. All displayed scores and selection reuse the same twelve progression-2 validation clips; no test audio is inferred.

Both parent replays reproduce all 24 prior probability sets bit-for-bit, along with source notes and decisions; completion clocks are measured again. Current warm processing/audio ranges from 0.152 to 0.178 across four views. Delivery statistics use different matched notes and cannot establish a general speed/latency improvement. New training phases total about 35.2 seconds; allocated GPU tensor peaks are roughly 281 MiB, separate from other device allocations and inference/report preparation.

Open `outputs/optimization13/review.html` for all four models, same-family 250/500 comparisons, audio and source/delivery event ledgers. `findings.md`, `protocol.json`, `input-audit.json`, parent references, resume snapshots and training receipts preserve the lineage. New final models are under `models/group8/final.pt` and `models/group1/final.pt`; the parents stay at their original locations.

```powershell
.\.venv\Scripts\python.exe -m optimization13 report
.\.venv\Scripts\python.exe -m optimization13 audit
```

The initial command is `python -m optimization13 run --device cuda` on a fresh root. A completed root protects its new-update budget, and `evaluate`, `report` and `audit` reuse saved weights. **417 tests pass**. Independent auditing checks 48 validation cases, 5,351 exact source notes/MIDI attacks, 10,702 events, every CSV interval, parent/restored AdamW moments, lifetime counters and final-only selection. Browser verification covers 48 views, 240 export links and 48 exact JSON downloads.

The next useful check can use zero new training: verify the shared source-clock contract with delayed oracle heads, then measure a small validation-only decoder calibration on frozen models. Raw-head accuracy must stay visible alongside any decoding gain, and independent recordings remain necessary before a generalization or microphone-product claim.

## Stage 14: frozen decoder calibration

Stage 14 uses the two Stage 13 final models' cached validation probability heads. It adds **zero optimizer updates and zero neural inference calls**. Each model retains 500 historical lifetime updates. All 24 unchanged-decoder replays reproduce the saved source notes, packet events and scores exactly after removing only measured completion fields. No test audio is loaded or scored.

Before computing scores, the new protocol fixes 27 candidates per model: onset thresholds 0.55/0.40/0.70, activity thresholds 0.35/0.25/0.50, and extra onset backdates 60/0/30 ms. Per-model selection uses macro onset F1, then onset-plus-offset F1, with the original candidate winning exact ties. Attack/release confirmation, model weights, cached probabilities and the **130 ms source-delay correction** stay fixed.

| Validation measurement | Eight groups / original | Eight groups / selected | One group / original | One group / selected |
| --- | ---: | ---: | ---: | ---: |
| Note onset F1 | 0.1012 | 0.1254 | 0.0915 | 0.1157 |
| Note onset-plus-offset F1 | 0.0625 | 0.0769 | 0.0421 | 0.0602 |
| String-aware note onset F1 | 0.0743 | 0.0952 | 0.0659 | 0.0924 |
| Matched onsets | 133 | 173 | 119 | 159 |
| Predicted notes | 1,287 | 1,523 | 1,169 | 1,430 |

Both models select onset threshold **0.70**, activity threshold **0.25**, and **zero extra onset backdate**. Removing the extra backdate alone gives 0.1207/0.1106 onset F1 at the original thresholds; the other threshold changes provide a smaller additional gain. Each model gains 40 matched onsets, but also produces more false notes. Raw state/pitch-head metrics remain unchanged, absolute quality remains low, and the existing defaults stay unchanged. The same repeatedly used twelve validation recordings select and report the gain.

Annotation-only oracle heads verify source timing with and without a 13-frame label delay. After compensation, both versions give onset F1 **0.9612** without extra backdating and **0.3584** with the old 60 ms extra backdate. This confirms the known source correction and shows that the legacy extra backdate overcorrects ideal aligned heads. Oracle labels contain privileged information; their scores do not describe acoustic model accuracy. Leading inactivity, conflict masking and truncated/censored tails remain explicit.

The viewer shows source boundaries and **packet availability only**. Cached replay excludes neural processing, device and microphone delays, so it cannot establish end-to-end latency or inference speed. Its conditional timing samples also change with the matched-note set.

Open `outputs/decoder14/review.html` and `findings.md`. The protocol, full grid, oracle records and selection are saved alongside exact JSON/CSV notes, quantized MIDI and event ledgers. **443 tests pass**. The reproduction audit checks all 648 grid cases, 48 review cases, 48 oracle cases, 6,470 source notes and 12,940 review events. Browser verification checks every review view, 240 export links, 48 exact JSON downloads, 48 paired switches and all 12,940 displayed packet events.

```powershell
.\.venv\Scripts\python.exe -m decoder14 report
.\.venv\Scripts\python.exe -m decoder14 audit
```

The initial command is `python -m decoder14 run` on a fresh output root. Completed roots reject repeat runs. The next acoustic experiment should use the remaining error types to choose one loss, data or representation change within a separate five-epoch / 500-new-update budget. These decoder gains do not establish a sole cause of the acoustic gap or performance on new recordings.

## Stage 15: residual acoustic errors and the next loss hypothesis

Stage 15 keeps both 500-update models and their Stage 14 selected decoder frozen. It spends **zero new optimizer updates and zero neural/test inference calls**. It verifies original targets from annotations, classifies remaining errors, and tests four fixed state/onset head combinations on the same twelve validation recordings. Acoustic accuracy is unchanged.

| Input to the fixed decoder | Eight-group onset F1 | One-group onset F1 |
| --- | ---: | ---: |
| Frozen predictions | 0.1254 | 0.1157 |
| Ideal state, predicted onset | 0.8801 | 0.8945 |
| Predicted state, ideal onset | 0.1284 | 0.1166 |
| Ideal state and onset | 0.9654 | 0.9654 |

The state intervention replaces both occupancy and fret identity; it does not isolate those components. All rows use onset threshold 0.70, activity threshold 0.25, zero extra backdate and fixed 130 ms source compensation. The ideal scores contain privileged labels and describe decoder diagnostics, not new model performance. Their thresholds differ from Stage 14's oracle check.

On 22,219 reference-active string-frames per model, selected activity decisions give correct frets on 14.5%/14.3%, predict inactivity on 20.3%/26.1%, and predict other wrong frets on 65.2%/59.6%. Same-string octave errors account for zero of these frames; separate note-level octave associations are still possible. Median state runs are 8/7 frames versus 25 in the labels. These counts indicate weak and unstable state evidence without establishing its training cause.

Pitch unions derived from the state decisions score 0.1403/0.1353 frame F1, versus 0.1674/0.1561 for the separate pitch head. This is a threshold-dependent comparison at the fixed declared settings. Together with the head interventions, it passes the declared rule for proposing **direct state-to-pitch auxiliary supervision**.

The new `diagnosis15/loss.py` maps the existing string/fret probabilities through standard tuning to 49 MIDI occupancies, collapses unisons using an independent-string union, and adds 0.5-weight pitch BCE to the unchanged legacy total. Crop/leading masks and any-string-conflict exclusion apply to this new term. It adds no model parameters and does not resolve every equivalent fingering. Algebra, masks, gradients and zero-coefficient identity are tested. **No acoustic model has been trained with this loss yet**, so no benefit is claimed.

The saved next experiment proposes two 250-update continuations from the same frozen eight-group parent: unchanged-loss control versus the new term. That would spend 500 new updates over five new sampled-crop epochs per arm, reaching 750 lifetime updates each. Details are in `outputs/diagnosis15/next-experiment.json` and `next-experiment-details.md`.

The interactive `outputs/diagnosis15/review.html` provides 24 recording/model views with one-to-one note associations, category filters, per-note acoustic evidence, seeking and exact exports. Nearby wrong-pitch associations are not proof of a particular acoustic cause. Repetition and multiple-overlap tags are separate from the exclusive reference partition.

**469 tests pass.** The reproduction audit checks 24 calibrated identities, 96 head interventions, 1,506 reference rows, 2,953 scored prediction rows, 20,654 intervention events and 3,538 full-context exported notes/MIDI attacks. Browser verification covers all 24 views, 144 category filters, 120 export links, 24 exact diagnosis downloads, seeking and playback-position-preserving switches.

```powershell
.\.venv\Scripts\python.exe -m diagnosis15 report
.\.venv\Scripts\python.exe -m diagnosis15 audit
```

Initial diagnosis uses `python -m diagnosis15 run` on a fresh root. Completed roots reject repeat runs. No defaults are changed, and this reused development set still supplies no independent generalization or microphone-product result.

## Stage 16: matched direct state-to-pitch supervision

Stage 16 executes the Stage 15 proposal using **500 new optimizer updates total**, split into two 250-update continuations. Both arms restore the same immutable 500-update eight-group parent, including model tensors, ordered AdamW moments/counters, AMP scaler and CPU/CUDA RNG states. Each reaches **750 lifetime updates** after five new sampled-crop epoch blocks. The shared parent's 500 historical updates are counted once; no new sanity optimization occurs.

Original targets/features, normalization, gain/crop descriptors from data epochs 11–15, inherited optimizer settings and constant 0.0001 learning rate are shared. The only treatment is the tested 0.5-weight state-derived pitch loss. Its probability-space BCE runs in float32 outside CUDA autocast. The original model and all legacy producer files remain unchanged.

| Validation measurement | Parent / 500 | Original loss / 750 | Added pitch loss / 750 |
| --- | ---: | ---: | ---: |
| Exact string/fret frame F1 | 0.0779 | 0.0866 | 0.0875 |
| Note-onset F1 | 0.1254 | 0.1634 | 0.1598 |
| Note onset-plus-offset F1 | 0.0769 | 0.0835 | 0.0844 |
| String-aware note-onset F1 | 0.0952 | 0.1138 | 0.1114 |
| State-derived pitch frame F1 | 0.1403 | 0.1796 | 0.1807 |
| Auxiliary pitch frame F1 | 0.1674 | 0.1771 | 0.1785 |
| MIDI 40–51 pitch frame F1 | 0.0947 | 0.0966 | 0.0982 |

The result is mixed: treatment raises primary state F1 by only **0.00091** versus the matched control and lowers note-onset F1 by **0.00360**. The declared state-first rule selects `state_pitch750` for this development comparison, while `control750` gives the higher note-onset and string-note scores. Both improve over the parent after additional training. This one-parent, one-coefficient comparison does not establish a broad loss improvement; defaults stay unchanged.

All twelve parent probability sets are bit-identical to the archived recordings, with exact source notes, calibrated events, head decisions and scores. All 34 actual AdamW counters reach 750 in each child. Both pre-update snapshots reproduce parent model/optimizer/scaler/RNG state, and final paired RNG streams remain identical. No validation gradients, early stopping, decoder search, scheduler restart, skipped-step replacement or test inference occurs.

New training phases take about 20.0/16.6 seconds, with roughly 281 MiB peak allocated GPU tensors. These totals exclude setup, restoration and validation. Fresh warm processing/audio is 0.235/0.246/0.248 for parent/control/treatment. Their inference graph is identical, and these timing differences are not a causal speed improvement. Matched-onset completion p50/p95 is 145.9/210.7 ms, 147.8/207.6 ms and 147.7/207.3 ms on different matched subsets. No microphone or sound-card capture delay is measured.

Open `outputs/supervision16/review.html` and `findings.md`. It contains all three models, paired comparisons, audio, exact exports and separate source/packet/virtual-completion clocks. The parent stays at its original path; new final models are `models/control/final.pt` and `models/state_pitch/final.pt` under the new output root.

**498 tests pass.** The reproduction audit checks 36 validation cases, 5,382 exact notes/MIDI attacks, 10,764 events, matched epoch exposure, restoration snapshots and final-only selection. Browser checks cover 36 views, 180 export links, 36 exact JSON downloads, 36 paired switches and all 10,764 three-clock event markers.

```powershell
.\.venv\Scripts\python.exe -m supervision16 report
.\.venv\Scripts\python.exe -m supervision16 audit
```

Initial training uses `python -m supervision16 run --device cuda` on a fresh root. Completed roots reject repeat training; `evaluate`, `report` and `audit` reuse saved models. The next separate acoustic question is a matched short-only versus combined short/long past-only representation, retaining a 500-new-update cap and capacity-matched controls. The earlier negative long-only result does not answer whether combining both scales helps. Independent recordings remain necessary before a generalization or product-quality claim.

## Stage 17: matched short/long spectral fusion

Two fresh two-channel CNN-GRUs compare **short + short (128/128 ms)** with **short + long (128/256 ms)**. Both start with identical tensors and random states, have 931,361 parameters and use the same 1,000 crop/gain descriptors. Each receives five sampled-crop epochs and 250 actual updates: **500 new updates total**, with 250 lifetime updates per arm and no warmstart. The legacy loss and fixed Stage14 decoder are retained.

| Development measurement | Short + short | Short + long |
| --- | ---: | ---: |
| Exact string/fret frame F1 | 0.0709 | 0.0684 |
| Note-onset F1 | 0.1098 | 0.1260 |
| Note onset-plus-offset F1 | 0.0412 | 0.0565 |
| String-aware note-onset F1 | 0.0829 | 0.0947 |
| Decoded pitch frame F1 | 0.1160 | 0.1281 |
| Separate pitch head MIDI 40–51 F1 | 0.1154 | 0.1005 |

Combined features help decoded notes, with more matched attacks and more extra notes, but lower the primary exact-state score. The declared state-first rule selects `short_short`; defaults stay unchanged. These are reused development recordings with a previously tuned decoder, one seed and short training. They do not establish independent generalization, and the fresh 250-update models are not a matched comparison with Stage16's 750-update continuations.

Features remain past-only log-frequency STFTs on a shared 10 ms right-edge grid. Both emit a zero-left-padded first frame after 10 ms, with zero future lookahead. The long window changes spectral detail and past-window age together. Training-only normalization exactly reproduces archived statistics. The control computes its duplicated FFT once; the treatment computes two FFTs, although neural model capacity is identical.

All 34 actual AdamW counters reach 250 per arm. Initial model/optimizer/scaler/RNG snapshots are identical, and paired final RNG streams agree. **517 tests pass.** The reproduction audit verifies 24 cases, 24 full/packet/cache feature comparisons, 3,028 exact notes and 6,056 events. Browser checks cover 24 views, 120 export links, 24 exact JSON downloads and all event clocks.

Training phases take about 16.5/16.2 seconds, with roughly 281/282 MiB allocated GPU tensors. Warm processing/audio is 0.242/0.254, including all frontend planes. Processing/audio excludes EOF flush cost, which is included in the completion clock. These measurements exclude hardware capture and do not establish a stable speed advantage.

Open `outputs/multires17/review.html` and `findings.md` for audio, paired playback, exact exports, scores and timing. Final models are `outputs/multires17/models/short_short/final.pt` and `models/short_long/final.pt`, loaded with `multires17.model.CausalCNNGRU` and the per-checkpoint two-plane frontend/normalization.

```powershell
.\.venv\Scripts\python.exe -m multires17 report
.\.venv\Scripts\python.exe -m multires17 audit
```

Initial training uses `python -m multires17 run --device cuda` on a fresh root; completed roots reject repeat training. The proposed next stage is a zero-training state/onset head-swap diagnosis using these saved probability recordings and the same decoder. It can investigate the note-versus-fret tradeoff before spending more compute; it cannot establish a pure training cause or independently validated hybrid improvement.

## Stage 18: frozen state/onset diagnosis

Stage18 spends **zero training updates and zero neural inference calls**. It reuses Stage17 probabilities, reproduces both original models' source decisions on all twelve clips, then crosses their state and onset heads under the same decoder. No hybrid or threshold is selected.

| State source | Onset source | Mean note-onset F1 | Matched attacks | Official extra notes |
| --- | --- | ---: | ---: | ---: |
| Short + short | Short + short | 0.1098 | 127 | 1,037 |
| Short + long | Short + long | 0.1260 | 153 | 1,194 |
| Short + short | Short + long | 0.1107 | 129 | 1,053 |
| Short + long | Short + short | 0.1254 | 152 | 1,192 |

State swaps change macro onset F1 by about **+0.0152 to +0.0155**; onset-only swaps change it by **+0.0006 to +0.0009**. These frozen comparisons associate most of the note gain with state output, with calibration mismatch limiting causal attribution. This decoder opens new notes/fret changes from four stable state frames and uses onset spikes only for same-fret retriggers, so small onset contrasts do not rank onset prediction under other decoders. The originals match 112 reference notes in common; combined features recover41 and lose15, with585 missed by both.

Argmax remains the Stage17 selection metric (0.0709/0.0684). At the fixed decoder activity rule, mean state F1 is0.0936/0.0942 but pooled state F1 is0.1319/0.1272: aggregation and decision policy matter. Decoder-policy activity recall rises64.73%→66.40%, while fret accuracy when both reference and prediction are active falls20.98%→20.02%. Best-active-fret accuracy over all reference-active frames is20.25%/19.01%. This is an occupancy/fret tradeoff, not a uniform improvement or a new selection result.

**536 tests pass.** The audit reproduces48 cases,24 frame diagnoses,6,072 exact notes/MIDI attacks,12,144 events and3,012 reference-evidence rows. JSON/CSV/MIDI exports and source associations are checked. The browser review exposes recording, state and onset selectors, audio-preserving head switches, note seeking, evidence filters and exact downloads. Its timeline shows source and packet availability only; prior measured completion clocks are deliberately removed. No hybrid processing speed or microphone latency is measured.

Open `outputs/headswap18/review.html` and `findings.md`. The existing default and Stage17 selection remain unchanged.

```powershell
.\.venv\Scripts\python.exe -m headswap18 report
.\.venv\Scripts\python.exe -m headswap18 audit
```

Initial diagnosis uses `python -m headswap18 run` on a fresh root; completed roots reject repeat runs. The next proposed experiment adds a0.5-weight active-only conditional-fret cross-entropy term to the legacy loss. Both arms would restore the same short/short250-update parent, add250 updates across five new sampled epochs and reach500 lifetime updates each. Future new budget500 total; the proposal has not trained. See `outputs/headswap18/next-experiment.json` for masks, restoration, fixed decoding and final-only comparison details.

## Stage 19: active-only conditional-fret supervision

Stage19 tests the Stage18 proposal with **500 new optimizer updates total**. Both arms restore the same short/short250-update parent, including model, ordered AdamW moments/options/counters, AMP scaler and CPU/CUDA RNG. Each adds250 updates over five new sampled epochs6–10 at constant0.0001 and reaches500 lifetime updates. The ended parent scheduler is archived without restart or new scheduler steps. The shared parent's250 historical updates are counted once.

| Development measurement | Parent / 250 | Original loss / 500 | Extra fret loss / 500 |
| --- | ---: | ---: | ---: |
| Argmax exact string/fret frame F1 | 0.0709 | 0.0828 | 0.0749 |
| Note-onset F1 | 0.1098 | 0.1148 | 0.1136 |
| Note onset-plus-offset F1 | 0.0412 | 0.0718 | 0.0698 |
| String-aware note-onset F1 | 0.0829 | 0.0920 | 0.0895 |
| Best active fret on reference-active frames | 20.25% | 18.66% | 19.55% |
| Correct fret when reference and prediction are active | 20.98% | 18.79% | 19.44% |

The extra0.5-weight CE term improves conditional fret choice versus its matched control but lowers primary state F1 by0.00789 and note-onset F1 by0.00122. The declared rule selects `control500`; the added loss is not adopted and defaults stay unchanged. The treatment improves note-onset F1 on four clips and lowers it on eight. Pooled decoder-policy state F1 rises slightly while mean clip F1 falls; different aggregation and supervision denominators remain explicit.

The term uses the existing25 active logits only, masks inactive/conflict targets and crop/leading frames, averages over valid active string frames, returns differentiable zero for empty support and runs in float32 outside autocast. It adds no parameters and has no direct auxiliary gradient on inactive logits. Shared weights and active-logit changes can still affect occupancy. Both arms keep the same931,361-parameter two-plane network, features, normalization, labels, gain/crop schedule and fixed decoder. No new sanity training, validation gradients, early stopping, decoder search, test inference or skipped-update replacement occurs.

All twelve parent probability sets are bit-identical to Stage17, with exact source notes, events, scores and Stage18 frame diagnoses. All34 real optimizer counters reach500 per child; pre-update restoration snapshots match the parent and final paired RNG streams agree. **565 tests pass**, including29 new loss/restoration contracts. The audit checks36 cases,5,134 exact notes/MIDI attacks,10,268 events and independently reconstructed active supervision. Browser checks cover36 views,180 export links,36 exact JSON downloads and playback-preserving model switches.

Training phases take about24.5/21.8 seconds with roughly283 MiB peak allocated GPU tensors. Warm processing/audio is0.329/0.334/0.333 for parent/control/treatment. Identical inference architecture prevents attributing timing variation to a speed improvement from the loss. EOF flush cost is included in completion clocks and excluded from processing/audio. No hardware capture latency is measured.

Open `outputs/fret19/review.html` and `findings.md`. New final models are `outputs/fret19/models/control/final.pt` and `models/conditional_fret/final.pt`, loaded with the unchanged `multires17.model.CausalCNNGRU` architecture and per-checkpoint two-plane normalization.

```powershell
.\.venv\Scripts\python.exe -m fret19 report
.\.venv\Scripts\python.exe -m fret19 audit
```

Initial training uses `python -m fret19 run --device cuda` on a fresh root. Completed roots reject repeat training; the original CUDA scaler/RNG state is required for this continuation. The next proposed stage is a zero-training audit of per-string/fret data coverage and actual supervised crop exposure, using saved probabilities to relate rare/unseen labels to errors. It has not run. Independent recordings remain necessary for a generalization claim.

## Stage 20: supervised-label coverage diagnosis

Stage 20 follows the recorded Stage 19 proposal with zero new training, neural inference or processing-clock measurements. The new `coverage20` package reads frozen target caches, crop schedules, probabilities and official saved note decisions. It never loads feature arrays, waveforms or a neural checkpoint.

The 48 training recordings contain 174,071 valid active source string-frames. The parent's sampled epochs1-5 present 242,592 active string-frames, covering 127,390 unique source-frame/string identities. Extending the shared history through epochs1-10 gives 476,406 presentations and 154,681 unique identities. Each history presents 88 of the 150 possible string/fret positions; all 64 positions used by these validation clips are among them.

For `control500` argmax, the 1000+ presentation bin contains 21,386 of 22,219 valid active GT string-frames and 19,049 errors: 95.81% of all active errors, with an 89.07% within-bin error rate. The remaining 833 active frames in lower-exposure bins all fail exact-state prediction. No validation GT frame or scored reference onset belongs to an unseen position. This does not estimate unseen-position performance or establish that repeated exposure is sufficient; repeated frames can still represent limited independent diversity.

Counts preserve full-recording target delay13, delayed-grid leading26 and crop-edge26 masks. Original source identity is `(recording, delayed frame - 13, string)`. Available frames, repeated presentations and unique identities are separate. Each child uses the same combined history; unique coverage is a union rather than the sum of separate epoch-block unique counts.

Frame policies use GT source frames0:1000 against saved predictions13:1013, with the first26 source frames masked. False activity is grouped by predicted position, with all valid inactive GT frames as denominator. Official pitch/onset note matches use all scored onsets in [0,10), including early onsets before the frame mask; string agreement within those official matches is an additional flag. Empty bins retain null rates. Raw argmax and fixed decoder-activity policies remain distinct.

The independent audit reconstructs delayed target arrays and masked crops separately from the primary range-counting algorithm. It checks both histories, original frame totals, all official note matches and exact values in 2,259 note CSV rows, 300 coverage rows and 900 class-error rows. **606 tests pass**, with one existing dependency warning. Browser QA checks six model/policy views, 420 filter states, 54 export links, six exact JSON downloads and 36 source-case links, including URL-quoted sharp signs. Desktop and mobile screenshots were inspected.

Open the local [coverage review](../outputs/coverage20/review.html), [findings](../outputs/coverage20/findings.md) or [verification](../outputs/coverage20/verification.json). The tracked [snapshot](../experiments/stage20.snapshot.json) preserves 658 artifact/code/input bindings. The offline default and Stage 19 research selection persist; no new selection occurred.

```powershell
.\.venv\Scripts\python.exe -m coverage20 audit
.\.venv\Scripts\python.exe -m coverage20 report
.\.venv\Scripts\python.exe scripts/verify_coverage20.py
.\.venv\Scripts\python.exe scripts/check_project.py --integrity-only
```

`python -m coverage20 run` performs the original zero-inference counting pass on a workspace with the frozen predecessors restored and no completed Stage 20 verification. Verified roots reject repeat generation. The proposed Stage 21 compares saved fret choices with training-frequency priors under GT-active conditioning; it has not run and cannot establish a deployable decoder or independent generalization result.

## Stage 21: training priors and conditional-fret confusions

The `prior21` package compares saved best-active fret decisions with fixed training-frequency priors, without training or neural inference. All comparisons use22,219 GT-active string-frames across64 supported positions from the same twelve development recordings. Source0:1000 aligns with saved predictions13:1013; the first26 source frames and per-string conflicts are excluded. Best-active argmax is over frets0-24 even when inactive25 wins the full-state decision. This GT-activity support is an oracle diagnostic, not end-to-end transcription accuracy.

The declared primary prior picks each string's most frequently presented training fret, resolving ties toward the lowest fret. All declared secondary views remain visible: unique-count and available-count majority priors, analytic distribution priors for all three sources, and uniform25. Priors use only each model's own frozen Stage20 training history. No baseline source, threshold or model was selected from the diagnosis.

For `control500`, frame-weighted best-active fret accuracy is18.66% versus16.62% for the primary prior, a2.04 percentage-point gain. Equal-supported-position accuracy is12.89% versus9.38%. The unique-count and available-count majority priors each reach18.03%, and remain secondary. The control wins on seven clips, ties on two and loses on three against the primary prior. Its mean paired clip gain is2.51 percentage points, which has different weighting from the pooled frame gain.

The model chooses its training-modal fret on60.64% of GT-active frames. Its own within-string output/GT marginal independence expectation is15.13%; its observed accuracy exceeds this by3.53 percentage points. Concentrated predictions coexist with some alignment beyond the marginals. This does not prove that class imbalance causes the errors or that balancing will improve state/note scores. Frame-weighted, equal-position and equal-string denominators remain separate; unsupported classes retain null recall.

Full25x25 within-string confusions retain every predicted fret. Distribution priors contain analytic fractional expected mass, not sampled decisions. The independent audit reconstructs model choices using scalar comparisons and prior expectations using explicit loops. It verifies exact CSV values in900 prior rows,3,600 class rows,144 string rows,288 case rows and38,400 supported-class confusion rows. **633 tests pass**, including27 new Stage21 tests, with one existing dependency warning. Browser QA covers24 model/predictor views,576 control states,264 export links,three exact confusion-JSON downloads and36 source-case links. Desktop and mobile screenshots were inspected.

Open the [prior/confusion review](../outputs/prior21/review.html), [findings](../outputs/prior21/findings.md) or [verification](../outputs/prior21/verification.json). The [tracked snapshot](../experiments/stage21.snapshot.json) preserves693 artifact/code/input bindings. The Stage19 research selection persists; conditional diagnostic scores do not select a replacement.

```powershell
.\.venv\Scripts\python.exe -m prior21 audit
.\.venv\Scripts\python.exe -m prior21 report
.\.venv\Scripts\python.exe scripts/verify_prior21.py
.\.venv\Scripts\python.exe scripts/check_project.py --integrity-only
```

`python -m prior21 run` creates the original zero-inference diagnosis when its frozen predecessors are restored and no completed Stage21 verification exists. Verified roots reject repeat generation. Stage22 is proposed, not trained: original loss, uniform active-only fret CE and bounded class-weighted CE receive the same three crop epochs from the same fully restored control500 state. The cap is450 new updates total,150 per arm. Unique-supervised training counts define fixed inverse-square-root weights clipped at4; the weighted active CE divides by the sum of active weights. No extra sanity training, calibration search, independent-test inference or production promotion is proposed.

## Sources

- [Google/Magenta MT3](https://github.com/magenta/mt3) and [paper](https://arxiv.org/html/2111.03017).
- [kunato MT3-PyTorch port](https://github.com/kunato/mt3-pytorch).
- [Pinned inference adapter](https://github.com/openmirlab/mt3-infer/tree/d8cbd287470291dee335f84315c5a9b9ff6e6c8a).
- [Spotify Basic Pitch](https://github.com/spotify/basic-pitch).
- [GuitarSet authors](https://guitarset.weebly.com/) and [dataset record](https://zenodo.org/records/3371780).
- [mir_eval transcription metrics](https://craffel.github.io/mir_eval/#module-mir_eval.transcription).
- [Electric-guitar robustness study and authors](https://robust-guitar-tabs.github.io/) and [EGSet12](https://zenodo.org/records/11406378).
- [Beat This official model/code](https://github.com/CPJKU/beat_this) and [its GuitarSet training annotations](https://github.com/CPJKU/beat_this_annotations/tree/main/guitarset).
- [MusicXML 4.0 specification](https://www.w3.org/2021/06/musicxml40/).
- [Official Demucs v4.0.1](https://github.com/facebookresearch/demucs/tree/v4.0.1) and [six-source release notes](https://github.com/facebookresearch/demucs/blob/main/docs/release.md).

The original papers and the longer design rationale remain in `Papers/` and `research-plan.md`.
