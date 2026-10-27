# Experiment index

Completed through **Stage 19**. See [current status](../docs/STATUS.md) for the present default and latest result. The [detailed guide](../docs/experiment-guide.md) preserves stage-specific instructions; [index.json](index.json) provides the paths and recorded verification metadata in machine-readable form.

Review and report links below refer to local, Git-ignored outputs. They work in the recorded workspace after those assets are restored; a source-only checkout does not contain them. A stage's result applies to its stated data, budget and timing scope, not a shared cross-stage leaderboard.

## Completed stages

| Stage | Question and recorded outcome | Implementation | Local evidence |
| --- | --- | --- | --- |
| 1. Pretrained baselines | MT3 adapter and Basic Pitch inference; frontend differences and overlap limits recorded. | [Code](../baseline/) · [Config](../configs/stage1.json) | [Review](../outputs/stage1/review.html) · [Report](../outputs/stage1/report.md) |
| 2. Offline CNN-GRU | Small custom guitar model; the completed offline workflow remains the default. | [Code](../cnn_gru/) · [Config](../configs/cnn_gru.json) | [Review](../outputs/cnn_gru/evaluation/review.html) · [Report](../outputs/cnn_gru/report.md) |
| 3. CNN-transformer comparison | Controlled temporal-model comparison with recorded offline budget and timing controls. | [Code](../cnn_transformer/) · [Config](../configs/cnn_transformer.json) | [Review](../outputs/cnn_transformer/comparison/review.html) · [Report](../outputs/cnn_transformer/report.md) |
| 4. Tone/effect robustness | Synthetic-effect augmentation trial; original clean-trained model retained. | [Code](../robustness/) · [Config](../configs/robustness.json) | [Review](../outputs/robustness/evaluation/review.html) · [Report](../outputs/robustness/report.md) |
| 5. Constrained fingering | Feasible string/fret assignments preserve pitch/timing; acoustic errors remain. | [Code](../fingering/) · [Config](../configs/fingering.json) | [Review](../outputs/fingering/evaluation/review.html) · [Report](../outputs/fingering/report.md) |
| 6. Rhythm and notation | Beat/meter/chord drafts, editable timed tabs and MusicXML demonstrations. | [Code](../rhythm/) · [Config](../configs/rhythm.json) | [Review](../outputs/rhythm/review.html) · [Report](../outputs/rhythm/report.md) |
| 7. Controlled mixtures | Frozen separation helps controlled mixtures; real-song validation remains separate. | [Code](../mixtures/) · [Config](../configs/mixtures.json) | [Review](../outputs/mixtures/review.html) · [Report](../outputs/mixtures/report.md) |
| 8. Causal streaming | Causal frontend/model, carried state and incremental file-PCM events; no hardware-capture measurement. | [Code](../streaming/) · [Config](../configs/streaming.json) | [Review](../outputs/streaming/review.html) · [Report](../outputs/streaming/report.md) |
| 9. Frozen streaming diagnosis | State/fret bottleneck and experimental decoder repair; default unchanged. | [Code](../diagnosis9/) · [Config](../configs/diagnosis9.json) | [Review](../outputs/diagnosis9/review.html) · [Report](../outputs/diagnosis9/report.md) |
| 10. Acoustic sanity and timing | Tiny-batch learning gate and a small delayed-target development gain. | [Code](../acoustic10/) · [Config](../configs/acoustic10.json) | [Review](../outputs/acoustic10/review.html) · [Report](../outputs/acoustic10/report.md) |
| 11. Spectral windows | Short 128 ms window wins the matched short-only versus long-only test. | [Code](../spectral11/) · [Config](../configs/spectral11.json) | [Review](../outputs/spectral11/review.html) · [Report](../outputs/spectral11/report.md) |
| 12. Frame-local normalization | One-group normalization gives a small development gain; no promotion. | [Code](../normalization12/) · [Config](../configs/normalization12.json) | [Review](../outputs/normalization12/review.html) · [Report](../outputs/normalization12/report.md) |
| 13. Optimization dose | Matched continuation improves note scores; `group8_500` selected within the trial. | [Code](../optimization13/) · [Config](../configs/optimization13.json) | [Review](../outputs/optimization13/review.html) · [Report](../outputs/optimization13/report.md) |
| 14. Frozen decoder calibration | Verified 130 ms source correction; zero additional onset backdate selected. | [Code](../decoder14/) · [Config](../configs/decoder14.json) | [Review](../outputs/decoder14/review.html) · [Report](../outputs/decoder14/report.md) |
| 15. Residual-error diagnosis | State evidence motivates a supervision hypothesis; zero new training. | [Code](../diagnosis15/) · [Config](../configs/diagnosis15.json) | [Review](../outputs/diagnosis15/review.html) · [Report](../outputs/diagnosis15/report.md) |
| 16. Direct state-pitch loss | Tiny primary state-F1 gain with a note-onset decline; mixed development result. | [Code](../supervision16/) · [Config](../configs/supervision16.json) | [Review](../outputs/supervision16/review.html) · [Report](../outputs/supervision16/report.md) |
| 17. Combined spectral scales | Short + long helps decoded notes but lowers primary state F1; short + short retained. | [Code](../multires17/) · [Config](../configs/multires17.json) | [Review](../outputs/multires17/review.html) · [Report](../outputs/multires17/report.md) |
| 18. Frozen head crossing | Activity/fret tradeoff diagnosed; no hybrid selected and no new training/inference. | [Code](../headswap18/) · [Config](../configs/headswap18.json) | [Review](../outputs/headswap18/review.html) · [Report](../outputs/headswap18/report.md) |
| 19. Conditional-fret loss | Targeted fret-choice gain but lower primary frame/note scores; `control500` selected. | [Code](../fret19/) · [Config](../configs/fret19.json) | [Review](../outputs/fret19/review.html) · [Report](../outputs/fret19/report.md) |

## Proposed next stage

**Stage 20: coverage diagnosis — not executed.** Compare available `(string, fret)` labels with actual supervised crop exposure, then relate rare/unseen coverage to saved validation errors. The proposal spends zero training updates and zero neural inference. See the local [proposal](../outputs/fret19/next-experiment.json) and [current-status explanation](../docs/STATUS.md#next-research-step).

## Verification and backups

[stage19.snapshot.json](stage19.snapshot.json) is a tracked, byte-identical copy of the completed Stage 19 verification record. Its artifact paths are relative to `outputs/fret19/`; code paths are relative to the project root. It records 304 artifact/code bindings, checkpoint hashes, the saved test result and the completed audit/viewer coverage. The original [verification record](../outputs/fret19/verification.json) remains in place.

The registry's verification paths point to each stage's existing records where present. A missing record is represented by `null`, not an implied failure or an invented successful audit. The first eight stages use their original reports and run records; this index does not claim that every historical pipeline has been rerun during housekeeping.

`recorded_new_optimizer_updates: null` means the budget was not extracted from a standardized verification record; it does not mean zero training. Use the stage's report for its original budget and accounting.

Git tracks source, configuration, tests, documentation, small reference fixtures and compact records. It does not back up ignored datasets, model weights, outputs or local runtimes. Preserve those separately. Checkpoint identity in [index.json](index.json) helps identify the saved default; it does not supply the checkpoint itself.

Historical stage packages and output paths stay where they are because imports and recorded hashes depend on them. New experiments need a separate package/output root. Routine documentation cleanup should not reformat archived producer modules or regenerate historical evidence.
