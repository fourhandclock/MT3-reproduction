"""Recompute held-out metrics/exports from saved heads without neural inference."""

from pathlib import Path
import numpy as np
import soundfile as sf
from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate
from cnn_gru.inference import tablature_metrics
from decoder14.audit import verify_exports
from decoder14.core import replay, without_compute
from headswap18.analysis import state_diagnosis
from headswap18.audit import verify_frame_partition
from spectral11.workflow import aligned_heads
from streaming.evaluate import save_json
from streaming.latency import measure
from balance22.evaluation import frame_config, fret_confusion, summarize
from .data import MODES, load_test_inputs, read, source_hashes, validate_config
from .workflow import contrasts, locked_selection


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(config, output):
    validate_config(config)
    output = Path(output)
    protocol = read(output / 'protocol.json')
    require(protocol['source_sha256'] == source_hashes() and protocol['config'] == config, 'Locked producers/config differ')
    require(protocol['new_optimizer_updates'] == protocol['optimizer_instantiations'] == protocol['normalization_updates'] == 0
            and not protocol['model_selection_allowed'] and not protocol['calibration_search_allowed']
            and not protocol['production_promotion'] and protocol['lock_written_before_test_input_reads']
            and not protocol['test_audio_or_annotations_read_before_lock'], 'Locked zero-training/test-selection scope differs')
    base, inputs, provenance, bindings = load_test_inputs(config, output)
    require(read(output / 'input-bindings.json') == bindings and read(output / 'test-inputs.json') == provenance, 'Test input provenance changed')
    for name, digest in bindings.items():
        require(sha256(ROOT / name) == digest, 'An immutable input changed: ' + name)
    run = read(output / 'test/run.json')
    require(run['source_sha256'] == source_hashes() and run['scoring_config'] == protocol['scoring_config']
            and run['new_optimizer_updates'] == run['optimizer_instantiations'] == run['normalization_updates'] == 0
            and run['test_scored_replays'] == 48 and run['test_audio_warmups'] == 4
            and run['locked_selection'] == locked_selection(config) and not run['production_promoted'], 'Test run scope changed')
    require((output / 'selection.json').read_bytes() == (ROOT / 'outputs/balance22/selection.json').read_bytes()
            and sha256(output / 'selection.json') == protocol['selection_sha256']
            and run['selection'] == read(output / 'selection.json'), 'Development selection was reselected or altered')
    identities = read(output / 'model-integrity.json')
    for mode in MODES:
        identity = identities[mode]
        require(identity['sha256_before'] == identity['sha256_after'] == config['models'][mode]['sha256']
                == sha256(ROOT / config['models'][mode]['checkpoint'])
                and identity['step'] == (500 if mode == 'parent500' else 650)
                and not identity['normalization_recomputed'], 'Checkpoint contents or lifetime metadata changed')
    records = [read(path) for path in sorted((output / 'test/results').glob('*/*-metrics.json'))]
    lookup = {(row['id'], row['mode']): row for row in records}
    require(len(records) == len(lookup) == 48, 'Held-out case coverage differs')
    ordered, notes_count, events_count = [], 0, 0
    for index, item in enumerate(inputs):
        entry, audio, duration, reference = item['entry'], item['audio'], item['duration'], item['reference']
        exported_audio, rate = sf.read(output / 'test/clips' / (entry['id'] + '.wav'), dtype='float32')
        require(rate == 16000 and np.array_equal(exported_audio, audio), 'Review audio differs from the scored excerpt')
        for mode in MODES[index % 4:] + MODES[:index % 4]:
            row = lookup[(entry['id'], mode)]
            path = output / row['probability_path']
            require(sha256(path) == row['probability_sha256'] and row['input_provenance'] == provenance[index], 'Recorded heads/input changed')
            with np.load(path, allow_pickle=False) as arrays: probability = {key: arrays[key] for key in arrays.files}
            notes, events, _ = replay(probability, base, config['decoder_parameters'], duration)
            require(without_compute(row['notes']) == notes and without_compute(row['events']) == events, 'Exact source decoder replay differs')
            require(all(event['delivered_at_seconds'] + 1e-9 >= event['emitted_at_audio_seconds'] for event in row['events']), 'Completion precedes availability')
            heads = aligned_heads(probability, item['targets'], config, base, config['source_frames'])
            diagnosis, _ = state_diagnosis(probability, item['targets'], frame_config(config))
            require(row['head_metrics'] == heads and row['frame_diagnosis'] == diagnosis
                    and row['state_derived_pitch'] == diagnosis['state_derived_pitch']
                    and row['best_active_confusion'] == fret_confusion(probability, item['targets']), 'Head/frame/fret scoring differs')
            verify_frame_partition(diagnosis)
            require(row['reference'] == reference
                    and row['pitch_metrics'] == evaluate(reference, row['notes'], protocol['scoring_config'], duration)
                    and row['tab_metrics'] == tablature_metrics(reference, row['notes'], protocol['scoring_config'], 10, duration)
                    and row['latency'] == measure(reference, row['notes'], protocol['scoring_config'], duration), 'Note/string/completion scoring differs')
            stem = output / 'test/results' / mode / entry['id']
            verify_exports(stem, row)
            require(len(row['events']) == 2 * len(row['notes']), 'Event ledger lost an attack or release')
            require(row['timing']['lookahead_samples'] == 0 and row['timing']['feature_windows_samples'] == [2048, 2048]
                    and row['timing']['unique_frontend_fft_count'] == 1, 'Fixed frontend changed')
            ordered.append(row)
            notes_count += len(row['notes'])
            events_count += len(row['events'])
    summary = summarize(ordered)
    summary.update(stage=23, **locked_selection(config))
    require(summary == read(output / 'test/summary.json') and contrasts(summary) == read(output / 'comparison.json'), 'Held-out summary or descriptive contrasts differ')
    names = ['protocol.json', 'selection.json', 'predecessor-bindings.json', 'input-bindings.json', 'test-inputs.json',
             'model-integrity.json', 'comparison.json', 'test/run.json', 'test/summary.json']
    result = {'passed': True, 'new_optimizer_updates': 0, 'optimizer_instantiations': 0, 'normalization_updates': 0,
        'model_selection_performed': False, 'locked_selected_arm': config['locked_selected_arm'], 'selection_stage': 22,
        'test_scored_replays': 48, 'test_audio_warmups': 4, 'target_cache_reproductions': 12,
        'exact_notes': notes_count, 'exact_events': events_count, 'checkpoint_files_unchanged': 4,
        'new_optimizer_updates_for_audit': 0, 'neural_inference_calls_for_audit': 0,
        'development_selection_byte_identical': True, 'production_promoted': False,
        'source_sha256': source_hashes(), 'player_overlap': protocol['player_overlap'],
        'artifact_sha256': {name: sha256(output / name) for name in names}}
    save_json(output / 'audit.json', result)
    print('Stage23 audit passed: 48 cached replays, exact exports, unchanged checkpoints and copied development selection.', flush=True)
    return result
