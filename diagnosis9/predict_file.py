"""Replay new isolated-guitar audio with either frozen stage-9 recipe."""

import json
from pathlib import Path

from baseline.download import ROOT, sha256
from cnn_gru.data import load_audio
from streaming.evaluate import load_checkpoint, save_json
from streaming.export import export
from .audit import validate_selection
from .replay import replay


def predict_file(audio_path, output, diagnostics_root, config, seconds, packet_ms, device, mode):
    diagnostics_root = Path(diagnostics_root).resolve()
    validate_selection(config, diagnostics_root)
    selection_path = diagnostics_root / 'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    parameters = (config['baseline_parameters'] if mode == 'strict'
                  else selection['selected_parameters'])
    checkpoint_path, checkpoint, model = load_checkpoint(ROOT / config['checkpoint_root'], device)
    # File loading/resampling finishes before incremental PCM consumption begins.
    audio_path = Path(audio_path).resolve()
    audio, duration = load_audio(audio_path, seconds, checkpoint['config'])
    notes, events, timing = replay(model, audio, checkpoint['normalization'], checkpoint['config'],
                                   parameters, device, packet_ms, mode)
    timing['model_warmup_performed'] = False
    timing['timing_scope'] = timing['timing_scope'].replace('Warm mono16k', 'Mono16k (no explicit warmup)', 1)
    metadata = {'input_audio': str(audio_path), 'input_audio_sha256': sha256(audio_path),
                'checkpoint_sha256': sha256(checkpoint_path), 'selection_sha256': sha256(selection_path),
                'input_duration_seconds': duration, 'mode': mode, 'decoder_parameters': parameters,
                'packet_ms': packet_ms, 'lookahead_samples': 0, 'timing': timing,
                'optimizer_updates': 0,
                'scope': 'Unpaced isolated-guitar file replay after offline loading/resampling. '
                         'The repair is a research ablation and is not promoted over the strict decoder. '
                         'Native microphone capture, source separation and notation are outside this loop.'}
    output = Path(output).resolve()
    export(notes, output, metadata)
    save_json(output.with_name(output.name + '-events.json'), {'metadata': metadata, 'events': events})
    print('Frozen decoder replay exported:', output, ';', len(notes), 'notes;', len(events), 'events', flush=True)
    return {'notes': notes, 'events': events, 'metadata': metadata}
