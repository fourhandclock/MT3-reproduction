import argparse
import contextlib
import ctypes
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .download import ROOT, MT3_SHA256, prepare_guitarset, prepare_mt3, sha256


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not 0 < config['score_seconds'] < config['clip_seconds']:
        raise ValueError('Use a positive scored prefix shorter than the input, retaining release context.')
    return config


def environment():
    import torch
    packages = ['torch', 'torchaudio', 'transformers', 'mt3-infer', 'basic-pitch',
                'onnxruntime', 'librosa', 'numpy', 'scipy', 'mir-eval']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'logical_cpu_count': os.cpu_count(), 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2
            if torch.cuda.is_available() else None,
            'packages': {name: importlib.metadata.version(name) for name in packages}}


def doctor():
    info = environment()
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ['physical', 'available', 'pagefile', 'available_pagefile', 'virtual', 'available_virtual', 'extended']]
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            info['ram_gib'] = memory.physical / 1024**3
            info['available_ram_gib'] = memory.available / 1024**3
    output = ROOT / 'outputs/hardware.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2), flush=True)


def load_backend(name, device, config):
    if name in ('mt3-pytorch', 'mt3-reference'):
        import torch
        from mt3_infer import load_model
        if sha256(ROOT / 'models/mt3_pytorch/mt3.pth') != MT3_SHA256:
            raise IOError('MT3 checkpoint no longer matches the recorded checksum.')
        model = load_model('mt3_pytorch', checkpoint_path=str(ROOT / 'models/mt3_pytorch'),
                           device=device, cache=False, auto_download=False,
                           auto_filter=config['mt3_auto_filter'])
        if name == 'mt3-reference':
            from .frontend import ReferenceSpectrogramProcessor
            model._spectrogram_processor = ReferenceSpectrogramProcessor(model._spectrogram_processor)

        def predict(audio_path, midi_path):
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype='float32')
            with torch.inference_mode():
                midi = model.transcribe(audio, sr=sr)
            midi.save(str(midi_path))
            return None

        provenance = json.loads((ROOT / 'models/mt3_pytorch/provenance.json').read_text())
        provenance['device'] = str(model._device)
        provenance['frontend'] = 'MT3 reference-style magnitude, end-padded STFT and mel-coordinate triangles' if name == 'mt3-reference' else 'unmodified third-party centered power frontend'
        if name == 'mt3-reference':
            provenance['frontend_sha256'] = sha256(ROOT / 'baseline/frontend.py')
        return predict, provenance

    import onnxruntime as ort
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model, predict as basic_predict
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    # Supply the library's Model interface with an explicitly bounded ONNX session.
    options = ort.SessionOptions()
    options.intra_op_num_threads = config['cpu_threads']
    options.inter_op_num_threads = 1
    model = Model.__new__(Model)
    model.model_type = Model.MODEL_TYPES.ONNX
    model.model = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])

    def predict(audio_path, midi_path):
        with contextlib.redirect_stdout(io.StringIO()):
            probabilities, midi, _ = basic_predict(audio_path, model)
        midi.write(str(midi_path))
        return probabilities

    provenance = {'implementation': 'Spotify Basic Pitch 0.4.0, ONNX CPU',
                  'model_sha256': sha256(model_path), 'device': 'cpu',
                  'providers': model.model.get_providers(),
                  'parameters': {'onset_threshold': 0.5, 'frame_threshold': 0.3,
                                 'minimum_note_length_ms': 127.7, 'melodia_trick': True},
                  'training_overlap': 'GuitarSet used in Basic Pitch research; exact overlap not established here'}
    return predict, provenance


def run(args, config):
    import numpy as np
    import soundfile as sf
    import torch
    from .evaluation import evaluate, midi_notes, reference_notes
    torch.set_num_threads(config['cpu_threads'])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    manifest_path = ROOT / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['selection'] != config:
        raise ValueError('Data selection/configuration differ. Run prepare with the current configuration first.')
    selected = manifest['excerpts'][:args.limit] if args.limit else manifest['excerpts']
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    clips_dir = directory / 'clips'
    clips_dir.mkdir(exist_ok=True)
    for entry in selected:
        if sha256(ROOT / entry['audio']) != entry['audio_sha256'] or sha256(ROOT / entry['annotation']) != entry['annotation_sha256']:
            raise IOError(f'Data integrity check failed for {entry["id"]}')
        audio, sr = sf.read(ROOT / entry['audio'], dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[:int(sr * config['clip_seconds'])]
        sf.write(clips_dir / (entry['id'] + '.wav'), audio, sr, subtype='FLOAT')
    metadata = {'config': config, 'manifest_sha256': sha256(manifest_path), 'environment': environment(),
                'source_sha256': {name: sha256(ROOT / 'baseline' / name) for name in ['__main__.py', 'evaluation.py', 'frontend.py']},
                'selected_ids': [entry['id'] for entry in selected],
                'timing_scope': 'Warm inference including file read, resampling, features, decoding and MIDI save; excludes model load, warmup and dataset/artifact preparation.',
                'training_performed': False, 'models': {},
                'limitations': ['Diagnostic acoustic-guitar subset, not a full paper reproduction.',
                                'MT3 backend is an unofficial PyTorch port; full original JAX model/decoder parity not independently established.',
                                'GuitarSet pretraining overlap must be accounted for before claiming generalization.',
                                'No predicted string/fret positions or meter in stage 1.',
                                'MT3 uses CUDA when available; Basic Pitch ONNX uses CPU, so speed is a deployment comparison, not a controlled architecture comparison.']}
    metadata_path = directory / 'run.json'
    if metadata_path.exists() and not args.overwrite:
        old = json.loads(metadata_path.read_text())
        if any(old.get(key) != metadata[key] for key in ['config', 'manifest_sha256', 'selected_ids', 'environment', 'source_sha256']):
            raise ValueError('Existing run has different configuration/data/environment. Use a new output directory or --overwrite.')
        metadata['models'] = old['models']
    for name in args.models:
        model_dir = directory / name
        model_dir.mkdir(exist_ok=True)
        print(f'Loading {name}...', flush=True)
        start = time.perf_counter()
        predict, provenance = load_backend(name, args.device, config)
        provenance['load_seconds'] = time.perf_counter() - start
        using_cuda = provenance['device'].startswith('cuda')
        first_audio, first_sr = sf.read(clips_dir / (selected[0]['id'] + '.wav'), dtype='float32')
        warmup_path = model_dir / 'warmup.wav'
        sf.write(warmup_path, first_audio[:2 * first_sr], first_sr, subtype='FLOAT')
        print(f'Warming up {name} on two seconds of audio...', flush=True)
        start = time.perf_counter()
        predict(warmup_path, model_dir / 'warmup.mid')
        if using_cuda:
            torch.cuda.synchronize()
        provenance['warmup_seconds'] = time.perf_counter() - start
        if name in metadata['models'] and not args.overwrite:
            previous = metadata['models'][name]
            for key in ['weights_sha256', 'config_sha256', 'model_sha256', 'device', 'auto_filter', 'parameters', 'frontend', 'frontend_sha256']:
                if previous.get(key) != provenance.get(key):
                    raise ValueError('Existing predictions use different model settings. Choose a new output directory.')
        metadata['models'][name] = provenance
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        for index, entry in enumerate(selected):
            output = model_dir / (entry['id'] + '.json')
            if output.exists() and not args.overwrite:
                print('Reusing completed result:', name, entry['id'], flush=True)
                continue
            midi_path = model_dir / (entry['id'] + '.mid')
            if using_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            probabilities = predict(clips_dir / (entry['id'] + '.wav'), midi_path)
            if using_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            clip_info = sf.info(clips_dir / (entry['id'] + '.wav'))
            reference = reference_notes(ROOT / entry['annotation'])
            predicted, drums = midi_notes(midi_path)
            result = evaluate(reference, predicted, config, clip_info.duration)
            result.update({'id': entry['id'], 'model': name, 'processing_seconds': elapsed,
                           'input_duration_seconds': clip_info.duration, 'real_time_factor': elapsed / clip_info.duration,
                           'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2 if using_cuda else None,
                           'gpu_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2 if using_cuda else None,
                           'drum_notes': drums, 'program_counts': dict(Counter(str(note.get('program')) for note in result['predicted']))
                           if name in ('mt3-pytorch', 'mt3-reference') else None})
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            if probabilities is not None:
                np.savez_compressed(model_dir / (entry['id'] + '.npz'), **probabilities)
            print(f'{name} {index + 1}/{len(selected)} {entry["id"]}: onset F1={result["onset"]["f1"]:.3f}, '
                  f'offset F1={result["onset_offset"]["f1"]:.3f}, {elapsed:.2f}s', flush=True)
        del predict
        import gc
        gc.collect()
        if using_cuda:
            torch.cuda.empty_cache()
    from .report import write_report
    write_report(directory)


def main():
    # Keep caches and temporary files in the writable project, including on Windows.
    for key, relative in [('HF_HOME', '.cache/huggingface'), ('NUMBA_CACHE_DIR', '.cache/numba'),
                          ('MPLCONFIGDIR', '.cache/matplotlib'), ('TEMP', 'tmp/runtime'), ('TMP', 'tmp/runtime')]:
        location = ROOT / relative
        location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    os.environ['TQDM_DISABLE'] = '1'
    parser = argparse.ArgumentParser(description='Small pretrained guitar transcription baseline. No training.')
    parser.add_argument('--config', default=str(ROOT / 'configs/stage1.json'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    prepare = commands.add_parser('prepare')
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument('--data-only', action='store_true')
    selection.add_argument('--models-only', action='store_true')
    execute = commands.add_parser('run')
    execute.add_argument('--models', nargs='+', choices=['mt3-pytorch', 'mt3-reference', 'basic-pitch'], default=['mt3-reference', 'basic-pitch'])
    execute.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    execute.add_argument('--limit', type=int)
    execute.add_argument('--output', default=str(ROOT / 'outputs/stage1'))
    execute.add_argument('--overwrite', action='store_true')
    report = commands.add_parser('report')
    report.add_argument('--output', default=str(ROOT / 'outputs/stage1'))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'doctor':
        doctor()
    elif args.command == 'prepare':
        if not args.models_only:
            prepare_guitarset(config)
        if not args.data_only:
            prepare_mt3()
    elif args.command == 'run':
        if args.limit is not None and args.limit <= 0:
            parser.error('--limit must be positive.')
        run(args, config)
    else:
        from .report import write_report
        write_report(Path(args.output).resolve())


if __name__ == '__main__':
    main()
