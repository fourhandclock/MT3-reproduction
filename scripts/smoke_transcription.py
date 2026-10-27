"""Exercise the real CLI and independently check its MIDI/JSON output contract."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
AUDIO = 'outputs/fret19/validation/clips/00_Rock2-85-F_comp.wav'
PACKAGES = ('torch', 'numpy', 'scipy', 'librosa', 'soundfile', 'mir_eval', 'pretty_midi', 'mido')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_origins():
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        raise ValueError('Smoke run requires a virtual environment')
    if 'include-system-site-packages = false' not in (prefix / 'pyvenv.cfg').read_text():
        raise ValueError('System site packages must be disabled')
    result = {}
    for name in PACKAGES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise ValueError(f'{name} was imported outside this environment: {path}')
        result[name] = str(path)
    return result


def validate_prediction(midi_path, checkpoint_sha256, audio, seconds):
    import pretty_midi

    result = json.loads(midi_path.with_suffix('.json').read_bytes())
    if (result['checkpoint_sha256'] != checkpoint_sha256
            or Path(result['input_audio']).resolve() != audio.resolve()
            or not math.isclose(result['duration_seconds'], seconds, abs_tol=1 / 16000)
            or result['tuning_midi'] != [40, 45, 50, 55, 59, 64]
            or result['decoder_source_sha256'] != digest(ROOT / 'cnn_gru/inference.py')):
        raise ValueError('Prediction provenance/duration/tuning differs')
    notes = result['notes']
    if not notes:
        raise ValueError('Known guitar smoke clip produced no notes')
    for note in notes:
        if (not all(math.isfinite(note[key]) for key in ('start', 'end'))
                or not 0 <= note['start'] < note['end'] <= seconds + 1 / 480
                or not 0 <= note['string'] < 6 or not 0 <= note['fret'] <= 24
                or note['pitch'] != result['tuning_midi'][note['string']] + note['fret']
                or note['program'] != 24):
            raise ValueError('Invalid timed string/fret note')
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    exported = []
    for instrument in midi.instruments:
        if instrument.is_drum or instrument.program != 24 or not instrument.name.startswith('string_'):
            raise ValueError('Unexpected MIDI instrument')
        string = int(instrument.name.split('_')[-1])
        for note in instrument.notes:
            exported.append((string, note.pitch, note.start, note.end))
    expected = [(note['string'], note['pitch'], note['start'], note['end']) for note in notes]
    if sorted(exported) != sorted(expected):
        raise ValueError('MIDI/JSON notes differ')
    for string in range(6):
        ordered = sorted((note for note in notes if note['string'] == string), key=lambda n: n['start'])
        if any(a['end'] > b['start'] + 1e-9 for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Overlapping frets on one string')
    return {'note_count': len(notes), 'duration_seconds': result['duration_seconds'],
            'midi_json_notes_identical': True, 'valid_string_fret_pitch_and_intervals': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if not directory.is_relative_to((ROOT / 'tmp/fresh-transcription').resolve()):
        parser.error('--output-dir must be under tmp/fresh-transcription')
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ('prediction.mid', 'prediction.json', 'smoke.json')):
        parser.error('Use a new output directory for each smoke run')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
              'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'device': 'cpu',
              'scope': 'Environment and export smoke check using an existing local checkpoint/clip; not an accuracy benchmark.'}
    try:
        if Path(sys.prefix).resolve() != directory / 'env':
            raise ValueError('Use the freshly created env inside this run directory')
        report['package_origins'] = package_origins()
        # setuptools can add its vendored distributions to sys.path on import.
        # Inspect installed site directories explicitly so those copies cannot
        # replace the real installed versions in this environment receipt.
        sites = sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})
        report['packages'] = {dist.metadata['Name']: dist.version
                              for dist in importlib.metadata.distributions(path=sites)}
        index = json.loads((ROOT / 'experiments/index.json').read_bytes())
        latest = index['latest_snapshot']
        snapshot_path = ROOT / latest['path']
        if digest(snapshot_path) != latest['sha256']:
            raise ValueError('Tracked snapshot hash differs')
        snapshot = json.loads(snapshot_path.read_bytes())
        audio = ROOT / AUDIO
        audio_key = audio.relative_to(ROOT / latest['artifact_root']).as_posix()
        checkpoint = ROOT / index['default']['checkpoint']
        expected_checkpoint = index['default']['checkpoint_sha256']
        if digest(checkpoint) != expected_checkpoint or digest(audio) != snapshot['artifact_sha256'][audio_key]:
            raise ValueError('Smoke input/checkpoint hash differs')
        report.update(audio=AUDIO, audio_sha256=digest(audio), checkpoint=index['default']['checkpoint'],
                      checkpoint_sha256=expected_checkpoint, snapshot_sha256=latest['sha256'])
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env['PYTHONNOUSERSITE'] = '1'
        command = [sys.executable, '-m', 'cnn_gru', 'transcribe', str(audio),
                   '--checkpoint', str(checkpoint), '--device', 'cpu', '--seconds', '12',
                   '--output', str(directory / 'prediction.mid')]
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        (directory / 'transcription.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise ValueError(f'Transcription failed ({result.returncode}); see transcription.log')
        report['cli_command'] = command
        report['output_checks'] = validate_prediction(directory / 'prediction.mid', expected_checkpoint, audio, 12)
        report['output_sha256'] = {name: digest(directory / name) for name in ('prediction.mid', 'prediction.json')}
        report['passed'] = True
        print(f"Smoke passed: {report['output_checks']['note_count']} timed notes; MIDI/JSON agree.", flush=True)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        report['passed'] = False
        report['error'] = str(exc)
        print(f'Smoke failed: {exc}'.encode('ascii', errors='backslashreplace').decode('ascii'), flush=True)
    (directory / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
