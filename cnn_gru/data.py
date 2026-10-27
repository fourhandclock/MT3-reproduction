import hashlib
import http.client
import json
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

from baseline.download import ROOT, RangeReader, load_record, select_ids, sha256
from baseline.evaluation import STANDARD_TUNING, reference_notes


class RetryRangeReader(RangeReader):
    def read(self,size=-1):
        for attempt in range(4):
            try:
                return super().read(size)
            except (http.client.IncompleteRead,ConnectionResetError,TimeoutError) as exc:
                if attempt==3: raise
                print(f'Retrying interrupted range download ({attempt+1}/3): {type(exc).__name__}',flush=True)
                time.sleep(min(2**attempt,4))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(config):
    metadata = load_record()
    archives = {entry['key']: entry for entry in metadata['files']}
    readers = [RetryRangeReader(archives[name]['links']['self'], archives[name]['size'])
               for name in ['annotation.zip', 'audio_mono-mic.zip']]
    test = json.loads((ROOT / 'data/manifest.json').read_text())
    manifest = {'dataset': 'GuitarSet', 'config': config, 'test_manifest_sha256': sha256(ROOT / 'data/manifest.json'),
                'split_rule': 'Progression 1 training, progression 2 validation, existing progression 3 final evaluation; players occur in all splits.',
                'license': metadata['metadata'].get('license'), 'train': [], 'validation': [], 'test': test['excerpts']}
    with zipfile.ZipFile(readers[0]) as annotations, zipfile.ZipFile(readers[1]) as audio:
        annotation_names = {Path(name).stem: name for name in annotations.namelist()
                            if name.endswith('.jams') and '__MACOSX' not in name}
        audio_names = {Path(name).name: name for name in audio.namelist()
                       if name.endswith('.wav') and '__MACOSX' not in name}
        for split, count, progression in [('train', config['train_count'], config['train_progression']),
                                          ('validation', config['validation_count'], config['validation_progression'])]:
            selected = select_ids(annotation_names.values(), count, progression, config['seed'])
            for clip_id in selected:
                entry = {'id': clip_id}
                for archive, member, folder, key in [
                    (annotations, annotation_names[clip_id], 'annotations', 'annotation'),
                    (audio, audio_names[clip_id + '_mic.wav'], 'audio', 'audio')]:
                    target = ROOT / 'data/guitarset' / folder / Path(member).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if zlib.crc32(target.read_bytes()) & 0xffffffff != archive.getinfo(member).CRC:
                            raise IOError(f'CRC mismatch: {target}')
                    else:
                        target.write_bytes(archive.read(member))
                    entry[key] = target.relative_to(ROOT).as_posix()
                    entry[key + '_sha256'] = sha256(target)
                manifest[split].append(entry)
                print('Prepared', split, clip_id, flush=True)
    validate_splits(manifest)
    manifest['downloaded_bytes_this_run'] = sum(reader.downloaded_bytes for reader in readers)
    destination = ROOT / 'data/cnn_gru'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Data ready; newly downloaded MiB:', round(manifest['downloaded_bytes_this_run'] / 1024**2, 1), flush=True)


def validate_splits(manifest):
    import re
    groups = {}
    for split in ['train', 'validation', 'test']:
        ids = [entry['id'] for entry in manifest[split]]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate source in {split}')
        groups[split] = set(ids)
        progression = {'train': 1, 'validation': 2, 'test': 3}[split]
        if any(int(re.match(r'^\d\d_[A-Za-z]+([123])-', clip_id).group(1)) != progression for clip_id in ids):
            raise ValueError(f'Unexpected progression in {split}')
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if groups[left] & groups[right]:
            raise ValueError(f'Split leakage: {left} / {right}')


def load_audio(path, seconds, config):
    import librosa
    import soundfile as sf
    with sf.SoundFile(path) as source:
        sample_rate = source.samplerate
        audio = source.read(frames=int(seconds * sample_rate), dtype='float32', always_2d=True).mean(axis=1)
    desired_rate = config['feature']['sample_rate']
    if sample_rate != desired_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=desired_rate, res_type='soxr_hq')
    return audio.astype(np.float32), len(audio) / desired_rate


def features(audio, config):
    import librosa
    feature = config['feature']
    result = librosa.cqt(audio, sr=feature['sample_rate'], hop_length=feature['hop_length'],
                         fmin=librosa.midi_to_hz(feature['fmin_midi']), n_bins=feature['n_bins'],
                         bins_per_octave=feature['bins_per_octave'], filter_scale=feature['filter_scale'],
                         tuning=0.0, pad_mode='constant', dtype=np.complex64)
    count = int(np.ceil(len(audio) / feature['hop_length']))
    result = np.log(np.maximum(np.abs(result[:, :count].T), feature['log_floor'])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite CQT feature.')
    return result


def make_targets(notes, count, config):
    model = config['model']
    fret_count = model['max_fret'] + 1
    fps = config['feature']['sample_rate'] / config['feature']['hop_length']
    duration = count / fps
    states = np.full((count, 6), fret_count, dtype=np.int16)
    occupancy = np.zeros((count, 6, fret_count), dtype=bool)
    onsets, offsets = np.zeros((count, 6), dtype=np.float32), np.zeros((count, 6), dtype=np.float32)
    pitch = np.zeros((count, model['pitch_max'] - model['pitch_min'] + 1), dtype=np.uint8)

    def mark_boundary(target, seconds, string):
        if not 0 <= seconds < duration:
            return
        center = int(np.rint(seconds * fps))
        for frame in range(max(0, center - 3), min(count, center + 4)):
            target[frame, string] = max(target[frame, string], np.exp(-0.5 * (frame - center)**2))

    for note in notes:
        if note['start'] >= duration or note['end'] <= 0:
            continue
        string, fret, midi_pitch = note['string'], note['fret'], note['pitch']
        if not 0 <= string < 6 or not 0 <= fret < fret_count or not model['pitch_min'] <= midi_pitch <= model['pitch_max']:
            raise ValueError(f'Note outside supported standard-tuning fretboard: {note}')
        start = max(0, int(np.ceil(note['start'] * fps)))
        end = min(count, int(np.ceil(note['end'] * fps)))
        occupancy[start:end, string, fret] = True
        pitch[start:end, midi_pitch - model['pitch_min']] = 1
        mark_boundary(onsets, note['start'], string)
        mark_boundary(offsets, note['end'], string)
    active = occupancy.any(axis=-1)
    states[active] = occupancy.argmax(axis=-1)[active]
    conflicts = occupancy.sum(axis=-1) > 1
    states[conflicts] = -100  # Mask inconsistent simultaneous fret labels on one string.
    return {'state': states, 'onset': onsets.astype(np.float16), 'offset': offsets.astype(np.float16),
            'pitch': pitch}, int(conflicts.sum())


def cache_path(entry, config, seconds):
    cache_id = fingerprint({'feature': config['feature'],
                            'model_targets': {key:config['model'][key] for key in ['max_fret','pitch_min','pitch_max']},
                            'seconds': seconds, 'audio_sha256': entry['audio_sha256'],
                            'annotation_sha256': entry['annotation_sha256']})[:16]
    return ROOT / 'data/cnn_gru/cache' / (entry['id'] + '-' + cache_id)


def cache_data(config):
    manifest = json.loads((ROOT / 'data/cnn_gru/manifest.json').read_text())
    validate_splits(manifest)
    audit = {'config_sha256': fingerprint(config), 'splits': {}}
    for split in ['train', 'validation', 'test']:
        seconds = config['train_audio_seconds'] if split == 'train' else config['evaluation_audio_seconds']
        audit['splits'][split] = []
        for entry in manifest[split]:
            if sha256(ROOT / entry['audio']) != entry['audio_sha256'] or sha256(ROOT / entry['annotation']) != entry['annotation_sha256']:
                raise IOError(f'Source checksum mismatch: {entry["id"]}')
            prefix = cache_path(entry, config, seconds)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            if not prefix.with_suffix('.json').exists():
                audio, duration = load_audio(ROOT / entry['audio'], seconds, config)
                x = features(audio, config)
                targets, conflicts = make_targets(reference_notes(ROOT / entry['annotation']), len(x), config)
                np.save(prefix.with_suffix('.npy'), x)
                np.savez_compressed(prefix.with_suffix('.npz'), **targets)
                info = {'id': entry['id'], 'duration': duration, 'frames': len(x), 'conflict_string_frames': conflicts,
                        'features_sha256': sha256(prefix.with_suffix('.npy')), 'targets_sha256': sha256(prefix.with_suffix('.npz'))}
                prefix.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
            info = json.loads(prefix.with_suffix('.json').read_text())
            if sha256(prefix.with_suffix('.npy')) != info['features_sha256'] or sha256(prefix.with_suffix('.npz')) != info['targets_sha256']:
                raise IOError(f'Cache checksum mismatch: {entry["id"]}')
            audit['splits'][split].append(info)
            print('Cached', split, entry['id'], info['frames'], 'frames', flush=True)
    # Normalization and class-balancing statistics use training data only.
    total = 0
    sums = np.zeros(config['feature']['n_bins'], dtype=np.float64)
    squares = sums.copy()
    onset_sum = offset_sum = pitch_sum = 0.0
    boundary_count = pitch_count = 0
    for entry in manifest['train']:
        prefix = cache_path(entry, config, config['train_audio_seconds'])
        x = np.load(prefix.with_suffix('.npy'), mmap_mode='r').astype(np.float32)
        sums += x.sum(axis=0, dtype=np.float64)
        squares += np.square(x).sum(axis=0, dtype=np.float64)
        total += len(x)
        with np.load(prefix.with_suffix('.npz')) as target:
            onset_sum += target['onset'].sum(dtype=np.float64)
            offset_sum += target['offset'].sum(dtype=np.float64)
            pitch_sum += target['pitch'].sum(dtype=np.float64)
            boundary_count += target['onset'].size
            pitch_count += target['pitch'].size
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - mean**2, 0.09))
    audit['normalization_source'] = 'train only'
    audit['normalization'] = {'mean': mean.tolist(), 'std': std.tolist()}
    audit['positive_weights'] = {'onset': float(np.clip((boundary_count-onset_sum)/max(onset_sum,1),2,20)),
                                 'offset': float(np.clip((boundary_count-offset_sum)/max(offset_sum,1),2,20)),
                                 'pitch': float(np.clip((pitch_count-pitch_sum)/max(pitch_sum,1),1,8))}
    (ROOT / 'data/cnn_gru/cache_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return manifest, audit


class CropDataset:
    def __init__(self, entries, config, normalization):
        self.entries, self.config, self.epoch = entries, config, 0
        self.mean = np.array(normalization['mean'], dtype=np.float32)
        self.std = np.array(normalization['std'], dtype=np.float32)

    def __len__(self):
        return self.config['training']['samples_per_epoch']

    def __getitem__(self, index):
        import torch
        train = self.config['training']
        rng = np.random.default_rng(self.config['seed'] + self.epoch * 100000 + index)
        entry = self.entries[int(rng.integers(len(self.entries)))]
        prefix = cache_path(entry, self.config, self.config['train_audio_seconds'])
        source = np.load(prefix.with_suffix('.npy'), mmap_mode='r')
        length = round(train['crop_seconds'] * self.config['feature']['sample_rate'] / self.config['feature']['hop_length'])
        if len(source) < length:
            raise ValueError('Recording shorter than training crop.')
        start = int(rng.integers(len(source) - length + 1))
        x = source[start:start+length].astype(np.float32)
        x += rng.normal(0, train['gain_augmentation_log_std'])
        x = np.clip((x-self.mean)/self.std, -5, 5)
        with np.load(prefix.with_suffix('.npz')) as all_targets:
            targets = {key: torch.from_numpy(all_targets[key][start:start+length].copy()).long() if key == 'state'
                       else torch.from_numpy(all_targets[key][start:start+length].astype(np.float32)) for key in all_targets.files}
        mask = np.ones(length, dtype=np.float32)
        edge = train['edge_mask_frames']
        if edge:
            mask[:edge] = mask[-edge:] = 0
        targets['mask'] = torch.from_numpy(mask)
        return torch.from_numpy(x), targets
