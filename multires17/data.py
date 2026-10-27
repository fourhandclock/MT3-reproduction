"""Read immutable stage-11 caches and create matched two-plane batches."""

import copy
from pathlib import Path
import numpy as np
import torch
from acoustic10.data import load_inputs as frozen_inputs
from baseline.download import ROOT, sha256
from decoder14.workflow import read
from optimization13.data import build_schedule
from spectral11.data import normalization_from_training
from streaming.data import fingerprint
from streaming.evaluate import save_json

ARMS = [{'name': 'short_short', 'windows': [2048, 2048]},
        {'name': 'short_long', 'windows': [2048, 4096]}]


def validate_config(config):
    old = read(ROOT/'configs/spectral11.json')
    expected = {'stage': 17, 'seed': 42, 'data_epoch_start': 0, 'total_optimizer_update_cap': 500,
                'arms': ARMS, 'common_target_delay_frames': 13, 'common_leading_mask_frames': 26,
                'training': old['training'], 'decoder_parameters': {'onset_threshold': .7,
                    'activity_threshold': .25, 'onset_backdate_seconds': 0.},
                'validation_packet_ms': 20, 'pitch_diagnostics': old['pitch_diagnostics'],
                'sanity_reference': 'outputs/acoustic10/sanity.json', 'test_inference_allowed': False,
                'production_promotion': False}
    if set(config) != set(expected)|{'selection_rule'} or any(config[key] != value for key,value in expected.items()):
        raise ValueError('The fixed fresh 250+250 Stage17 protocol changed')
    if not isinstance(config['selection_rule'], str) or not config['selection_rule']:
        raise ValueError('A final-only selection rule is required')


def model_configuration(base, config, arm):
    result = copy.deepcopy(base)
    result['training'].update(config['training'])
    result['feature_planes'] = arm['windows']
    result['stage_budget'] = {'total_max_optimizer_updates': 500, 'arm_updates': 250,
                              'new_sanity_updates': 0, 'purpose': 'Stage17 fresh spectral fusion'}
    return result


def load_inputs(config, output):
    validate_config(config)
    base, old_audit, datasets, provenance = frozen_inputs(read(ROOT/'configs/acoustic10.json'))
    path = ROOT/'outputs/spectral11/cache-audit.json'
    if sha256(path) != read(ROOT/'outputs/spectral11/verification.json')['cache_audit_sha256']:
        raise ValueError('Archived spectral cache audit changed')
    archived = read(path)
    for key,digest in read(ROOT/'outputs/spectral11/validation/run.json')['source_sha256'].items():
        if sha256(ROOT/key) != digest:
            raise ValueError('Archived spectral producer changed: '+key)
    long_records = {}
    bindings = {}
    for split,records in datasets.items():
        long_records[split] = []
        infos = archived['arms']['long256'][split]
        if len(records) != len(infos):
            raise ValueError('Cache record count differs')
        for record,info in zip(records, infos):
            if record['entry']['id'] != info['id']:
                raise ValueError('Cache ordering differs')
            for key in ['features_path', 'receipt_path', 'targets_path']:
                file = ROOT/info[key]
                digest = sha256(file)
                expected = info[{'features_path':'features_sha256','receipt_path':'receipt_sha256',
                                 'targets_path':'targets_sha256'}[key]]
                if digest != expected:
                    raise ValueError('Frozen cache changed: '+str(file))
                bindings[info[key]] = digest
            receipt = read(ROOT/info['receipt_path'])
            if (not receipt['full_vs_20ms_packets_byte_equal'] or receipt['lookahead_samples'] != 0
                    or info['audio_sha256'] != record['entry']['audio_sha256']
                    or info['annotation_sha256'] != record['entry']['annotation_sha256']):
                raise ValueError('Cache origin or causal receipt differs')
            features = np.load(ROOT/info['features_path'], mmap_mode='r')
            if features.shape != record['features'].shape or features.dtype != np.float32 or not np.isfinite(features).all():
                raise ValueError('Long cache shape/dtype/value differs')
            record['long_features'] = features
            long_records[split].append({**record, 'features':features})
    short = normalization_from_training(datasets['train'])
    long = normalization_from_training(long_records['train'])
    # The helper includes its frame count; archived statistics use the same object.
    if short != archived['normalization']['short128'] or long != archived['normalization']['long256']:
        raise ValueError('Train-only normalization does not reproduce the archived recipe')
    normalization = {}
    for arm in ARMS:
        stats = [short, short if arm['name']=='short_short' else long]
        normalization[arm['name']] = {key:[item[key] for item in stats] for key in ['mean','std']}
    audit = {'normalization':normalization, 'positive_weights':old_audit['positive_weights']}
    provenance.update(spectral_cache_audit_sha256=sha256(path), long_cache_bindings=bindings)
    input_audit = {**provenance, **audit, 'config_sha256':fingerprint(config),
                   'features_rewritten':False, 'targets_rewritten':False,
                   'normalization_training_frames':sum(len(r['features']) for r in datasets['train']),
                   'new_optimizer_updates_for_preparation':0}
    target = Path(output)/'input-audit.json'
    if target.exists() and read(target) != input_audit:
        raise ValueError('Previously frozen Stage17 inputs changed')
    if not target.exists():
        save_json(target,input_audit)
    provenance['input_audit_sha256'] = sha256(target)
    return base,audit,datasets,provenance,build_schedule(config,datasets['train'])


def batch_from_schedule(descriptors, records, targets, normalization, edge, arm):
    mean,std = (np.asarray(normalization[key], np.float32) for key in ['mean','std'])
    xs,pieces = [],[]
    for row in descriptors:
        index,start,length = row['record_index'],row['start_frame'],row['frames']
        record = records[index]
        second = record['features'] if arm['name']=='short_short' else record['long_features']
        x = np.stack([record['features'][start:start+length],second[start:start+length]], axis=1)
        x += row['gain_log']
        xs.append(np.clip((x-mean)/std,-5,5))
        item = {key:value[start:start+length].copy() for key,value in targets[index].items()}
        if edge:
            item['mask'][:edge] = item['mask'][-edge:] = 0
        pieces.append(item)
    tensors = {key:torch.from_numpy(np.stack([item[key] for item in pieces])).long() if key=='state'
               else torch.from_numpy(np.stack([item[key] for item in pieces]).astype(np.float32)) for key in pieces[0]}
    return torch.from_numpy(np.stack(xs)),tensors
