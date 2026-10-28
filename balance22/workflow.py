"""Single-use 450-update budget and a separate, repeatable validation command."""

import gc
from pathlib import Path
import torch
from baseline.download import sha256
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .data import load_inputs, validate_config
from .restore import restore_parent
from .training import train_arm, producer_hashes, verified_sanity


def run(config, output, device):
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Stage22 already started; its acoustic update budget cannot be repeated')
    base, audit, datasets, provenance, parent, schedule, weights = load_inputs(config, output)
    model, optimizer, scaler, proof = restore_parent(parent, device)
    if proof['model_parameters'] != 931361 or proof['parameter_states'] != 34:
        raise ValueError('Parent capacity or parameter ordering differs')
    del model, optimizer, scaler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    sanity = verified_sanity(config)
    save_json(output / 'protocol.json', {'config': config, 'source_sha256': producer_hashes(),
        'parent': provenance['parent'], 'total_new_optimizer_update_cap': 450,
        'planned_new_optimizer_updates': {arm['name']: 150 for arm in config['arms']},
        'historical_unique_parent_updates': 500, 'new_sanity_updates': 0,
        'restoration_preflight': proof, 'sanity_reference': sanity, 'test_inference_allowed': False,
        'production_promotion': False, 'selection_rule': config['selection_rule']})
    save_json(output / 'crop-schedule.json', {'descriptors': schedule, 'sha256': fingerprint(schedule)})
    receipts = {}
    for arm in config['arms']:
        receipts[arm['name']] = train_arm(arm, config, base, audit, datasets['train'], provenance,
                                         parent, schedule, weights, output, device)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if sum(value['new_optimizer_steps'] for value in receipts.values()) != 450:
        raise ValueError('Actual acoustic update total differs from the fixed cap')
    save_json(output / 'training-summary.json', {'config': config, 'arms': receipts,
        'new_optimizer_updates': 450, 'historical_unique_parent_updates': 500,
        'final_lifetime_updates_per_arm': 650, 'new_sanity_updates': 0,
        'additional_sampled_epoch_blocks_per_arm': 3, 'validation_used_during_training': False,
        'test_inference_calls': 0, 'source_sha256': producer_hashes(),
        'protocol_sha256': sha256(output / 'protocol.json'), 'sanity_reference': sanity})
    del datasets, parent
    gc.collect()
    from .evaluation import evaluate_validation
    return evaluate_validation(config, output, device)
