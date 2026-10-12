"""PRIMEU configuration and checkpoint assets; training stays in run_train."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from examples.primeu.contract import export_contract, load_checkpoint_contract, load_dataset_contract


def resolve_path(value, base):
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if '$' in expanded:
        raise ValueError(f'Unresolved environment variable in path: {value}')
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get('train'), dict):
        raise ValueError('Expected a train mapping in YAML')
    cfg = copy.deepcopy(cfg)
    train = cfg['train']
    unexpected = set(train['policy']) - {'type', 'config_path', 'device', 'pretrained_path'}
    if unexpected:
        raise ValueError(f'Put model/optimizer settings in the policy YAML, not train.policy: {sorted(unexpected)}')
    for obj, key in ((train['dataset'], 'root'), (train, 'output_dir'), (train['policy'], 'config_path')):
        if not obj.get(key):
            raise ValueError(f'PRIMEU requires an explicit {key}')
        obj[key] = str(resolve_path(obj[key], path.parent))
    if train['policy'].get('pretrained_path'):
        train['policy']['pretrained_path'] = str(resolve_path(train['policy']['pretrained_path'], path.parent))
    if train.get('eval_freq', 0) != 0:
        raise ValueError('This offline PRIMEU instance has no robot/simulation eval environment; set eval_freq: 0')
    if train.get('dataset', {}).get('skip_bad_video_samples', False):
        raise ValueError('PRIMEU data errors must not be hidden by random resampling')
    return cfg


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resume_invariants(section):
    dataset = section['dataset']
    return {'model': section['policy']['type'],
            'episodes': dataset.get('episodes'),
            'image_resize': dataset.get('image_resize'),
            'use_imagenet_stats': dataset.get('use_imagenet_stats', True)}


def source_record():
    root = Path(__file__).resolve().parents[2]
    git = shutil.which('git') or str(Path(sys.executable).parent / 'git')
    try:
        commit = subprocess.check_output([git, '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = 'unavailable'
    files = list((root / 'examples/primeu').rglob('*.py')) + list((root / 'examples/primeu/configs').rglob('*.yaml'))
    files += [root / 'scripts/core/run_train.py']
    versions = {}
    for name in ('torch', 'torchvision', 'torchcodec', 'lerobot', 'accelerate', 'diffusers', 'numpy', 'wandb'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    import lerobot
    dependency_root = Path(lerobot.__file__).resolve().parent
    dependency_files = [dependency_root / 'datasets/lerobot_dataset.py', dependency_root / 'processor/hil_processor.py']
    return {'commit': commit, 'python': sys.version, 'versions': versions,
            'lerobot_source_sha256': {str(p.relative_to(dependency_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependency_files},
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}


def run(config_path, *, check_only=False):
    cfg = load_config(config_path)
    section = cfg['train']
    contract = load_dataset_contract(section['dataset']['root'])
    policy_spec = yaml.safe_load(Path(section['policy']['config_path']).read_text())
    kind = section['policy']['type']
    if kind not in ('act', 'diffusion'):
        raise ValueError('PRIMEU currently supports act or diffusion')
    if policy_spec.get('n_obs_steps') != 1:
        raise ValueError('PRIMEU serving accepts one observation; use n_obs_steps: 1')
    if policy_spec.get('pretrained_path') or section['policy'].get('pretrained_path'):
        # Native resume is handled below; initialization from another run needs an explicit compatible export.
        init_path = section['policy'].get('pretrained_path') or policy_spec['pretrained_path']
        if load_checkpoint_contract(init_path).contract_sha256 != contract.contract_sha256:
            raise ValueError('Initialization checkpoint uses a different PRIMEU data contract')
    from scripts.core.run_train import (
        TrainPipelineConfig, _resolve_resume_checkpoint_dir,
        apply_cuda_visible_devices_from_train_cfg, run_train,
    )
    apply_cuda_visible_devices_from_train_cfg(section)
    train_cfg = TrainPipelineConfig(section)
    if train_cfg.resume:
        checkpoint = _resolve_resume_checkpoint_dir(train_cfg.output_dir) / 'pretrained_model'
        prior = load_checkpoint_contract(checkpoint)
        if prior.contract_sha256 != contract.contract_sha256:
            raise ValueError('Resume dataset contract/basis does not match checkpoint')
        provenance = json.loads((checkpoint / 'primeu/run.json').read_text())
        if provenance['policy_spec_sha256'] != fingerprint(policy_spec):
            raise ValueError('Resume policy YAML changed; restore the original model/optimizer configuration')
        if resume_invariants(provenance['config']['train']) != resume_invariants(section):
            raise ValueError('Resume cannot change model type, selected episodes or image normalization/resize')
        if provenance['dataset_root'] != section['dataset']['root']:
            raise ValueError('Resume requires the same dataset root')
    report = {'model': kind, 'state_dim': 60, 'action_dim': contract.action_dim,
              'contract_sha256': contract.contract_sha256, 'dataset_root': section['dataset']['root'],
              'output_dir': section['output_dir'], 'steps': section['steps'], 'batch_size_per_rank': section['batch_size'],
              'action_semantics': 'stored_absolute_joint_targets', 'robot_motion': False}
    print(json.dumps(report, indent=2), flush=True)
    if check_only:
        return report
    record = source_record()
    record.update(report, policy_spec_sha256=fingerprint(policy_spec), config=cfg, policy_spec=policy_spec)

    def save_assets(pretrained_model):
        export_contract(section['dataset']['root'], pretrained_model)
        assets = pretrained_model / 'primeu'
        (assets / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        (assets / 'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (assets / 'policy.yaml').write_text(yaml.safe_dump(policy_spec, sort_keys=False))

    run_train(train_cfg, checkpoint_callback=save_assets)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Validate config/contract without training or hardware')
    args = parser.parse_args()
    run(args.config, check_only=args.check)


if __name__ == '__main__':
    main()
