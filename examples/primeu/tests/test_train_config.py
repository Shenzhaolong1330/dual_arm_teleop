"""Reject configuration drift before loading models or touching hardware."""
from pathlib import Path
import yaml
import pytest
from examples.primeu.train import load_config, resume_invariants

BASE = Path(__file__).resolve().parents[1] / 'configs/act_train.yaml'


def test_train_paths_resolve_relative_to_yaml():
    config = load_config(BASE)
    assert Path(config['train']['policy']['config_path']) == BASE.parent / 'policies/act.yaml'


@pytest.mark.parametrize('field,value', [('n_obs_steps', 2), ('normalization_mapping', {}), ('optimizer_lr', 0.1)])
def test_legacy_overrides_cannot_bypass_primeu_contract(tmp_path, field, value):
    config = yaml.safe_load(BASE.read_text())
    config['train']['policy'][field] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='policy YAML'):
        load_config(path)


def test_resume_invariants_track_model_and_input_processing():
    import copy
    original = load_config(BASE)['train']
    for key, value in [('episodes', [0]), ('use_imagenet_stats', False), ('image_resize', [224, 224])]:
        changed = copy.deepcopy(original)
        changed['dataset'][key] = value
        assert resume_invariants(original) != resume_invariants(changed)
    changed = copy.deepcopy(original)
    changed['steps'] += 10
    assert resume_invariants(original) == resume_invariants(changed)
