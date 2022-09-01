from pathlib import Path
from config import config
from src.model.optimization import load_artifacts, objective
import pandas as pd


def test_load_artifacts():
    run_id = open(Path(config.CONFIG_DIR, 'run_id.txt')).read()
    artifacts = load_artifacts(run_id=run_id)

    assert type(artifacts['params'].initial_learning_rate) == float, 'params.initial_learning_rate is not a float'
    assert type(artifacts['params'].clipnorm) == float, 'params.clipnorm is not a float'
    assert type(artifacts['params'].decay_steps) == int,  'params.decay_steps is not an int'
    assert type(artifacts['params'].shuffle) == bool,   'params.shuffle is not a bool'

    assert type(artifacts['metrics']['overall']) == dict, 'metrics.overall is not a dict'
    assert type(artifacts['metrics']['classes']) == dict, 'metrics.classes is not a dict'

    for key in artifacts['metrics']['overall']:
        assert type(artifacts['metrics']['overall'][key]) == float, 'metrics.overall.{} is not a float'.format(key)



