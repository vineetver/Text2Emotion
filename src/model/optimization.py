import json
from argparse import Namespace
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd

from config import config
from config.config import logger
from src.model.classifier import BERT
from src.utils import get_dict


def load_artifacts(run_id: str = None) -> dict:
    """This function loads the artifacts from a given run_id

    Args:
        run_id (str, optional): id of the run to load artifacts from. Defaults to None.

    Returns:
        dict: artifacts
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, 'run_id.txt')).read()

    # locate artifacts
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY,
                         experiment_id, run_id, 'artifacts')

    # Load Data
    params = Namespace(**get_dict(filepath=Path(artifacts_dir, 'params.json')))
    model = joblib.load(Path(artifacts_dir, 'model.pkl'))
    performance = joblib.load(Path(artifacts_dir, 'performance.json'))

    return {
        'params': params,
        'model': model,
        'performance': performance
    }


def objective(params: Namespace, df: pd.DataFrame, trial: optuna.trial.Trial) -> float:
    """
    Objective function for optimization of hyperparameters.

    Args:
        params (Namespace): hyperparameters
        df (pd.DataFrame): dataframe
        trail (optuna.trial.Trial): trial

    Returns:
        float: metric value to be used as score
    """

    params.initial_learning_rate = trial.suggest_float(
        'learning_rate', 3e-6, 1e0)
    params.clipnorm = trial.suggest_float('clipnorm', 0.01, 1)
    params.decay_steps = trial.suggest_int('decay_steps', 1, 600)

    # train and evaluate model
    bert = BERT(params=params)
    artifacts = bert.fit(df=df, trial=trial)

    # set additional metrics
    overall_performance = artifacts['metrics']['overall']
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr('precision', overall_performance['precision'])
    trial.set_user_attr('recall', overall_performance['recall'])
    trial.set_user_attr('f1', overall_performance['f1'])

    return overall_performance['f1']
