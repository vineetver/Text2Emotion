import json
from argparse import Namespace
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from src.dataset.create_dataset import read_dataset, combine_dataset
from src.feature.preprocessing import drop_annotator_column, str_to_index, apply_index_to_class, apply_ekman_mapping, \
    apply_clean_text, one_hot_encode
from src.model import optimization
from src.model.classifier import BERT
from src.model.optimization import load_artifacts
from src.utils import get_dict, write_dict

app = typer.Typer()


@app.command()
def etl_data():
    """
    Extracts, loads and transformers our data
    """
    logger.info('✅ Starting Data extraction, load and transform ✅')

    # Read the training, testing, and validation datasets
    train_df, test_df, valid_df = read_dataset(
        config.TRAIN_URL, config.TEST_URL, config.VALID_URL)

    # Combine the training, testing, and validation datasets using concat
    df = combine_dataset(train_df, test_df, valid_df)

    # drop the 'annotator' column from the dataset
    df = drop_annotator_column(df)

    # convert the string class indices to list of indices
    df = str_to_index(df)

    # convert the list of indices to list of class labels
    df = apply_index_to_class(df)

    # apply the Ekman mapping to the dataset to reduce the number of classes
    df = apply_ekman_mapping(df)

    # apply the clean text function to the dataset (remove punctuation, lowercase, etc.)
    df = apply_clean_text(df)

    # one-hot encode the class labels
    df = one_hot_encode(df)

    df.to_csv(Path(config.DATA_DIR, 'preprocessed.csv'),
              sep='\t', encoding='utf-8')

    logger.info('✅ Data successfully extracted and transformed ✅')


@app.command()
def train_model(params_path: str = 'config/parameters.json', experiment_name: str = 'bert_model', run_name: str = 'PolynomialDecay',
                test_run: bool = False) -> None:
    """
    Trains a model with the provided parameters

    Args:
        params_path: path to the parameters file
        experiment_name: name of the experiment
        run_name: name of the run
        test_run: whether to test the model or not
    """
    # Load data
    df = pd.read_csv(Path(config.DATA_DIR, 'preprocessed.csv'),
                     sep='\t', encoding='utf-8')

    params = Namespace(**get_dict(filepath=params_path))

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f'🏁 Starting Run ID: {run_id} 🏁')
        bert = BERT(params=params)
        artifacts = bert.fit(df)
        performance = artifacts['metrics']

        # log metrics and params
        mlflow.log_metrics({'precision': performance['overall']['precision']})
        mlflow.log_metrics({'recall': performance['overall']['recall']})
        mlflow.log_metrics({'f1': performance['overall']['f1']})

        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        artifacts_path = Path(config.MODEL_REGISTRY, experiment_id, run_id, 'artifacts')
        write_dict(vars(artifacts['params']), Path(artifacts_path, 'params.json'), cls=NumpyEncoder)
        artifacts['model'].save(Path(artifacts_path, 'bert_model.hdf5'))
        write_dict(performance, Path(artifacts_path, 'metrics.json'))
        mlflow.log_artifact(str(artifacts_path))

        if not test_run:
            open(Path(config.CONFIG_DIR, 'run_id.txt'), 'w').write(run_id)
            write_dict(performance, Path(
                config.CONFIG_DIR, 'performance.json'))

        logger.info(f'🏁 Run ID: {run_id} Finished 🏁')


@app.command()
def optimize(params_path: str = 'config/parammeters.json', experiment_name: str = 'optimization', n_trails: int = 10) -> None:
    """This function optimizes the model's hyperparameters

    Args:
        params_path (str, optional): Location where hyperparameters are stored. Defaults to 'config/parammeters.json'.
        experiment_name (str, optional): name of the experiment e.g. optimization. Defaults to 'optimization'.
        n_trails (int, optional): number of trails to run. Defaults to 10.

    Returns:
        None
    """
    # Load data
    df = pd.read_csv(Path(config.DATA_DIR, 'preprocessed.csv'),
                     sep='\t', encoding='utf-8')

    params = Namespace(**get_dict(filepath=params_path))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=experiment_name, direction='maximize', pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name='f1')

    logger.info('🏁 Starting Optimization 🏁')
    study.optimize(
        lambda trial: optimization.objective(params, df, trial),
        n_trials=n_trails,
        callbacks=[mlflow_callback]
    )
    logger.info('🏁 Optimization Complete 🏁')

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(['user_attrs_f1'], ascending=False)
    params = {**params.__dict__, **study.best_trial.params}
    write_dict(params, filepath=params_path, cls=NumpyEncoder)
    logger.info(f'✅ Best value of F1 Score {study.best_trial.value} ✅')
    logger.info(
        f'✅ Best HyperParammeters: {json.dumps(study.best_trial.params, indent=2)} ✅')


@app.command()
def predict_emotion(prompt: str = None, run_id: str = None) -> None:
    """This function predicts the emotion of a prompt (text)

    Args:
        prompt (str, optional): prompt to predict the emotion of. Defaults to None.
        run_id (str, optional): run_id of the model to use. Defaults to None.
    """
    if not run_id:
        # TO-DO CHANGE PATH TO FINAL MODEL DIR
        run_id = open(Path(config.CONFIG_DIR, 'run_id.txt')).read()

    artifacts = load_artifacts(run_id=run_id)

    # Load weights
    bert = BERT(params=artifacts['params'])
    model = bert.model
    # TO-DO CHANGE PATH TO FINAL MODEL DIR
    model.load_weights(Path(config.MODEL_REGISTORY, 'bert_model.hdf5'))

    # Predict
    pred, prob = bert.predict(
        prompt=prompt, threshold=artifacts.params.threshold, model=model)
    logger.info(f'🏁 Prediction: {pred} and probability {prob} 🏁')

    return pred, prob


if __name__ == '__main__':
    app()
