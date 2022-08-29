from pathlib import Path
from argparse import Namespace
from src.utils import get_dict, write_dict
from config import config
import typer
import json
import tempfile
import pandas as pd
from numpyencoder import NumpyEncoder
import tensorflow as tf
import joblib
import mlflow
from config.config import logger
import keras.backend as K
from src.dataset.create_dataset import read_dataset, write_dataset, combine_dataset, split_dataset
from src.feature.preprocessing import drop_annotator_column, str_to_index, apply_index_to_class, apply_ekman_mapping, \
    apply_clean_text, one_hot_encode
from src.model.classifier import BERT

app = typer.Typer()


@app.command()
def etl_data():
    """
    Extracts, loads and transformers our data
    """
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

    logger.info('✅ Data successfully extracted and transformed')


def train_model(
        params_path: str = 'config/parameters.json',
        experiment_name: str = 'bert_model',
        run_name: str = 'PolynomialDecay',
        test_run: bool = False
) -> None:
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
        logger.info(json.dumps(performance, indent=2))

        # Clear session
        K.clear_session()

        # log metrics and params
        mlflow.log_metrics({'precision': performance['overall']['precision']})
        mlflow.log_metrics({'recall': performance['overall']['recall']})
        mlflow.log_metrics({'f1': performance['overall']['f1']})
        mlflow.log_metrics(vars(artifacts['params']))

        # log artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            write_dict(vars(artifacts['params']), Path(
                tmpdir, 'params.json'), cls=NumpyEncoder)
            joblib.dumb(artifacts['model'], Path(tmpdir, 'model.pkl'))
            write_dict(performance, Path(tmpdir, 'metrics.json'))
            mlflow.log_artifact(tmpdir)

        if not test_run:
            open(Path(config.CONFIG_DIR, 'run_id.txt'), 'w').write(run_id)
            write_dict(performance, Path(
                config.CONFIG_DIR, 'performance.json'))
