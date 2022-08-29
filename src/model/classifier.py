from abc import ABC
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from keras.initializers.initializers_v2 import TruncatedNormal
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support, f1_score
import keras.backend as K
from tensorflow.python.ops.gen_dataset_ops import BatchDataset
from transformers import TFBertModel, BertTokenizerFast, BertConfig, TFAutoModelForSequenceClassification
from typing import Tuple, List

from config import config
from config.config import logger
from src.dataset.create_dataset import split_dataset
from src.feature.preprocessing import ekman_map, clean_text
from src.utils import set_seeds

model_name = 'bert-base-uncased'
bertconfig = BertConfig.from_pretrained(model_name, output_hidden_states=False)
tokenizer = BertTokenizerFast.from_pretrained(
    pretrained_model_name_or_path=model_name, config=bertconfig)
transformer_model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name, config=bertconfig)


class Models(ABC):
    """
    Abstract class for a model.
    """

    def __init__(self, params: dict = None):
        self.params = params

    def preprocess(self, df: pd.DataFrame):
        """
        Preprocess the dataframe for model training.
        """
        pass

    def tokenize(self, df: pd.DataFrame):
        """
        Tokenize the dataframe.
        """
        pass

    def fit(self, df: pd.DataFrame):
        """
        Fit the model to the training data.
        """
        pass

    def evaluate(self, X, Y):
        """
        Evaluate the model on the testing data.
        """
        pass


class BERT(Models, ABC):
    """
    Pre-trained BERT model
    """

    def __init__(self, params: dict):
        """
        Initialize the BERT model with parameters.

        Args:
            params: dictionary of parameters
        """
        self.params = params
        self.model = self.build_model()
        self.features = 'text'
        self.labels = ekman_map.keys()

    def tokenize(self, df: pd.DataFrame) -> Tuple[tf.Tensor, np.ndarray]:
        """
        Tokenize the dataframe.

        Args:
            df: dataframe
        Returns:
            tokenized_input (tensor): tokenized data
        """
        X, Y = self.preprocess(df)

        tokenized = tokenizer(
            text=list(X),
            add_special_tokens=True,
            max_length=self.params.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        return tokenized, Y

    def return_tensor(self, tokenized_input: tf.Tensor, labels: np.ndarray, shuffle: int = None) -> BatchDataset:
        """
        Return the tensor for the model.

        Args:
            tokenized_input: tokenized data
            labels: labels
            shuffle: shuffle the data

        Returns:
            tensor: the input tensor

        """
        inputs = {'input_ids': tokenized_input['input_ids'],
                  'attention_mask': tokenized_input['attention_mask'],
                  'token_type_ids': tokenized_input['token_type_ids']}

        if shuffle is not None:
            tensor = tf.data.Dataset.from_tensor_slices(
                (inputs, labels)).shuffle(shuffle).batch(self.params.batch_size).prefetch(1)
        else:
            tensor = tf.data.Dataset.from_tensor_slices(
                (inputs, labels)).batch(self.params.batch_size).prefetch(1)

        return tensor

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataframe for model training.

        Args:
            df: dataframe

        Returns:
            X: features
            Y: labels
        """
        if self.params.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        X = df[self.features].values
        Y = df.loc[:, self.labels].values

        return X, Y

    def build_model(self, n_labels: int = len(ekman_map)) -> Model:
        """
        Build the pre-trained BERT model

        Args:
            n_labels: number of labels
        """

        # main layer
        bert = transformer_model.layers[0]

        # inputs
        input_ids = Input(
            shape=(self.params.max_length,), name='input_ids', dtype='int32')
        attention_mask = Input(
            shape=(self.params.max_length,), name='attention_mask', dtype='int32')
        token_type_ids = Input(
            shape=(self.params.max_length,), name='token_type_ids', dtype='int32')
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'token_type_ids': token_type_ids}

        # layers
        bert_model = bert(inputs)[1]
        dropout = Dropout(bertconfig.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(bert_model, training=False)

        # output
        emotion = Dense(units=n_labels, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=bertconfig.initializer_range),
                        name='output')(pooled_output)
        outputs = emotion

        model = Model(inputs=inputs, outputs=outputs, name='BERT')

        return model

    def fit(self, df: pd.DataFrame, trial: optuna.trial.Trial = None) -> None:
        """
        Fit the model to the training data.
        """
        set_seeds()

        # split data into training and validation sets
        train_df, test_df, val_df = split_dataset(df, test_size=0.1)

        # tokenize train and valid sets
        train_tokenized, y_train = self.tokenize(df=train_df)
        val_tokenized, y_val = self.tokenize(df=val_df)
        test_tokenized, y_test = self.tokenize(df=test_df)

        # convert tokenized data to tensor
        train_tensor = self.return_tensor(
            tokenized_input=train_tokenized, labels=y_train, shuffle=5000)
        val_tensor = self.return_tensor(
            tokenized_input=val_tokenized, labels=y_val)
        test_tensor = self.return_tensor(
            tokenized_input=test_tokenized, labels=y_test)

        model = self.model

        linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.params.initial_learning_rate,
            end_learning_rate=self.params.end_learning_rate,
            decay_steps=self.params.decay_steps,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=linear_decay, clipnorm=0.1)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        model.fit(
            train_tensor, epochs=self.params.epochs, validation_data=val_tensor, verbose=1)

        train_loss = model.evaluate(train_tensor, verbose=0)
        val_loss = model.evaluate(val_tensor, verbose=0)


        if not trial:
            mlflow.log_metrics(
                {'train_loss': train_loss[0], 'val_loss': val_loss[0]})

        if trial:
            trial.report(val_loss[0], step=1)
            if trial.should_prune():
                raise optuna.structs.TrialPruned()

        # Threshold
        best_threshold = 0
        best_f1 = 0

        pred = model.predict(val_tensor)

        for threshold in np.arange(0.10, 0.99, 0.01):
            preds = np.where(pred > threshold, 1, 0)

            f1 = f1_score(y_val, preds, average='weighted', zero_division=0)

            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
            else:
                continue

        logger.info(f'Found best threshold: {best_threshold} with f1 score: {best_f1}')

        y_pred = np.where(pred > best_threshold, 1, 0)

        # evaluate model
        metrics = self.get_metrics(y_val, y_pred, ekman_map)
        logger.info(f'Metrics: {metrics}')
        
        # save model
        model.save(Path(config.MODEL_DIR, 'bert_model.hdf5'))

        return {
            'params': self.params,
            'model': model,
            'metrics': metrics,
        }


    @staticmethod
    def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_labels: list) -> dict:
        """
        Get the metrics for the model.

        Args:
            y_true: true labels
            y_pred: predicted labels
            n_labels (list): list of class labels

        Returns:
            metrics (dict): dictionary of metrics
        """
        metrics = {'overall': {}, 'classes': {}}

        overall_metrics = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        metrics['overall']['precision'] = overall_metrics[0]
        metrics['overall']['recall'] = overall_metrics[1]
        metrics['overall']['f1'] = overall_metrics[2]
        metrics['overall']['num_samples'] = float(len(y_true))

        class_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)
        for i, label in enumerate(n_labels):
            metrics['classes'][label] = {'precision': class_metrics[0][i],
                                         'recall': class_metrics[1][i],
                                         'f1': class_metrics[2][i],
                                         'num_samples': float(len(y_true))
                                         }

        return metrics

    def predict(self, prompt: List[str], threshold: float, model: Model) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict the emotion of the text.

        Args:
            prompt: text
            threshold: threshold for the prediction
            model: trained bert model

        Returns:
            emotion: emotion
        """
        text = [clean_text(text) for text in prompt]

        tokenized = tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=self.params.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        inputs = {'input_ids': tokenized['input_ids'],
                  'attention_mask': tokenized['attention_mask'],
                  'token_type_ids': tokenized['token_type_ids']}

        pred = model.predict(inputs)

        probabilities = pred
        probabilities = pd.DataFrame(probabilities, columns=ekman_map.keys())
        probabilities.index = text
        probabilities.reset_index(inplace=True)

        pred = np.where(pred > threshold, 1, 0)

        pred = pd.DataFrame(pred, columns=ekman_map.keys())
        pred['emotion'] = pred.iloc[:, 1:].idxmax(axis=1)
        pred.drop(columns=ekman_map.keys(), inplace=True)
        pred.index = text
        pred.reset_index(inplace=True)

        return pred, probabilities
