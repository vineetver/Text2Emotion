from abc import ABC
from typing import Any

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from tensorflow.python.ops.gen_dataset_ops import BatchDataset
from transformers import TFBertModel, BertTokenizerFast, BertConfig


model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, output_hidden_states=False)
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)
transformer_model = TFBertModel.from_pretrained(model_name, config=config)


class Models(ABC):
    """
    Abstract class for a model.
    """

    def __init__(self, features: list[str] = None, labels: list[str] = None, params: dict = None):
        self.features = features
        self.labels = labels
        self.params = params
        self.model = None

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

    def split(self, df: pd.DataFrame):
        """
        Split the dataframe into training, testing and validation sets.
        """
        pass

    def fit(self, X, Y):
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

    def __init__(self, features: str = None, labels: list[str] = None, params: dict = None):
        """
        Initialize the BERT model with the features, labels and parameters.

        Args:
            features: list of features
            labels: list of labels
            params: dictionary of parameters
        """
        super().__init__(features, labels, params)

    def tokenize(self, df: pd.DataFrame) -> tuple[tf.Tensor, np.ndarray]:
        """
        Tokenize the dataframe.

        Args:
            df: dataframe

        """
        X, Y = self.preprocess(df)

        tokenized = tokenizer(
            text=list(X),
            add_special_tokens=True,
            max_length=self.params['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        return tokenized, Y

    def return_tensor(self, tokenized_input: tf.Tensor, labels: np.ndarray) -> BatchDataset:
        """
        Return the tensor for the model.

        Args:
            tokenized_input: tokenized data
            labels: labels

        Returns:
            tensor: the input tensor
        """
        inputs = {'input_ids'     : tokenized_input['input_ids'],
                  'attention_mask': tokenized_input['attention_mask'],
                  'token_type_ids': tokenized_input['token_type_ids']}

        tensor = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.params['batch_size'])

        return tensor

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataframe for model training.

        Args:
            df: dataframe

        Returns:
            X: features
            Y: labels
        """
        X = df[self.features].values
        Y = df.loc[:, self.labels].values

        return X, Y
