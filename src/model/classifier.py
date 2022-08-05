from abc import ABC

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.initializers.initializers_v2 import TruncatedNormal
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.python.ops.gen_dataset_ops import BatchDataset
from transformers import TFBertModel, BertTokenizerFast, BertConfig

from src.feature.preprocessing import ekman_map, clean_text

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
        inputs = {'input_ids'     : tokenized_input['input_ids'],
                  'attention_mask': tokenized_input['attention_mask'],
                  'token_type_ids': tokenized_input['token_type_ids']}

        if shuffle is not None:
            tensor = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.params['batch_size']).shuffle(shuffle).prefetch(1)
        else:
            tensor = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.params['batch_size']).prefetch(1)

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

    def build_model(self, n_labels: int = len(ekman_map)) -> Model:
        """
        Build the pre-trained BERT model

        """

        # main layer
        bert = transformer_model.layers[0]

        # inputs
        input_ids = Input(shape=(self.params['max_length'],), name='input_ids', dtype='int32')
        attention_mask = Input(shape=(self.params['max_length'],), name='attention_mask', dtype='int32')
        token_type_ids = Input(shape=(self.params['max_length'],), name='token_type_ids', dtype='int32')
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

        # layers
        bert_model = bert(inputs)[1]
        dropout = Dropout(0.3, name='pooled_output')
        pooled_output = dropout(bert_model, training=False)

        # output
        emotion = Dense(units=n_labels, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                        name='output')(pooled_output)
        outputs = emotion

        model = Model(inputs=inputs, outputs=outputs, name='BERT')

        return model

    def predict(self, prompt: list[str], threshold: float, model: Model) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            max_length=self.params['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        inputs = {'input_ids'     : tokenized['input_ids'],
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
