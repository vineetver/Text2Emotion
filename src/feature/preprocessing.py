import re
import string

import emoji
import pandas as pd

labels = {
    0 : 'admiration',
    1 : 'amusement',
    2 : 'anger',
    3 : 'annoyance',
    4 : 'approval',
    5 : 'caring',
    6 : 'confusion',
    7 : 'curiosity',
    8 : 'desire',
    9 : 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'grief',
    17: 'joy',
    18: 'love',
    19: 'nervousness',
    20: 'optimism',
    21: 'pride',
    22: 'realization',
    23: 'relief',
    24: 'remorse',
    25: 'sadness',
    26: 'surprise',
    27: 'neutral'
}

ekman_map = {
    'anger'   : ['anger', 'annoyance', 'disapproval'],
    'disgust' : ['disgust'],
    'fear'    : ['fear', 'nervousness'],
    'joy'     : ['joy', 'amusement', 'approval', 'excitement', 'gratitude', 'love', 'optimism', 'relief', 'pride',
                 'admiration', 'desire', 'caring'],
    'sadness' : ['sadness', 'disappointment', 'embarrassment', 'grief', 'remorse'],
    'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
    'neutral' : ['neutral']
}

sentiment_map = {
    'positive' : ['amusement', 'excitement', 'joy', 'love', 'desire', 'optimism', 'caring', 'pride', 'admiration',
                  'gratitude', 'relief', 'approval'],
    'negative' : ['fear', 'nervousness', 'remorse', 'embarrassment', 'disappointment', 'sadness', 'grief', 'disgust',
                  'anger', 'annoyance', 'disapproval'],
    'ambiguous': ['realization', 'surprise', 'curiosity', 'confusion', 'neutral']
}


def drop_annotator_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function drops the annotator column from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        dataset

    Returns
    -------
    df: pd.DataFrame
        dataset without annotator column
    """
    df = df.drop(columns=['annotator'], axis=1)

    return df


def str_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts a string of class indices to a list of class indices
    e.g. 0 -> [0]
         0, 26 -> [0, 26]

    Parameters
    ----------
    df: pd.DataFrame
        dataset

    Returns
    -------
    df : pd.DataFrame
        dataset with class indices as list
    """

    df['list_of_emotions'] = df['emotion'].apply(lambda x: x.split(','))

    return df


def index_to_class(index: list) -> list:
    """
    This function converts a list indices to a list of class labels
    e.g. [0] -> ['admiration']
         [0, 1] -> ['admiration', 'amusement']

    Parameters
    ----------
    index : list
        List of class indices

    Returns
    -------
    class_labels : list
        List of class labels
    """
    class_labels = []
    for i in index:
        class_labels.append(labels[int(i)])

    return class_labels


def emotion_mapping(class_labels: list) -> list:
    """
    This function maps the class labels
    to the corresponding ekman class label to reduce the number of classes.

    e.g. ['anger', 'annoyance', 'disapproval'] -> 'anger'


    Parameters
    ----------
    class_labels : str
        Class label

    Returns
    -------
    ekman_class_label : list
        List of ekman class labels
    """
    ekman_class_label = []
    for label in class_labels:
        if label in ekman_map['anger']:
            ekman_class_label.append('anger')
        if label in ekman_map['disgust']:
            ekman_class_label.append('disgust')
        if label in ekman_map['fear']:
            ekman_class_label.append('fear')
        if label in ekman_map['joy']:
            ekman_class_label.append('joy')
        if label in ekman_map['sadness']:
            ekman_class_label.append('sadness')
        if label in ekman_map['surprise']:
            ekman_class_label.append('surprise')
        if label == 'neutral':
            ekman_class_label.append('neutral')

    return ekman_class_label


def sentiment_mapping(class_labels: list) -> list:
    """
    This function maps the class labels
    to the corresponding sentiment class label to reduce the number of classes.

    e.g. ['fear', 'nervousness', 'remorse', 'embarrassment', 'disappointment', 'sadness', 'grief', 'disgust',
                  'anger', 'annoyance', 'disapproval'] -> 'negative'

    Parameters
    ----------
    class_labels : list
        Class label

    Returns
    -------
    sentiment_class_label : list
        List of sentiment class labels
    """
    sentiment_class_label = []
    for label in class_labels:
        if label in sentiment_map['positive']:
            sentiment_class_label.append('positive')
        if label in sentiment_map['negative']:
            sentiment_class_label.append('negative')
        if label in sentiment_map['ambiguous']:
            sentiment_class_label.append('ambiguous')

    return sentiment_class_label


def clean_text(text: str) -> str:
    """
    This function cleans the text by removing the punctuation, emojis, and special characters.

    Parameters
    ----------
    text : str
        Text to be cleaned

    Returns
    -------
    text : str
        text without punctuation and special characters
    """

    text = emoji.demojize(text)  # remove emojis
    text = str(text).lower()  # text to lower case
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation

    return text

