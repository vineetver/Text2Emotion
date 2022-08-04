import os

import pandas as pd


def read_dataset(train_url: str, test_url: str, valid_url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function reads the training/testing/validation dataset from the url
    and returns a pandas dataframe.

    Parameters
    ----------
    train_url : str
        url of the training dataset
    test_url : str
        url of the testing dataset
    valid_url : str
        url of the validation dataset

    Returns
    -------
    train_df : pd.DataFrame
        pandas dataframe of the training dataset
    test_df : pd.DataFrame
        pandas dataframe of the testing dataset
    valid_df : pd.DataFrame
        pandas dataframe of the validation dataset
    """

    train_df = pd.read_csv(train_url, sep='\t',
                           encoding='utf-8',
                           names=['text', 'emotion', 'annotator'],
                           header=None)
    test_df = pd.read_csv(test_url, sep='\t',
                          encoding='utf-8',
                          names=['text', 'emotion', 'annotator'],
                          header=None)
    valid_df = pd.read_csv(valid_url, sep='\t',
                           encoding='utf-8',
                           names=['text', 'emotion', 'annotator'],
                           header=None)

    return train_df, test_df, valid_df


def write_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    """
    This function writes the training/testing/validation dataset to the url
    and returns a pandas dataframe.

    Parameters
    ----------
    train_df : pd.DataFrame
        pandas dataframe of the training dataset
    test_df : pd.DataFrame
        pandas dataframe of the testing dataset
    valid_df : pd.DataFrame
        pandas dataframe of the validation dataset
    train_url : str
        url of the training dataset
    test_url : str
        url of the testing dataset
    valid_url : str
        url of the validation dataset

    Returns
    -------
    None
    """
    path = '../data/'

    train_path = os.path.join(path, 'train.csv')
    test_path = os.path.join(path, 'test.csv')
    valid_path = os.path.join(path, 'valid.csv')

    train_df.to_csv(train_path, sep='\t', encoding='utf-8', index=False, header=False)
    test_df.to_csv(test_path, sep='\t', encoding='utf-8', index=False, header=False)
    valid_df.to_csv(valid_path, sep='\t', encoding='utf-8', index=False, header=False)

    return None