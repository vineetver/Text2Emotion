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
