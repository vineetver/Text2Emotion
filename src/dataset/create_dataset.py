import pandas as pd


def read_dataset(url: str) -> pd.DataFrame:
    """
    This function reads the training/testing/validation dataset from the url
    and returns a pandas dataframe.

    Parameters
    ----------
    url : str
        url of the dataset

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe of the dataset
    """
    df = pd.read_csv(url, sep='\t',
                     encoding='utf-8',
                     names=['text', 'emotion', 'annotator'],
                     header=None)
    return df
