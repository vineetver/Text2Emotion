import pandas as pd
from keras.models import load_model

from src.dataset.create_dataset import split_dataset
from src.feature.preprocessing import ekman_map
from src.model.classifier import BERT


def main():
    # Load data
    df = pd.read_csv('../data/preprocessed.csv', sep='\t', encoding='utf-8')

    # features and labels
    features = 'text'
    labels = ekman_map.keys()

    # split data into training, testing and validation sets
    train_df, test_df, val_df = split_dataset(df, test_size=0.1)

    # initialize BERT model
    bert = BERT(features=features, labels=labels, params={'max_length': 33, 'batch_size': 80})

    # tokenize test set
    test_tokenized, y_test = bert.tokenize(df=test_df)

    # convert tokenized data to tensor
    test_tensor = bert.return_tensor(tokenized_input=test_tokenized, labels=y_test)

    # load model
    model = load_model('../model/bert_model.h5')

    # evaluate model
    model.evaluate(test_tensor)


if __name__ == "__main__":
    main()
