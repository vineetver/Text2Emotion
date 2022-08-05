import pandas as pd
from src.model.classifier import BERT
from src.dataset.create_dataset import split_dataset
from src.feature.preprocessing import ekman_map


def main():
    # Load data
    df = pd.read_csv('../data/preprocessed.csv', sep='\t',
                     encoding='utf-8')

    # features and labels
    features = 'text'
    labels = ekman_map.keys()

    # split data into training, testing and validation sets
    train_df, test_df, val_df = split_dataset(df, test_size=0.1)

    bert = BERT(features=features, labels=labels, params={'max_length': 33, 'batch_size': 80})

    # tokenize train, test and valid sets
    train_tokenized, y_train = bert.tokenize(df=train_df)
    test_tokenized, y_test = bert.tokenize(df=test_df)
    val_tokenized, y_val = bert.tokenize(df=val_df)

    # convert tokenized data to tensor
    train_tensor = bert.return_tensor(tokenized_input=train_tokenized, labels=y_train)
    test_tensor = bert.return_tensor(tokenized_input=test_tokenized, labels=y_test)
    val_tensor = bert.return_tensor(tokenized_input=val_tokenized, labels=y_val)




if __name__ == "__main__":
    main()
