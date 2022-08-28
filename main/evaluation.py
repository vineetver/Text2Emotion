from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, f1_score

from config import config
from src.dataset.create_dataset import split_dataset
from src.feature.preprocessing import ekman_map
from src.model.classifier import BERT


def main():
    # Load data
    df = pd.read_csv(Path(config.DATA_DIR, 'preprocessed.csv'), sep='\t', encoding='utf-8')

    train_df, test_df, val_df = split_dataset(df, test_size=0.1)

    # initialize BERT model
    bert = BERT(params={'max_length': 33, 'batch_size': 64})

    # tokenize test set
    test_tokenized, y_test = bert.tokenize(df=test_df)

    # convert tokenized data to tensor
    test_tensor = bert.return_tensor(tokenized_input=test_tokenized, labels=y_test)

    # load model
    model = bert.model
    model.load_weights('../model/bert_model.hdf5')

    # evaluate model
    y_pred = model.predict(test_tensor)
    y_pred = np.where(y_pred > 0.41, 1, 0)

    f1 = []
    precision = []
    emotions = ekman_map.keys()
    for i in range(len(emotions)):
        f1.append(f1_score(y_test[:, i], y_pred[:, i], average='macro'))
        precision.append(precision_score(y_test[:, i], y_pred[:, i], average='macro'))
    results = pd.DataFrame({'precision': precision, 'f1': f1})
    results.index = emotions

    print(results.mean())


if __name__ == "__main__":
    main()
