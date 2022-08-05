import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, f1_score

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
    bert = BERT(features=features, labels=labels, params={'max_length': 33, 'batch_size': 32})

    # tokenize test set
    test_tokenized, y_test = bert.tokenize(df=test_df)

    # convert tokenized data to tensor
    test_tensor = bert.return_tensor(tokenized_input=test_tokenized, labels=y_test)

    # optimizer and loss function
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-5,
        decay_rate=0.004,
        decay_steps=340,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, epsilon=1e-08)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # load model
    model = bert.build_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.load_weights('../model/bert_model.hdf5')

    # evaluate model
    y_pred = model.predict(test_tensor)

    probabilities = y_pred

    probabilities = pd.DataFrame(probabilities, columns=ekman_map.keys())
    probabilities.index = test_df['text']
    probabilities.reset_index(inplace=True)
    probabilities.to_csv('../data/probabilities.csv', sep='\t', encoding='utf-8', index=False)

    y_pred = np.where(y_pred > 0.8, 1, 0)

    f1 = []
    precision = []
    emotions = ekman_map.keys()

    for i in range(len(emotions)):
        f1.append(f1_score(y_test[:, i], y_pred[:, i], average='macro'))
        precision.append(precision_score(y_test[:, i], y_pred[:, i], average='macro'))

    results = pd.DataFrame({'precision': precision, 'f1': f1})
    results.index = emotions

    results.to_csv('../data/results.csv', sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
