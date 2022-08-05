import pandas as pd
import tensorflow as tf

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

    bert = BERT(features=features, labels=labels, params={'max_length': 33, 'batch_size': 16})

    # tokenize train, test and valid sets
    train_tokenized, y_train = bert.tokenize(df=train_df)
    test_tokenized, y_test = bert.tokenize(df=test_df)
    val_tokenized, y_val = bert.tokenize(df=val_df)

    # convert tokenized data to tensor
    train_tensor = bert.return_tensor(tokenized_input=train_tokenized, labels=y_train, shuffle=5000)
    test_tensor = bert.return_tensor(tokenized_input=test_tokenized, labels=y_test)
    val_tensor = bert.return_tensor(tokenized_input=val_tokenized, labels=y_val)

    # build model
    model = bert.build_model()

    # optimizer and loss function
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-5,
        decay_rate=0.7,
        decay_steps=340,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.002)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # train model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(train_tensor, epochs=3, validation_data=val_tensor)

    # save model
    model.save_weights('../model/bert_model.hdf5')


if __name__ == "__main__":
    main()
