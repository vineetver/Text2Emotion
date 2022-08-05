import tensorflow as tf

from src.feature.preprocessing import ekman_map
from src.model.classifier import BERT

THRESHOLD = 0.8
PROMPT = [
    'A Ukrainian woman who escaped Russias assault on Mariupol says troops were targeting apartment buildings as if they were playing a computer game',
    'I often go to parks to walk and destress and enjoy nature',
    'How can this be',
    'This is the worst muffin ive ever had']


def main():
    # features and labels
    features = 'text'
    labels = ekman_map.keys()

    # initialize BERT model
    bert = BERT(features=features, labels=labels, params={'max_length': 33, 'batch_size': 32})

    # optimizer and loss function
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-5,
        decay_rate=0.004,
        decay_steps=340,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, epsilon=1e-08)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model = bert.build_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.load_weights('../model/bert_model.hdf5')

    pred, prob = bert.predict(prompt=PROMPT, threshold=THRESHOLD, model=model)

    print(prob)

if __name__ == "__main__":
    main()
