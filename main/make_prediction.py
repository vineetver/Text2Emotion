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
    # initialize BERT model
    bert = BERT()

    model = bert.model
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=['accuracy'])
    model.load_weights('../model/bert_model.hdf5')

    pred, prob = bert.predict(prompt=PROMPT, threshold=THRESHOLD, model=model)

    print(prob)


if __name__ == "__main__":
    main()
