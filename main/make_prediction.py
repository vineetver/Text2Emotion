from keras.models import load_model
import tensorflow as tf

THRESHOLD = 0.83
PROMPT = [
    'A Ukrainian woman who escaped Russias assault on Mariupol says troops were targeting apartment buildings as if they were playing a computer game',
    'I often go to parks to walk and destress and enjoy nature',
    'How can this be',
    'This is the worst muffin ive ever had']


def main():
    bert = tf.keras.models.load_model('../model/bert_model.hdf5')

    pred, prob = bert.predict(prompt=PROMPT, threshold=THRESHOLD, model=bert)

    print(f'Prediction: {pred}')
    print(f'Probabilities: {prob}')


if __name__ == "__main__":
    main()
