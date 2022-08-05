<h2 align="center"> Text2Emotion </h2>

## Description

The task is to build a web application that uses **natural language processing** to predict the emotion
of a text. This is a multi-label classification task. The labels are 7 emotions: anger, disgust, fear, joy,
sadness, surprise, and neutral. To evaluate the performance of the model F1 score is used.

The current best model is a `fine-tuned BERT` with `tf.keras.optimizers.Adam` optimizer, `learning_rate = 5e-5`,
and `ExponentialDecay` scheduler. The evaluation precision score is 82%. Choose between precision and recall for your usecase. 

## Model Architecture

![model_png](https://github.com/vineetver/Text2Emotion/blob/main/model.png)

## About the Data

The model is trained on GoEmotions. GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with
human annotations to 27 emotion categories or Neutral.

- Number of examples: 58,009.
- Number of labels: 27 + Neutral.

Maximum sequence length in training and evaluation datasets: 30.
On top of the raw data, they also include a version filtered based on reter-agreement, which contains a
train/test/validation split:

- Size of training dataset: 43,410.
- Size of test dataset: 5,427.
- Size of validation dataset: 5,426.

The emotion categories are: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire,
disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness,
optimism, pride, realization, relief, remorse, sadness, surprise.

For more details about [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)


## Dependencies

    $ pip install -r requirements.txt



[//]: # (## Running the pipeline)

[//]: # ()
[//]: # (    $ git clone repo.git)

[//]: # (    $ cd repo)


## Running the tests

    py.test tests

## Roadmap

- [x] Prototype
- [ ] Pipeline
- [ ] Web application
- [ ] Deployment

## License

Distributed under the MIT License. See `LICENSE.md` for more information.


## Contact

Vineet Verma - vineetver@hotmail.com - [Goodbyeweekend.io](https://www.goodbyeweekend.io/)

