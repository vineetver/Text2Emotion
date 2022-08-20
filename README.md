<h2 align="center"> Text2Emotion </h2>


## Description
Understanding emotions expressed in text is an essential task for many applications. Fine-grained emotion detection refers to identifying which emotions are represented by a text and the degree or intensity to which the emotion is expressed. This is a challenging task, as emotions are often expressed indirectly through metaphors or other figurative languages. The task is to build a web application that uses **natural language processing** to predict the emotion of a text. This is a multi-label classification task. The labels are 7 emotions: anger, disgust, fear, joy,
sadness, surprise, and neutral. To evaluate the performance of the model F1 score is used.

The current best model is a `fine-tuned BERT` with `tf.keras.optimizers.Adam` optimizer, `learning_rate = 5e-5`,
and `PolynomialDecay` scheduler. The evaluation f1 score is 66%. The baseline published by the Goemotion team is evaluated at 65% f1.

## Model Architecture

The model used is a pre-trained BERT. BERT is a Natural Language Processing (NLP) model developed by Google based on Transformer block architectures. It is one of the most recent advancements in NLP and is used for tasks such as classification, question answering, and entity recognition. BERT's architecture comprises an encoder and a decoder and introduces a new type of cognitive attention called self-attention.   

To fine-tune a pre-trained BERT model, the last layer of the BERT model is removed and replaced with a custom classification layer. The activation function used for the output layer is sigmoid, which outputs the probability for each emotion.

![model_png](https://github.com/vineetver/Text2Emotion/blob/main/model.png)

## Hyperparameter Choice  

The choice of hyperparameters is critical in training any machine learning model. I have tried various configurations, and this is the configuration that resulted in a 1% increase from the best model provided by the GoEmotions team, with a 65% f1-score on the test set `tf.keras.optimizers.Adam` optimizer, `learning_rate = 5e-5`, `batch_size = 64`, `dropout = 0.1 to 0.25` and `PolynomialDecay` scheduler. 

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
- [x] Pipeline
- [ ] Web application
- [ ] Deployment

## License

Distributed under the MIT License. See `LICENSE.md` for more information.

## Contact

Vineet Verma - vineetver@hotmail.com - [Goodbyeweekend.io](https://www.goodbyeweekend.io/)

