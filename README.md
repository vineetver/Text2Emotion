<h2 align="center"> Text2Emotion </h2>


## Description
Understanding emotions expressed in text is an essential task for many applications. Fine-grained emotion detection refers to identifying which emotions are represented by a text and the degree or intensity to which the emotion is expressed. This is a challenging task, as emotions are often expressed indirectly through metaphors or other figurative languages. The task is to build a web application that uses **natural language processing** to predict the emotion of a text. This is a multi-label classification task. The labels are 7 emotions: anger, disgust, fear, joy,
sadness, surprise, and neutral. To evaluate the performance of the model F1 score is used.

The current best model is a `fine-tuned BERT` with `tf.keras.optimizers.Adam` optimizer, `learning_rate = 5e-5`,
and `PolynomialDecay` scheduler. The evaluation f1 score is 66%. The baseline published by the Goemotion team is evaluated at 65% f1.

## Hard Requirements

    Python>=3.7
    CUDA=11.x
    CUDA Enabled GPU
    

## Setting up the environment

Before we can run the pipline, PYTHONPATH must be set for the config directory.

```bash
git clone https://github.com/vineetver/Text2Emotion.git
cd Text2Emotion
export PYTHONPATH=$PYTHONPATH:${PWD}/config
```
Setting up the virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Installing cudnn 8.1.x, cudnn needs to be installed separately before we can install tensorflow.

For Ubuntu 20.04

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo apt-get install -f  # resolve dependency errors you saw earlier
```

```bash
pip install -r requirements.txt
```
## Workflow

Documentation for the main.py is available by running the following command.

```bash
python main/main.py --help
```

```bash
python main/main.py etl-data
python main/main.py train-model --params_file='config/parammeters.json' --experiment_name='fine-tuned-bert' --run_name='PolynomailDecay'
python main/main.py optimize --params_file='config/parammeters.json' --experiment_name='optimization' --n_trials=10
python main/main.py predict-emotion --prompt='I am very happy'
```

## API

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8080 --reload --reload-dir src --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # production
```

## Hyperparameter Choice

The choice of hyperparameters is critical in training any machine learning model. I have tried various configurations, and this is the configuration that resulted in a 1% increase from the best model provided by the GoEmotions team, with a 66% f1-score on the test set `tf.keras.optimizers.Adam` optimizer, `learning_rate = 5e-5`, `batch_size = 64`, `dropout = 0.1 to 0.25` and `PolynomialDecay` scheduler.

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
