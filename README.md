## About

I built a web application that uses natural language processing to identify emotions of a given text. The application is able to identify 7 emotions according to the ekman emotion map.

The model is trained on GoEmotions. GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human annotations to 27 emotion categories or Neutral.

Number of examples: 58,009.
Number of labels: 27 + Neutral.
Maximum sequence length in training and evaluation datasets: 30.
On top of the raw data, they also include a version filtered based on reter-agreement, which contains a train/test/validation split:

Size of training dataset: 43,410.
Size of test dataset: 5,427.
Size of validation dataset: 5,426.
The emotion categories are: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise.

For more details about [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)  



## Getting Started

### Dependancies

List of all the libraries you need to run the code.

  ```sh
nltk
emoji
tensorflow
transformers
pandas
sklearn
matplotlib
seaborn
  ```


<!-- USAGE EXAMPLES -->
## Usage

  ```sh
  $ conda create -n "env-name" python=3.x, anaconda, nltk, emoji, tensorflow, transformers
 
  $ conda activate "env-name"
  
  $ cd Emotion
  
  $ jupyter notebook
  ```
  
## View code

[NBViewer](https://nbviewer.org/github/vineetver/Emotion/blob/main/Textual_emotion_detection.ipynb)

## Roadmap

- [x] Preprocessing
- [x] Visualization
- [x] Model Creation
- [x] Model Training
- [x] Evaluation
- [x] Hyperparameter tuning
- [x] Model Training
- [x] Evaluation
- [ ] Web application


## License

Distributed under the MIT License. See `LICENSE.md` for more information.


## Contact

Vineet Verma - vineetver@hotmail.com - [Goodbyeweekend.io](https://www.goodbyeweekend.io/)
