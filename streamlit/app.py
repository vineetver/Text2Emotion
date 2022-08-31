from pathlib import Path

import pandas as pd

import streamlit as st
from config import config
from main.main import predict_emotion
from src.utils import get_dict


st.title('Text2Emotion - Streamlit App')

# Sections

st.header('Data')
st.write('Data is loaded from `data/preprocessed.csv`')
df = pd.read_csv(config.DATA_DIR, 'preprocessed.csv')
st.write(df)


# Performance
st.header('Performance')
st.write('Performance is measured by running `main.py`')
performance = get_dict(config.CONFIG_DIR, 'performance.json')
st.text('Overall Model Performance')
st.write(performance.overall)
tag = st.selectbox('Select a class: ', list(performance.classes.keys()))
st.write(performance.classes[tag])

st.header('🚀 Inference')
text = st.text_input('Enter a PROMPT: ', 'I am feeling happy today.')
run_id = st.text_input('Enter a RUN ID: ', open(Path(config.CONFIG_DIR, 'run_id.txt')).read())
pred, probs = predict_emotion(text, run_id)

st.write(f'Prediction: {pred}')
st.write(f'Probabilities: {probs}')
