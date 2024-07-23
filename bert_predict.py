import gradio as gr
# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI
from bertopic import BERTopic
import pandas as pd

app = FastAPI(
    title="Bertopic model ",
    description="A simple API that use NLP model to predict topics",
    version="0.1",
)

# Load the BERTopic model
with open("Bertopic_model.pkl", "rb") as f:
    model = joblib.load(f)

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    
    return text

@app.get("/predict-review")
def predict_topics(review: str):
    cleaned_review = text_cleaning(review)
    num_of_topics = 3
    similar_topics, similarity = model.find_topics(cleaned_review, top_n=num_of_topics)
    topics_name = pd.read_excel("topic_list.xlsx")
    topic_dict = topics_name.set_index("Topic")["Name"].to_dict()
    
    return {
        f"Topic {i+1}": f'Topic Number: {str(similar_topics[i])} Associated Words:{topic_dict[similar_topics[i]]}'
        for i in range(num_of_topics)
    }

# Wrap the FastAPI route with Gradio
gr.Interface(predict_topics, inputs="text", outputs="text").launch(share=True)
