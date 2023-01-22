# Sentiment_Analysis.py
from textblob import TextBlob

def get_sentiment(text):
    # create a TextBlob object
    blob = TextBlob(text)

    # get the sentiment
    sentiment = blob.sentiment.polarity

    return sentiment

def get_subjectivity(text):
    # create a TextBlob object
    blob = TextBlob(text)

    # get the subjectivity
    subjectivity = blob.sentiment.subjectivity

    return subjectivity