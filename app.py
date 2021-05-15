from logging import debug
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

#removing stop words
def remove_stopwords(review_list):
  stop = stopwords.words('english')
  removed = []
  for review in review_list:
    removed.append(' '.join([word for word in review.split() if word not in stop]))
  return removed

#lemmatizing text
def get_lemmatized(review_list):
  lemmatizer = WordNetLemmatizer()
  return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in review_list]

@app.route("/submit", methods=['POST'])
def submit():
    #Alternative Usage of Saved Model
    ngram_model = open('ngram_vectorizer','rb')
    clf = joblib.load(ngram_model)
    
    model = open('sentiment_analyser','rb')
    clf_model = joblib.load(model)

    if request.method == 'POST':
        message = request.form['message']
        review = [message]
        review = remove_stopwords(review)
        review = get_lemmatized(review)
        my_review = np.array(review)
        my_test_review = clf.transform(my_review)
        my_prediction = clf_model.predict(my_test_review)
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
