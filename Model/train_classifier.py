#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd 
import numpy
from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
# from sklearn.model_selection import GridSearchCV
import sklearn
import joblib

import nltk
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load database and get dataset
    """
    # load data from database 
    database_filepath = "sqlite:///" + database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('DisasterResponseTable', engine)
    X=df['message']
    Y=df.iloc[ : , 4 : ] 
    return X,Y.astype(int)

def tokenize(text):
    """
    define a function to remove numbers, remove white space, remove stop words, normalize words
    """
    result = text.lower() #Convert text to lowercase
    result = re.sub(r'\d+', '', result) #Remove numbers
    result= "".join([char for char in result if char not in string.punctuation]) #Removing Punctuation
    result = result.strip() #remove white space

    stop_words = set(stopwords.words('english')) 
    match_tokenizer = RegexpTokenizer("[\w']+")
    match_tokens = match_tokenizer.tokenize(result)
    result = [i for i in match_tokens if not i in stop_words]  #Stop words removal

    lemmatizer=WordNetLemmatizer()    #reduce inflectional forms to a common base form.
    Lemmatization = [lemmatizer.lemmatize(word) for word in result]
    return Lemmatization

def build_model():
    """
    Create and return a ML pipeline to train categorizations of text
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    # cv = sklearn.grid_search.GridSearchCV((pipeline, parameters))
    return cv

def evaluate_model(trained_model, X_test, Y_test):
    """
    Print a classification report for each of the outputs of a multioutput classification model
    """
    Y_pred = trained_model.predict(X_test)
    for i, col in enumerate(Y_test.columns): 
        print("category: ", col) 
        print(classification_report(Y_test.values[i], Y_pred[i]))

def save_model(model, model_filepath):
    """
    Save the trained model, `model` to the specified path, `model_filepath`.
    """
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        # X, Y, categories = load_data(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '             'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# // python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
