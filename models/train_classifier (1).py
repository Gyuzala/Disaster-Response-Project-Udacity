import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
from sqlalchemy import create_engine
import os
import matplotlib as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import multioutput
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    #drop child_alone column, as it has only null values
    df = df.drop(['child_alone'],axis=1)
    
    df = df[df.related !=2]
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    return X,Y 


def tokenize(text):
    """Tokenization function. Receives as input raw text which afterwards normalized, stop words removed, stemmed and lemmatized.
    Returns tokenized text"""
    
    #Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize
    words = word_tokenize(text)
    #stop words removal
    words = [w for w in words if w not in stopwords.words('english')]
    #lemmatizing
    lemmed = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in words]
    
    return lemmed

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    pipeline = Pipeline ([("vect", CountVectorizer(tokenizer=tokenize)),
                          ("tfidf", TfidfTransformer()),
                         ("clf", multioutput.MultiOutputClassifier (RandomForestClassifier()))])
    parameters = { 'vect__max_df': (0.75, 1.0),
                'clf__estimator__n_estimators': [10, 20],
                'clf__estimator__min_samples_split': [2, 5]
              }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv 

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    This function applies a ML pipeline to a test set and prints out the model performance
    """
    y_pred = model.predict(X_test)
    # Print classification report on test data
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))


def save_model(model, model_filepath):
    """
    Save Pipeline function
    This function saves trained model as Pickle file, to be loaded later.   
    """
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()