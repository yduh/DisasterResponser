# Access the proprocessed data, apply NLP process and feed in ML model, return results in a pickle file
# Usage: python3 train_classifier.py --database_filename ../data/preprocessed.db.sqlite3 --model_filename clf.pkl --grid_search_cv

import sys, re
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from joblib import dump, load

import nltk
#from nltk.tokenize import word_tokenize, RegexpTokenizer
#from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

DATABASE_FILENAME = 'preprocessed.db.sqlite3'
TABLE_NAME = 'data_preprocessed'
MODEL_FILENAME = 'clf.pkl'

def load_data(database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df = pd.read_sql_table(TABLE_NAME, engine)

    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return (X, y, category_names)


def tokenize(text):
    # replace all non-alphabets and non-numbers with blank space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    tokens = word_tokenize(text)

    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # instantiate stemmer
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        # lemmtize token using noun as part of speech
        clean_tok = lemmatizer.lemmatize(tok)
        # lemmtize token using verb as part of speech
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        # stem token
        clean_tok = stemmer.stem(clean_tok)
        # strip whitespace and append clean token to array
        clean_tokens.append(clean_tok.strip())

    #url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #detected_urls = re.findall(url_regex, text)
    #for url in detected_urls:
    #    text = text.replace(url, 'urlplaceholder')

    #tokens = [word for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    #tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    #lemmatizer = WordNetLemmatizer()
    #cleaned_tokens = []
    #for tok in tokens:
    #    clean_tok = lemmatizer.lemmatize(tok).lower().strip()
    #    cleaned_tokens.append(clean_tok)

    return clean_tokens    

def build_model(grid_search_cv = False):
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #pipeline.get_params()
    if grid_search_cv:
        print('Running GridSearchCV...')
        parameters = {
            #'vect__ngram_range': ((1, 2), (1, 3)), #((1, 1), (1, 2)),
            #'vect__max_df': (0.4, 0.5, 0.6), #(0.5, 0.75, 1.0),
            'clf__estimator__n_estimators': [50, 100, 200],
            #'clf__estimator__min_samples_split': [2, 3, 4],
        }

        pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)

    # Calculate the accuracy for each of them.	
    for i in range(len(category_names)):   
        print('Category: {} '.format(category_names[i]))
        print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])))
        #print(classification_report(y_test.iloc[:, 1:].values, np.array([x[1:] for x in y_pred]), target_names = category_names))
 

def save_model(model, model_filename):
    dump(model, model_filename)


def parse_input_argument():
    parser = argparse.ArgumentParser(description = 'Disaster Responser Train Classifier')
    parser.add_argument('--database_filename', type=str, default='DATABASE_FILENAME', help='Filename of the preprocessed data (input)')
    parser.add_argument('--model_filename', type=str, default='MODEL_FILENAME', help='Pickle filename for trained classifier model (output)')
    parser.add_argument('--grid_search_cv', action='store_true', default=False, help='Run grid search CV for the parameters')
    args = parser.parse_args()

    return (args.database_filename, args.model_filename, args.grid_search_cv)

    
def runTraining(database_filename, model_filename, grid_search_cv=False):
    print('Loading data...\n    DATABASE: {}'.format(database_filename))
    X, y, category_names = load_data(database_filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print('Download needed NLTK libraries...')
    nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
    
    print('Building model...')
    model = build_model(grid_search_cv)
        
    print('Training model...')
    model.fit(X_train, y_train)
    if grid_search_cv:
        print('Best Parameters:', model.best_params_)
        
    print('Evaluating model...')
    evaluate_model(model, X_test, y_test, category_names)
    
    print('Saving model...\n    MODEL: {}'.format(model_filename))
    save_model(model, model_filename)
    print('Trained model saved!')


if __name__ == '__main__':
    database_filename, model_filename, grid_search_cv = parse_input_argument()
    runTraining(database_filename, model_filename, grid_search_cv)

