import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals 
from sqlalchemy import create_engine
import joblib

import sys, re

import nltk
#from nltk.tokenize import word_tokenize, RegexpTokenizer
#from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

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

DATABASE_FILENAME = 'preprocessed.db.sqlite3'
TABLE_NAME = 'data_preprocessed'
MODEL_FILENAME = 'clf.pkl'

# load data
engine = create_engine('sqlite:///../data/'+DATABASE_FILENAME)
df = pd.read_sql_table(TABLE_NAME, engine)

# load model
model = joblib.load("../models/"+MODEL_FILENAME)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
   
    cat_counts_sorted =  df.iloc[:,4:].sum().sort_values(ascending=False)
    cat_names = list(cat_counts_sorted.index)
    cat_counts = list(cat_counts_sorted)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # graph 1:    
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # graph 2:
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'margin':{
                    'b':200
                },
                'automargin':True
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
