# Disaster Response Pipeline Project

This project is a part of the Udacity Data Scientist Nanodegree Program which aims to apply data engineering skills learned to analyze disaster data from [Figure Eight](https://appen.com/datasets/combined-disaster-response-data/) to build a model for an API that classifies disaster messages. The project creates a machine learning pipeline and a web app where an emergency worker can input a new message and get classification results in several categories.

### Overview of the data:
This dataset contains 26,249 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the USA in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

### File Description:
Here's the file structure of the project:
<pre>
- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- preprocessed.db.sqlite3   # database to save clean data to

- models
|- train_classifier.py
|- clf.pkl  # saved model 

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- README.md
</pre>


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
    
        `python3 process_data.py --message_filename disaster_messages.csv --category_filename disaster_categories.csv --database_filename preprocessed.db.sqlite3`
    
    - To run ML pipeline that trains classifier and saves:
    
        `python3 train_classifier.py --database_filename ../data/preprocessed.db.sqlite3 --model_filename clf.pkl --grid_search_cv`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Go to http://0.0.0.0:3001/

### Results:
Finally, the results will be shown on a web app where you can input a message and get classification results.

Screenshot of the web App:
![Screenshot of Web App](WebApp.png)

An example message input and the output categorizations:
![example1](example1.png)
