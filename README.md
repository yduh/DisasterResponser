# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
    
        `python3 process_data.py --message_filename disaster_messages.csv --category_filename disaster_categories.csv --database_filename preprocessed.db.sqlite3`
    
    - To run ML pipeline that trains classifier and saves:
    
        `python3 train_classifier.py --database_filename ../data/preprocessed.db.sqlite3 --model_filename clf.pkl --grid_search_cv`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
