# Preprocess the data and save it in a database
# Usage: python3 process_data.py --message_filename disaster_messages.csv --category_filename disaster_categories.csv --database_filename preprocessed.db.sqlite3

import sys, os
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

MESSAGE_FILENAME = 'disaster_messages.csv'
CATEGORY_FILENAME = 'disaster_categories.csv'
DATABASE_FILENAME = 'preprocessed.db.sqlite3'
TABLE_NAME = 'data_preprocessed'

def load_data(message_filepath, category_filename):
    '''
    Load csv files: one with messages, one with category names
    '''
    messages = pd.read_csv(message_filename)
    categories = pd.read_csv(category_filename)

    df = pd.merge(categories, messages, how='outer', on='id')

    return df


def clean_data(df):
    '''
    Clean the read-in dataframe. 
    '''
    # create a dataframe of the n individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe,
    # use this row to extract a list of new column names for categories
    row = list(categories.iloc[0])
    category_colnames = [x.split('-')[0] for x in row]
    # rename the columns of 'categories'
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from df
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new dataframe
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Save and return the cleaned dataframe as a SQLite database.
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(TABLE_NAME, engine, index=False)


def parse_input_argument():
    '''
    Argument parser. The functions are shown in the help descriptions. 
    '''
    parser = argparse.ArgumentParser(description = 'Disaster Responser Data Processor')
    parser.add_argument('--message_filename', type=str, default=MESSAGE_FILENAME, help="Filename of the messages database (input)")
    parser.add_argument('--category_filename', type=str, default=CATEGORY_FILENAME, help="Filename of the categories database (input)")
    parser.add_argument('--database_filename', type=str, default=DATABASE_FILENAME, help="Filename for the cleaned data that is going to save (output)")
    args = parser.parse_args()
    
    return (args.message_filename, args.category_filename, args.database_filename)


def processor(message_filename, category_filename, database_filename):
    '''
    The main function.
    Provided inputs:
        - message_filename: the input message csv file name.
        - category_filename: the input message category csv file name.
        - database_filename: the output saved database name.
    '''
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(message_filename, category_filename))
    df = load_data(message_filename, category_filename)

    print('Cleaning data...')
    df = clean_data(df)
        
    print('Saving data...\n    DATABASE: {}'.format(database_filename))
    save_data(df, database_filename)
        
    print('Cleaned data saved to database!')
    

if __name__ == '__main__':
    message_filename, category_filename, database_filename = parse_input_argument()
    processor(message_filename, category_filename, database_filename)

