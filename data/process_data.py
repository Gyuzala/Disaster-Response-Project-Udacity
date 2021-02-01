import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    """
    Load data from a database
    Input: messages_filepath - String
            The path to the messagages data csv
           categories_filepath - String
            The path to the categories data csv
    Output: df = DataFrame
            Merged DataFrame of message and categories data
    """
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    df = messages.merge(categories, on = ['id'], how = 'outer')
    
    return df

def clean_data(df):
    """
    Extracts categories and flags from categories data, remove duplicates
    Input: df - DataFrame
            Dataframe output from load_data function
    Output: df - DataFrame
            Cleansed dataframe of the input data
    """   
    categories = df['categories'].str.split(pat=';',expand=True)
    
    
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
    
    df = df.drop(['categories'], axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],join='inner', axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    
def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

   
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data to SQLite DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
        

if __name__ == '__main__':
    main()