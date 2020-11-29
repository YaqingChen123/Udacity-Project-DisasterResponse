#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy
import pandas as pd
from sqlalchemy import create_engine


# In[ ]:


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from dataset and merge both dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    frames = [messages, categories]
    df =  pd.concat(frames,axis=1, join='inner')
    return df


# In[ ]:


def clean_data(df):
    """
    Clean the loaded messages and categories data.
    Convert 'categories' column string into a set of numeric columns, and remove duplicate rows
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:len(x) - 2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
  
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`

    df=df.drop(columns=['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe

    frames = [df, categories]
    df =  pd.concat(frames,axis=1, join='inner')
    # drop duplicates
    df=df.drop_duplicates()
    return df


# In[ ]:


def save_data(df, database_filename):
    """
    save the clean dataset into an sqlite database
    """
    database_filename = "sqlite:///" + database_filename
    engine = create_engine(database_filename)
    df.to_sql("DisasterResponseTable", engine, index=False)

# In[ ]:
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '              'datasets as the first and second argument respectively, as '              'well as the filepath of the database to save the cleaned data '              'to as the third argument. \n\nExample: python process_data.py '              'disaster_messages.csv disaster_categories.csv '              'DisasterResponse.db')


# In[ ]:


if __name__ == '__main__':
    main()    
    
#input CMD
#python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisaterResponse.db
