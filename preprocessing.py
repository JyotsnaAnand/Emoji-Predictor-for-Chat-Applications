#
#
#
# Importing required packages
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# method to read messages from the corpus and convert it into a data frame for the first time read 
def read_messages_dataframe_original(file_name):
    df = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    df.columns = ['message_id', 'province_code', 'city_code', 'gender', 'screen_name', 'original_message', 'word_count', 'words', 'words_tags']
    df.to_csv('clean_parsed_messages.csv')
    return df

# method to read messages from the corpus and convert it into a data frame
def read_messages_dataframe(file_name):
    df = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    return df

# method to read emoticons from the corpus and convert it into a data frame for the first time read 
def read_emoticons_dataframe(file_name):
    df = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    df.columns = ['emoji_id', 'emoji', 'emoji_link']
    return df

# method to extract specific columns from the data frame
def get_specific_columns(data_frame, cols):
    new_data_frame = data_frame.filter(cols, axis=1)
    return new_data_frame


# method to split the data frame into different splits when required
def split_data(data_frame, training = 70, development = 20, testing = 10):
    training_frame, test_frame = train_test_split(data_frame, test_size = 0.3)

    development_frame, testing_frame = train_test_split(test_frame, test_size = 0.33)

    return (training_frame, development_frame, testing_frame)


