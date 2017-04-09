import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


#def filter_data(lines):
#    lines = [i.split(',') for i in lines]

'''
the Sina Weibo message ID
the original message text
the Sina Weibo code for the user's province
the Sina Weibo code for the user's city
the user's gender (m/f)
the user's screen name
the number of words in this message
the message text with word boundaries marked
the message text with word boundaries marked and POS tags indicated
'''



def read_messages_dataframe_original(file_name):
    df = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    #print len(df.columns)
    df.columns = ['message_id', 'province_code', 'city_code', 'gender', 'screen_name', 'original_message', 'word_count', 'words', 'words_tags']
    #print df['message_id']
    #print len(df)
    df.to_csv('clean_parsed_messages.csv')
    return df

def read_messages_dataframe(file_name):
    df = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    #print df.columns
    return df

def read_emoticons_dataframe(file_name):
    df = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    df.columns = ['emoji_id', 'emoji', 'emoji_link']
    return df

def get_specific_columns(data_frame, cols):
    new_data_frame = data_frame.filter(cols, axis=1)
    return new_data_frame

def split_data(data_frame, training = 70, development = 20, testing = 10):
    training_frame, test_frame = train_test_split(data_frame, test_size = 0.3)

    development_frame, testing_frame = train_test_split(test_frame, test_size = 0.33)

    return (training_frame, development_frame, testing_frame)



class Messages:
    def __init__(self, file_name=None):
        print file_name
        print 'hi'



