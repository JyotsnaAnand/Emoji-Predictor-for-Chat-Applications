import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from collections import Counter
from itertools import chain
from collections import OrderedDict
import pickle
import math
import preprocessing


def get_model_data_from(file_name):
    f = open(file_name)
    lis = pickle.load(f)
    f.close()
    return (lis[0], lis[1], lis[2])

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False)

    return data

def main():

    training_data = get_data("top_emoji_training.txt")
    development_data = get_data("top_emoji_testing.txt")

    training_data = training_data.dropna()
    development_data = development_data.dropna()

    training_data = preprocessing.get_specific_columns(training_data, ['words', 'emojis'])
    development_data = preprocessing.get_specific_columns(development_data, ['words', 'emojis'])

    training_data['emojis'] = training_data['emojis'].apply(lambda x: str(x).split(','))
    development_data['emojis'] = development_data['emojis'].apply(lambda x: str(x).split(','))
    train_msgs = training_data['words'].tolist()
    dev_msgs = development_data['words'].tolist()

    train_emojis = training_data['emojis'].tolist()
    dev_emojis = development_data['emojis'].tolist()

    #print train_emojis[0:10]


    #print train_emojis

    #gender_data['emojis'] = gender_data['words'].apply(lambda x: ','.join(list(set(x.split(' ')).intersection(emoji_set))))


    #train_emojis = [str(i).split(',') for i in train_emojis]
    #dev_emojis = [str(i).split(',') for i in dev_emojis]



    f = open("svm_input_top_emojis.txt", "w+")
    print dev_emojis[1:10]
    print dev_msgs[1:10]
    pickle.dump([np.array(train_msgs[0:30000]), train_emojis[0:30000], np.array(dev_msgs[0:500]), dev_emojis[0:500]], f)
    f.close()

if __name__ == "__main__":
    main()


def get_model_data_from(file_name):
    f = open(file_name)
    lis = pickle.load(f)
    f.close()
    return (lis[0], lis[1], lis[2], lis[3])
