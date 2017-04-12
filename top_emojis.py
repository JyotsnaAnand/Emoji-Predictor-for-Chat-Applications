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

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    return data

def main():
    emoji_data = get_data('messages_emojis.csv')
    emoji_data = emoji_data.dropna()

    print len(emoji_data.index)
    #(a, emoji_data) = train_test_split(emoji_data, test_size = 0.0001)
    #print len(emoji_data.index)
    
    emoji_data["emojis"] = emoji_data["emojis"].apply(lambda x: str(x))
    emoji_list = Counter(','.join(emoji_data["emojis"].tolist()).split(',')).most_common(15)

    top_emojis = dict(emoji_list).keys()
    print top_emojis
    return

    emoji_data['emojis'] = emoji_data['emojis'].apply(lambda x: ','.join(list(set(x.split(',')).intersection(top_emojis))))

    emoji_data['emojis'].replace('', np.nan, inplace=True)
    emoji_data = emoji_data.dropna()

    #print len(emoji_data.index)
    #print emoji_data.sample(n=10)

    (top_training, top_testing) = train_test_split(preprocessing.get_specific_columns(emoji_data, ['words', 'emojis']), test_size = 0.8)

    top_training.to_csv('top_emoji_training.txt')
    top_testing.to_csv('top_emoji_testing.txt')

    return


if __name__ == "__main__":
    main()