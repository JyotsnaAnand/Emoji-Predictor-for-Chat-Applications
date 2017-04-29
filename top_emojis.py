#
#
# importing the packages required
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
import operator

#
#
#
# importing Internal packages
import preprocessing

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    return data

def main():
    emoji_data = get_data('messages_emojis_full.csv')
    emoji_data = emoji_data.dropna()

    print len(emoji_data.index)

    emoji_data["emojis"] = emoji_data["emojis"].apply(lambda x: str(x))
    emoji_list = Counter(' '.join(emoji_data["emojis"].tolist()).split(' '))

    f = open('tcluster_output.txt')
    cluster_list = pickle.load(f)
    f.close()
    print type(emoji_list)
    for each in cluster_list:
        if each != []:
            temp_dic = {}
            for ele in each:
                temp_dic[ele] = emoji_list[ele]
            sorted_x = sorted(temp_dic.items(), key=operator.itemgetter(1))
            sorted_x.reverse()
            
            # Printing top 10 emojies in each cluster
            #print sorted_x[0:10]

    # the list of emojis based on the freequency displayed by the above code from each cluster
    top_emojis =['\xe9\x9e\xad\xe7\x82\xae', '\xe4\xb8\x8d', '\xe5\x95\x8a', '\xe7\x88\xb1', '\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7', '1', '\xe4\xb8\x8b\xe9\x9b\xa8', '\xe6\xb8\xa9\xe6\x9a\x96', '\xe8\x99\x8e', '\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7', '\xe6\x9d\xa5',  '\xe5\x8e\xbb', '\xe7\x82\xb9', '\xe5\xa4\x84\xe5\xa5\xb3', '\xe7\xba\xa2\xe5\x8c\x85']
    emoji_data['emojis'] = emoji_data['emojis'].apply(lambda x: ','.join(list(set(x.split(',')).intersection(top_emojis))))

    emoji_data['emojis'].replace('', np.nan, inplace=True)
    emoji_data = emoji_data.dropna()

    (top_training, top_testing) = train_test_split(preprocessing.get_specific_columns(emoji_data, ['words', 'emojis']), test_size = 0.8)

    top_testing.to_csv('top_emoji_training_new.txt')
    top_training.to_csv('top_emoji_testing_new.txt')

    return


if __name__ == "__main__":
    main()