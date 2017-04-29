#
#
#
#
# Naive Bayes Classifier learning algorithm for emoji classifier

#importing external packages required

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from collections import Counter
from itertools import chain
from collections import OrderedDict
import pickle
import math

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False, nrows = 45000)
    return data


def main():

    # reading Data containing messages and the emojis in each message 
    data = get_data('top_emoji_training_new.txt')
    total_count = len(data.index)

    # class labels - emojies considered based on the frequency and clusters formed through emoji2vec
    Labels = ['\xe9\x9e\xad\xe7\x82\xae', '\xe4\xb8\x8d', '\xe5\x95\x8a', '\xe7\x88\xb1', '\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7', '1', 
    '\xe4\xb8\x8b\xe9\x9b\xa8', '\xe6\xb8\xa9\xe6\x9a\x96', '\xe8\x99\x8e', 
    '\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7', '\xe6\x9d\xa5',  '\xe5\x8e\xbb', '\xe7\x82\xb9', 
    '\xe5\xa4\x84\xe5\xa5\xb3', '\xe7\xba\xa2\xe5\x8c\x85']

    words_dict = {}
    priors_dict = {}
    for each in Labels:
        words_dict[each] = ''
        priors_dict[each] = 0

    for row in data.itertuples():
        emoji_lis = row[-1].split(',')
        msg = row[-2]
        for each in emoji_lis:
            words_dict[each] = words_dict[each]+' '+msg
            priors_dict[each] = priors_dict[each] + 1


    for K,V in words_dict.iteritems():
        words_dict[K] = Counter(V.split(' ')).most_common()

    total_words = ' '.join(data['words'].tolist())
    total_words_f = Counter(total_words.split(' ')).most_common()

    for each in words_dict.keys():
        words_dict[each] = pd.DataFrame(words_dict[each], columns=['word', 'count_'+each]).set_index('word')

    total_frame = pd.DataFrame(total_words_f, columns=['word', 'count_t']).set_index('word')

    count_1 = words_dict['1']
    count_2 = words_dict['\xe9\x9e\xad\xe7\x82\xae']
    count_3 = words_dict['\xe7\x88\xb1']
    count_4 = words_dict['\xe5\x95\x8a'] 
    count_5 = words_dict['\xe6\x9d\xa5'] 
    count_6 = words_dict['\xe4\xb8\x8d'] 
    count_7 = words_dict['\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7'] 
    count_8 = words_dict['\xe4\xb8\x8b\xe9\x9b\xa8'] 
    count_9 = words_dict['\xe6\xb8\xa9\xe6\x9a\x96'] 
    count_10 = words_dict['\xe8\x99\x8e'] 
    count_11 = words_dict['\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7'] 
    count_12 = words_dict['\xe5\xa4\x84\xe5\xa5\xb3'] 
    count_13 = words_dict['\xe5\x8e\xbb'] 
    count_14 = words_dict['\xe7\xba\xa2\xe5\x8c\x85'] 
    count_15 = words_dict['\xe7\x82\xb9']

    final_table = pd.concat([total_frame, count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9, count_10, count_11, count_12, count_13, count_14, count_15],axis = 1)


    final_table['count_1'] += 1
    final_table['count_\xe9\x9e\xad\xe7\x82\xae'] += 1
    final_table['count_\xe7\x88\xb1'] += 1
    final_table['count_\xe5\x95\x8a'] += 1
    final_table['count_\xe6\x9d\xa5'] += 1
    final_table['count_\xe4\xb8\x8d'] += 1
    final_table['count_\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7'] += 1
    final_table['count_\xe4\xb8\x8b\xe9\x9b\xa8'] += 1
    final_table['count_\xe6\xb8\xa9\xe6\x9a\x96'] += 1
    final_table['count_\xe8\x99\x8e'] += 1
    final_table['count_\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7'] += 1
    final_table['count_\xe5\xa4\x84\xe5\xa5\xb3'] += 1
    final_table['count_\xe5\x8e\xbb'] += 1
    final_table['count_\xe7\xba\xa2\xe5\x8c\x85'] += 1
    final_table['count_\xe7\x82\xb9'] += 1

    final_table['count_t'] += 15

    final_table = final_table.fillna(1)

    for each in Labels:
        priors_dict[each] = float(priors_dict[each])/total_count


    final_table['count_1'] = final_table['count_1']/final_table['count_t']
    final_table['count_\xe9\x9e\xad\xe7\x82\xae']= final_table['count_\xe9\x9e\xad\xe7\x82\xae']/final_table['count_t']
    final_table['count_\xe7\x88\xb1'] = final_table['count_\xe7\x88\xb1']/final_table['count_t']
    final_table['count_\xe5\x95\x8a'] = final_table['count_\xe5\x95\x8a']/final_table['count_t']
    final_table['count_\xe6\x9d\xa5']= final_table['count_\xe6\x9d\xa5']/final_table['count_t']
    final_table['count_\xe4\xb8\x8d'] =  final_table['count_\xe4\xb8\x8d']/final_table['count_t']
    final_table['count_\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7'] =final_table['count_\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7']/final_table['count_t']
    final_table['count_\xe4\xb8\x8b\xe9\x9b\xa8'] = final_table['count_\xe4\xb8\x8b\xe9\x9b\xa8']/final_table['count_t']
    final_table['count_\xe6\xb8\xa9\xe6\x9a\x96'] = final_table['count_\xe6\xb8\xa9\xe6\x9a\x96']/final_table['count_t']
    final_table['count_\xe8\x99\x8e'] = final_table['count_\xe8\x99\x8e']/final_table['count_t']
    final_table['count_\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7'] = final_table['count_\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7']/final_table['count_t']
    final_table['count_\xe5\xa4\x84\xe5\xa5\xb3'] = final_table['count_\xe5\xa4\x84\xe5\xa5\xb3']/final_table['count_t']
    final_table['count_\xe5\x8e\xbb']  = final_table['count_\xe5\x8e\xbb'] /final_table['count_t']
    final_table['count_\xe7\xba\xa2\xe5\x8c\x85'] =  final_table['count_\xe7\xba\xa2\xe5\x8c\x85']/final_table['count_t']
    final_table['count_\xe7\x82\xb9'] =  final_table['count_\xe7\x82\xb9']/final_table['count_t']

    # writing the model parameters to the file
    f = open("naive_model.txt", "w+")
    pickle.dump([final_table, priors_dict], f)
    f.close()
    return

if __name__ == "__main__":
    main()