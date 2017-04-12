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

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    return data


def main():
    data = get_data('top_emoji_training.txt')
    (ran, data) = train_test_split(data, test_size = 0.5)
    total_count = len(data.index)
    print total_count
    Labels = ['\xe5\xbf\x83', '\xe5\x96\x9c\xe6\xac\xa2', '\xe7\x88\xb1', '\xe5\x95\x8a', '\xe6\x9d\xa5', '3', '\xe5\x93\x88\xe5\x93\x88', '1', '\xe4\xb8\x8d', '2', '4', '\xe5\xbc\x80\xe5\xbf\x83', '\xe5\x8e\xbb', '\xe5\xb9\xb8\xe7\xa6\x8f', '\xe7\x82\xb9']

    words_dict = {}
    priors_dict = {}
    for each in Labels:
        words_dict[each] = ''
        priors_dict[each] = 0

    #print words_dict
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
        #print words_dict[each]
        words_dict[each] = pd.DataFrame(words_dict[each], columns=['word', 'count_'+each]).set_index('word')

    #print words_dict['2']
    total_frame = pd.DataFrame(total_words_f, columns=['word', 'count_t']).set_index('word')

    print words_dict.keys()


    count_1 = words_dict['1']
    count_2 = words_dict['\xe5\x96\x9c\xe6\xac\xa2']
    count_3 = words_dict['\xe7\x88\xb1']
    count_4 = words_dict['\xe5\x95\x8a'] 
    count_5 = words_dict['\xe6\x9d\xa5'] 
    count_6 = words_dict['\xe4\xb8\x8d'] 
    count_7 = words_dict['\xe5\x93\x88\xe5\x93\x88'] 
    count_8 = words_dict['\xe5\xbf\x83'] 
    count_9 = words_dict['3'] 
    count_10 = words_dict['2'] 
    count_11 = words_dict['4'] 
    count_12 = words_dict['\xe5\xbc\x80\xe5\xbf\x83'] 
    count_13 = words_dict['\xe5\x8e\xbb'] 
    count_14 = words_dict['\xe5\xb9\xb8\xe7\xa6\x8f'] 
    count_15 = words_dict['\xe7\x82\xb9']

    final_table = pd.concat([total_frame, count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9, count_10, count_11, count_12, count_13, count_14, count_15],axis = 1)

    #print final_table.sample(n=5)
    

    final_table['count_1'] += 1
    final_table['count_\xe5\x96\x9c\xe6\xac\xa2'] += 1
    final_table['count_\xe7\x88\xb1'] += 1
    final_table['count_\xe5\x95\x8a'] += 1
    final_table['count_\xe6\x9d\xa5'] += 1
    final_table['count_\xe4\xb8\x8d'] += 1
    final_table['count_\xe5\x93\x88\xe5\x93\x88'] += 1
    final_table['count_\xe5\xbf\x83'] += 1
    final_table['count_3'] += 1
    final_table['count_2'] += 1
    final_table['count_4'] += 1
    final_table['count_\xe5\xbc\x80\xe5\xbf\x83'] += 1
    final_table['count_\xe5\x8e\xbb'] += 1
    final_table['count_\xe5\xb9\xb8\xe7\xa6\x8f'] += 1
    final_table['count_\xe7\x82\xb9'] += 1

    final_table['count_t'] += 15

    final_table = final_table.fillna(1)

    #print final_table.sample(n=5)
    #print priors_dict

    for each in Labels:
        priors_dict[each] = float(priors_dict[each])/total_count

    #print priors_dict
    

    final_table['count_1'] = final_table['count_1']/final_table['count_t']
    final_table['count_\xe5\x96\x9c\xe6\xac\xa2']= final_table['count_\xe5\x96\x9c\xe6\xac\xa2']/final_table['count_t']
    final_table['count_\xe7\x88\xb1'] = final_table['count_\xe7\x88\xb1']/final_table['count_t']
    final_table['count_\xe5\x95\x8a'] = final_table['count_\xe5\x95\x8a']/final_table['count_t']
    final_table['count_\xe6\x9d\xa5']= final_table['count_\xe6\x9d\xa5']/final_table['count_t']
    final_table['count_\xe4\xb8\x8d'] =  final_table['count_\xe4\xb8\x8d']/final_table['count_t']
    final_table['count_\xe5\x93\x88\xe5\x93\x88'] =final_table['count_\xe5\x93\x88\xe5\x93\x88']/final_table['count_t']
    final_table['count_\xe5\xbf\x83'] = final_table['count_\xe5\xbf\x83']/final_table['count_t']
    final_table['count_3'] = final_table['count_3']/final_table['count_t']
    final_table['count_2'] = final_table['count_2']/final_table['count_t']
    final_table['count_4'] = final_table['count_4']/final_table['count_t']
    final_table['count_\xe5\xbc\x80\xe5\xbf\x83'] = final_table['count_\xe5\xbc\x80\xe5\xbf\x83']/final_table['count_t']
    final_table['count_\xe5\x8e\xbb']  = final_table['count_\xe5\x8e\xbb'] /final_table['count_t']
    final_table['count_\xe5\xb9\xb8\xe7\xa6\x8f'] =  final_table['count_\xe5\xb9\xb8\xe7\xa6\x8f']/final_table['count_t']
    final_table['count_\xe7\x82\xb9'] =  final_table['count_\xe7\x82\xb9']/final_table['count_t']

    print final_table.sample(n=5)
    
    f = open("naive_model.txt", "w+")
    pickle.dump([final_table, priors_dict], f)
    f.close()
    return

if __name__ == "__main__":
    main()