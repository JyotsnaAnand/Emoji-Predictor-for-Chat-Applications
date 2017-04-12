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
    return (lis[0], lis[1])

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False)
    return data

(final_table, priors_dict) = get_model_data_from("naive_model.txt")

def calculate(x, l):
    p = 1
    lis = str(x).split(' ')
    for each in lis:
        if each in final_table.index:
            p = p * float(final_table.get_value(each, 'count_'+l))
    return p

def main():
    
    testing_data = get_data("top_emoji_testing.txt")
    (ran, testing_data) = train_test_split(testing_data, test_size = 0.003)
    print len(testing_data.index)
    Labels = ['\xe5\xbf\x83', '\xe5\x96\x9c\xe6\xac\xa2', '\xe7\x88\xb1', '\xe5\x95\x8a', '\xe6\x9d\xa5', '3', '\xe5\x93\x88\xe5\x93\x88', '1', '\xe4\xb8\x8d', '2', '4', '\xe5\xbc\x80\xe5\xbf\x83', '\xe5\x8e\xbb', '\xe5\xb9\xb8\xe7\xa6\x8f', '\xe7\x82\xb9']


    for l in Labels:
        testing_data[l] = testing_data['words'].apply(lambda x: calculate(x, l))

    def my_test(row):
        m = row['1']
        l = '1'
        ss = row['1']
        s = '1'
        for each in Labels:
            if m < row[each]:
                s = l
                ss = m
                m = row[each]
                l = each
            elif ss < row[each]:
                s = each
                ss = row[each]
        return str(l)+','+str(s) 
    testing_data['max'] = testing_data.apply(lambda row: my_test(row), axis=1)
    e_l = testing_data['emojis'].tolist()
    m_l = testing_data['max'].tolist()
    le = len(m_l)
    pos = 0
    for i in range(0, le):
        sec = str(m_l[i]).split(',')
        ems = str(e_l[i]).split(',')
        if sec[0] in ems or sec[1] in ems:
            pos = pos + 1

    print le, pos, float(pos)/le


    res = preprocessing.get_specific_columns(testing_data, ['emojis','max'])
    print res.sample(n=10)
    return

if __name__ == "__main__":
    main()