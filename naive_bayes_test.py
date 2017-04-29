#
#
#
#importing required packages
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

#importing internal python file
import preprocessing

def get_model_data_from(file_name):
    f = open(file_name)
    lis = pickle.load(f)
    f.close()
    return (lis[0], lis[1])

def get_data(file_name):
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False, nrows = 5000)
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
    #
    #reading the testing data containing messages
    testing_data = get_data("top_emoji_testing.txt")
    print len(testing_data.index)
    Labels = ['\xe9\x9e\xad\xe7\x82\xae', '\xe4\xb8\x8d', '\xe5\x95\x8a', '\xe7\x88\xb1', '\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7', '1', 
    '\xe4\xb8\x8b\xe9\x9b\xa8', '\xe6\xb8\xa9\xe6\x9a\x96', '\xe8\x99\x8e', 
    '\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7', '\xe6\x9d\xa5',  '\xe5\x8e\xbb', '\xe7\x82\xb9', 
    '\xe5\xa4\x84\xe5\xa5\xb3', '\xe7\xba\xa2\xe5\x8c\x85']

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

    # Calculating the accuracy of the prediction
    #print le, pos, float(pos)/le


    res = preprocessing.get_specific_columns(testing_data, ['emojis','max'])
    real_lis = testing_data['emojis'].tolist()
    pred_lis = testing_data['max'].tolist()

    # Calculating the f1 score of the predition
    l = len(real_lis)
    tp = {}
    fp = {}
    fn = {}
    for label in Labels:
        tp[label] = 1.0
        fp[label] = 1.0
        fn[label] = 1.0
    for i in range(0, l):
        pr = pred_lis[i].split(',')
        re = real_lis[i].split(',')
        if len(re) == 1:
            if re[0] in pr:
                tp[re[0]] += 1
            else:
                for a in pr:
                    if a in re:
                        tp[a] += 1
                    else:
                        fp[a] += 1
            for a in re:
                if a not in pr and a in Labels:
                    fn[a] +=1

    for a in Labels:
        print tp[a], fp[a], fn[a]

    precision = {}
    recall = {}
    for l in Labels:
        precision[l] = tp[l] /(tp[l]+fp[l])
        recall[l] = tp[l] /(tp[l]+fn[l])
        print l, " precision: ", precision[l], " recall: ", recall[l], " f1: ",2*precision[l]*recall[l]/(precision[l]+recall[l])
    return

if __name__ == "__main__":
    main()