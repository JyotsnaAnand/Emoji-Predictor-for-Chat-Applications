#implementation of K nearest neighbors algorithm to predict the 5 most likely emoticons for each test instance

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import preprocessing
import sys
from collections import Counter
from itertools import chain
from collections import OrderedDict
import pickle
import math
import preprocessing
from functools import reduce
import ast
import codecs

#helper function to read input data and convert it to a dataframe
def get_data(file_name):    
    data = pd.read_csv(file_name, sep=',', error_bad_lines=False, encoding ='utf-8')
    return data

#helper function to convert tokenized train/test data sets into a vector of 0 and 1. This will be used to compute distance measure for KNN. 
def getVector(VectorInput):
     mlb = MultiLabelBinarizer()
     target=mlb.fit_transform(VectorInput)
     return target

#find distance between data points and identify the top 5 common emoticons among the K neighbors
#Three distance measures (Euclidean, Manhattan, Chi-squared were used and Chi-squared distance gave the highest accuracy on the Sina Weibo data corpus.     
def predict(X_train, y_train, x_test, k, disType):
    distances = []
    targets = []

    for i in range(len(X_train)):
        
        
        #euclidean distance
        if disType==0:
            distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        
        #manhattan distance
        elif disType==1: 
            distance=np.sum(x_test - X_train[i, :])

        #chi-squared distance
        elif disType==2:
            distance = (np.sum((np.square(x_test - X_train[i, :]))/np.sum(X_train[i, :])))
        
        distances.append([distance, i])

    #Sort distances found above and add the target labels of 'k' nearest neighbors to the 'targets' list. 
    distances = sorted(distances)
    for i in range(k):
            index=distances[i][1] 
            emoji_lis=y_train[index].split(',')
            
            for each in emoji_lis:
                targets.append(each)
        
    #Find top 5 emojis predicted from the 'targets' list
    topPred=Counter(targets).most_common(5)
    return ((topPred[0][0],topPred[1][0],topPred[2][0], topPred[3][0], topPred[4][0]))

#Method that implements KNN algorithm.
def kNearestNeighbor(X_train, y_train, X_test, predictions, k, disType):
    #predict emoticons for each test instance
    #disType determines the type of distance measure used. 
    for i in range(len(X_test)):
        predictions.append(set(predict(X_train, y_train, X_test[i, :], k, disType)))
                
        
def main():
    #get preprocessed data (Sina weibo) messages appended with a column specifying the emoticons occuring in the sentence
    emoji_data = get_data('top_emoji_training_new.txt')
    words=[]
    for row in emoji_data.itertuples():
        wordList=row[-2].split()
        words.append(wordList)
        
    #transform the words into a vector
    wordsVector=getVector(words)

    #split data and target labels into training and development sets
    (top_testing_vec, top_training_vec) = train_test_split(wordsVector, test_size = 0.8, random_state=100)
    (top_testing_data, top_training_data) = train_test_split(preprocessing.get_specific_columns(emoji_data, ['words', 'emojis']), test_size = 0.8,random_state=100)
    top_testing_data.columns=['words','emoji']
    top_training_data.columns=['words','emoji']    
    trainLabels=emoji_data['emojis']

    #transform target labels of development set into a vector 
    emojiTestLabels=[]
    for row in top_testing_data.itertuples():
        emojiList=(row[-1].split())
        emojiTestLabels.append(set(emojiList))
    emojiTestLabels = np.asarray(emojiTestLabels)

    #test the KNN model for 3 distance measures (Euclidean, Manhattan, Chi-square)
    #i values - 0 = Euclidean, 1=Manhattan, 2=Chi square
    scoreList=[]
    
    for i in range(0,3):
        scoreList.append([])
        #test KNN for different values of 'K' to find the optimal number of neighbors 
        for k in range(20,200,5):

            #predictions[] stores the predictions returned by KNN
            predictions = []
            
            #call to the KNN method to find predictions for development set
            kNearestNeighbor(top_training_vec,trainLabels,top_testing_vec, predictions, k, i)
            
            # transform predictions list into an array
            predictions = np.asarray(predictions)

            #find accuracy of predicitons found and true target labels for development set
            accuracyTest=np.asarray(emojiTestLabels & predictions)            
            correctPredictions = (accuracyTest != set()).sum()
            accuracyScore=(correctPredictions/len(top_testing_data))*100
            print(k,accuracyScore)
            scoreList[i].append(accuracyScore)
        print(scoreList[i])

        #find maximum accuracy, K value that gives the maximum accuracy
        print ("Distance measure: ",i, max(scoreList[i]))
        print("position:", scoreList[i].index(max(scoreList[i])))
    
    #create dictionaries for the emoji clusters found in pre-processing phase
    #to find precision, recall, F1 score and Confidence measures for each Emoji class
        
    truePosDict = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}
    falsePosDict = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}
    falseNegDict = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}

    precision = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}
    recall = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}
    f1score = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}
    confidence = {'鞭炮': 1, '1': 1, '不':1,'啊':1, '爱':1, '巨蟹座':1, '下雨':1, '红包':1, \
                      '温暖':1, '虎':1, '狮子座':1, '来':1, '去':1,  '点':1, '处女':1}
    
    
    #find true positives, false negatives, false positives   
    truePosSet=emojiTestLabels & predictions
    falsePosSet=emojiTestLabels - predictions
    falseNegSet=predictions  - emojiTestLabels    
    for x in truePosSet:
       for s in x:
           truePosDict[s]+=1
    for x in falsePosSet:
       for s in x:
           falsePosDict[s]+=1
    for x in falseNegSet:
       for s in x:
           falseNegDict[s]+=1
      
    
    #find precision
    for k,v in precision.items():
        try:
            precision[k]=truePosDict[k]/(truePosDict[k]+falsePosDict[k])
        except ZeroDivisionError:
            print("Zero Division in Precision", k)
            
    #find recall
    for k,v in recall.items():
        try:
            recall[k]=truePosDict[k]/(truePosDict[k]+falseNegDict[k])
        except ZeroDivisionError:
            print("Zero Division in Recall", k)
            
    #find f1 score
    for k,v in f1score.items():
        try:
            f1score[k]=2*(precision[k]*recall[k])/(precision[k]+recall[k])
        except ZeroDivisionError:
            print("Zero Division in Recall", k)
            
    #find confidence
    for x in predictions:
       for s in x:
           if s not in confidence:
               confidence[s]=1/5
           else:
               confidence[s]+=(1/5)
    confSum=sum(confidence.values())
    for k,v in confidence.items():
        confidence[k]=confidence[k]/confSum
            
    print(precision)
    print(recall)
    print(f1score)
    print(confidence)
          
    
main()
