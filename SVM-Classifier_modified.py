import numpy
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
emojis=[]

f=open("svm_input_top_emojis.txt")
data=pickle.load(f)
f.close()
train_msgs=data[0][0:50000]
train_emoji_labels=data[1][0:50000] ##training list of messages
dev_msgs=data[2][0:12500]
dev_emoji_labels=data[3][0:12500]  ## test set of messages
X_train = np.array(train_msgs)
y_train_text = train_emoji_labels  
X_test = np.array(dev_msgs)
y_test_text = dev_emoji_labels
print "one"
## set of 15 emoticons obtained by clustering algorithm 
top_emojis= ['\xe9\x9e\xad\xe7\x82\xae', '\xe4\xb8\x8d', '\xe5\x95\x8a', '\xe7\x88\xb1', '\xe5\xb7\xa8\xe8\x9f\xb9\xe5\xba\xa7', '1', '\xe4\xb8\x8b\xe9\x9b\xa8', '\xe6\xb8\xa9\xe6\x9a\x96', '\xe8\x99\x8e', '\xe7\x8b\xae\xe5\xad\x90\xe5\xba\xa7', '\xe6\x9d\xa5', '\xe5\x8e\xbb', '\xe7\x82\xb9', '\xe5\xa4\x84\xe5\xa5\xb3', '\xe7\xba\xa2\xe5\x8c\x85']
lb = preprocessing.MultiLabelBinarizer(classes=tuple(top_emojis))
print "lb: ",lb
print "two"
Y = lb.fit_transform(y_train_text)
Y_test = lb.fit_transform(y_test_text)

#clf = OneVsRestClassifier()
print Y_test,"**************"
f = open("Y_actual answer.txt", "w")
#f.write(str(Y_test))
b = np.array(Y_test)
np.savetxt('Y_actual answer.txt', b,fmt='%i')

#numpy.savetxt('Y_actual answer.txt', Y_test, delimiter = ',')  
f = open("Y.txt", "w")
#f.write(str(Y)) 
b = np.array(Y)
np.savetxt('Y.txt', b,fmt='%i')

print Y
print "three"
classifier = Pipeline([
('vectorizer', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', OneVsRestClassifier(sklearn.svm.SVC(kernel='rbf', C=1000, gamma = 0.0001)))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)

print "Answer:\n",predicted
f = open("predicted.txt", "w",)
b = np.array(predicted)
np.savetxt('predicted.txt', b,fmt='%i')

#f.write(str(predicted))

print ("<------------------------------>")


##calculating precision,recall,fscores and confidence

precision, recall, fscore, support = score(Y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print ("<------------------------------>")
added = np.add(Y_test, predicted)
t_c = len(added)
h_c = 0
for each in added:
    if 2 in each:
        h_c = h_c + 1

print t_c, h_c, float(h_c)/t_c
print "Accuracy Score: ",accuracy_score(Y_test, predicted)
