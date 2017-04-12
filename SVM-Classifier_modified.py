import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle
emojis=[]
###with open('emoji-list.txt') as fp:
   ### for line in fp:
      ###  line = line.strip()
#	emojis.append(str(line))
#fp.close()
#################
f=open("svm_input_top_emojis.txt")
data=pickle.load(f)
f.close()

train_msgs=data[0]

#train_msgs = train_msgs
train_emoji_labels=data[1]
#train_emoji_labels = train_emoji_labels


dev_msgs=data[2]
#dev_msgs = dev_msgs
dev_emoji_labels=data[3]

#dev_emoji_labels = dev_emoji_labels
#################
#X_train = np.array(train_msgs)
X_train = train_msgs
y_train_text = train_emoji_labels  


#X_test = np.array(dev_msgs)
X_test = dev_msgs

y_test_text = dev_emoji_labels

#print "one"
top_emojis=['\xe5\xbf\x83', '\xe5\x96\x9c\xe6\xac\xa2', '\xe7\x88\xb1', '\xe5\x95\x8a', '\xe6\x9d\xa5', '3', '\xe5\x93\x88\xe5\x93\x88', '1', '\xe4\xb8\x8d', '2', '4', '\xe5\xbc\x80\xe5\xbf\x83', '\xe5\x8e\xbb', '\xe5\xb9\xb8\xe7\xa6\x8f', '\xe7\x82\xb9']
lb = preprocessing.MultiLabelBinarizer(classes=tuple(top_emojis))
#print "two"
Y = lb.fit_transform(y_train_text)
Y_test = lb.fit_transform(y_test_text)
#clf = OneVsRestClassifier()
#print Y_test,"**************"
#print Y
#print "three"
classifier = Pipeline([
('vectorizer', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', C=1000, gamma = 0.0001)))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
#print "Answer:\n",predicted


added = np.add(Y_test, predicted)
t_c = len(added)
h_c = 0
for each in added:
    if 2 in each:
        h_c = h_c + 1

print 'Accuracy:', t_c, h_c, float(h_c)/t_c
#print "Accuracy Score: ",accuracy_score(Y_test, predicted)
