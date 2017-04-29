
## Table containing confidence values for emoticons predicted by Naive Bayes
with open('NBpredictionConfidence.txt') as fp:
   for line in fp:
      line = line.strip()
      nbEmojiConfidence.append(str(line))
fp.close()

## Table containing confidence values for emoticons predicted by Support Vector Machines
with open('SVMpredictionConfidence.txt') as fp:
   for line in fp:
      line = line.strip()
      svmEmojiConfidence.append(str(line))
fp.close()

## Table containing confidence values for emoticons predicted by K-Nearest Neighbours
with open('KNNpredictionConfidence.txt') as fp:
   for line in fp:
      line = line.strip()
      knnEmojiConfidence.append(str(line))
fp.close()

## Table containing accuracy values for each of the algorithms
with open('accuracyTable.txt') as fp:
   for line in fp:
      line = line.strip()
      emojiAccuracy.append(str(line))
fp.close()

for i in range(len(nbEmojiConfidence)):
	tempNb = nbEmojiConfidence[i].split('##')
	tempSvm = SVMEmojiConfidence[i].split('##')
	tempKnn = KNNEmojiConfidence[i].split('##')
	ranklist = []	
	for j in range(len(tempNb)):
		nb_emoticon_i_confidence_i = tempNb[j].split(' ')
		svm_emoticon_i_confidence_i = tempSvm[j].split(' ')
		knn_emoticon_i_confidence_i = tempKnn[j].split(' ')
		e1 = nb_emoticon_i_confidence_i[0]
		e2 = nb_emoticon_i_confidence_i[0]
		e3 = nb_emoticon_i_confidence_i[0]
		e4 = svm_emoticon_i_confidence_i[0]
		e5 = svm_emoticon_i_confidence_i[0]
		e6 = svm_emoticon_i_confidence_i[0]
		e7 = knn_emoticon_i_confidence_i[0]
		e8 = knn_emoticon_i_confidence_i[0]
		e9 = knn_emoticon_i_confidence_i[0]
		
		c1 = nb_emoticon_i_confidence_i[1]
		c2 = nb_emoticon_i_confidence_i[1]
		c3 = nb_emoticon_i_confidence_i[1]
		c4 = svm_emoticon_i_confidence_i[1]
		c5 = svm_emoticon_i_confidence_i[1]
		c6 = svm_emoticon_i_confidence_i[1]
		c7 = knn_emoticon_i_confidence_i[1]
		c8 = knn_emoticon_i_confidence_i[1]
		c9 = knn_emoticon_i_confidence_i[1]
		
## Computing product of accuracy and confidence values to generate ranking values
		ranklist.append(emojiAccuracy[e1,"NB"]*c1)
		ranklist.append(emojiAccuracy[e2,"NB"]*c2)
  		ranklist.append(emojiAccuracy[e3,"NB"]*c3)
  		ranklist.append(emojiAccuracy[e4,"SVM"]*c4)
  		ranklist.append(emojiAccuracy[e5,"SVM"]*c5)
  		ranklist.append(emojiAccuracy[e6,"SVM"]*c6)
  		ranklist.append(emojiAccuracy[e7,"KNN"]*c7)
  		ranklist.append(emojiAccuracy[e8,"KNN"]*c8)
  		ranklist.append(emojiAccuracy[e9,"KNN"]*c9)
  		  		
ranklist.append(emojiAccuracy[emoticon,"SVM"]*confidence)			
		ranklist.append(emojiAccuracy[emoticon,"KNN"]*confidence)
	ranklist.sort()
	ranklist.reverse()
## Printing the top 3 emoticons with greatest accuracy
	print ranklist[0],ranklist[1],ranklist[2]
