# emoJI

Emojis are widely used in social media today. The number of new Emojis emerging have also increased. Predicting Emojis based on the context and content of messages is an important problem in the field of Natural Language Processing. 

We experimented with Naive Bayes, SVM and KNN algorithms and tested the Emoji predictability across all of these algorithms. We then implemented a 'Voting algorithm' which takes input of precision,recall, F1 and confidence from the above mentioned models and predicts the most likely Emojis based on the highest value received. 

emoji2vec.py - to Create emoji2vec using Deep Learning
preprocessing.py - contatining the methods used to cleanup the corpus
top_emojis.py - to pick the right emojis for the further classification

Naive Bayes:
  naive_bayes_learn.py - create the naive bayes model for the 15 classes
  naive_bayes_test.py - emoji prediction using the Naive Bayes model file created while learning
  
SVM:
  svm_input_modified.py - to create the input file suitable to feed the SVM algorithm
  SVM_Classifier_modified.py - SVM algorithm implementation for the emoji prediction
KNN: 
  KNN_emoji_prediction.py - predicts the emojies using KNN algorithm
  
Ranking the emojies predected for better prediction-
  rankingAlgorithm.py

References: 
Data Corpus: http://lwc.daanvanesch.nl/openaccess.php 

We would like to thank Leiden Weibo Corpus for making their Dataset open source. 

Contributors:

Jyotsna Anand 			janand@usc.edu
Santhana Gopalan Raghavan 	santhanr@usc.edu
Saravanan Ravanan		ravanan@usc.edu
Supriya Nallapeta 		nallapet@usc.edu
