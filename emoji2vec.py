# coding: utf-8
#
#
#
#importing required packages

from gensim.models import word2vec
import pandas as pd
import numpy as np
from collections import Counter
import pickle
from sklearn.cluster import KMeans

#
# internal file contatining the common code
import preprocessing

#
# Reading emojis data in each sentense of the corpus
emojis = pd.read_csv('/Users/supriya/Desktop/emoji/messages_emojis.csv', sep=',', error_bad_lines=False)
#
# Splitting emojis in each sentense by space and creating a list of list from it
emojis['emojis'] = emojis['emojis'].str.split(' ')
emojis_list = emojis['emojis'].tolist()
#
# Eliminating NaN in the data
emojis_list = [x for x in emojis_list if str(x) != 'nan']
#
# Creating a emoji2vec model using gensim word2vec model with vector size of 200
model = word2vec.Word2Vec(emojis_list, size=200)
#
# Reading all the emoticons used in 'weibo' into a data frame using preprocessing function 
df = preprocessing.read_emoticons_dataframe('meta_emoticons.txt')
#
# extracting only the column containing emoji names
emoji_column = preprocessing.get_specific_columns(df, ['emoji'])
emoji_column["emoji"] = emoji_column["emoji"].str.strip(']').str.strip('[')
emoji_name_list = emoji_column["emoji"].tolist() 
#
# extracting only the emoticons present in the corpus
new_emoji_list = []
for each in emoji_name_list:
    if each in model.wv.vocab:
        new_emoji_list.append(each)

# applying K-means clustering algorithm to find the clusters of emoticons in the emoji2vec model
# 40 vlaue denotes the average number of emoticons in each cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 40

#
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

#
# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

#
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.wv.index2word, idx ))

#
# For the 15 clusters formed
clusters_list = []
for cluster in xrange(0,15):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    clusters_list.append(words)
    print words

#
# Storing the cluster results
f = open('cluster_output.txt','w+')
pickle.dump(clusters_list, f)
f.close()


# to find the most similar emoticons
#model.wv.most_similar('\xe6\xa1\xa3\xe6\xa1\x88')

#
#
# to find the similarity between two emotions in the model
#model.wv.similarity('\xe6\xa1\xa3\xe6\xa1\x88','\xe6\x9d\xa5')

