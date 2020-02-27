# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:08:38 2020

@author: Melika
"""


#Define Libraries
import nltk
import random
import numpy as np 
import pandas as pd 
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("stopwords")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
set(stopwords.words("english"))
from nltk.tokenize import word_tokenize
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from random import seed
from random import randint
from random import shuffle
import array
import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sklearn.datasets 
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import re
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
########## IMPORT FOR CLUSTERING ##############
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans



#########################
#define list of possible books from gutenberg
nltk.corpus.gutenberg.fileids() 
["austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt", "bible-kjv.txt",
"blake-poems.txt", "bryant-stories.txt", "burgess-busterbrown.txt",
"carroll-alice.txt", "chesterton-ball.txt", "chesterton-brown.txt",
"chesterton-thursday.txt", "edgeworth-parents.txt", "melville-moby_dick.txt",
"milton-paradise.txt", "shakespeare-caesar.txt", "shakespeare-hamlet.txt",
"shakespeare-macbeth.txt", "whitman-leaves.txt"]
#########################


#########################
#choosing 7 books in list in lower case
#origin is the origin version of chosen books
origin = [nltk.corpus.gutenberg.raw("austen-emma.txt"), nltk.corpus.gutenberg.raw("milton-paradise.txt"),
          nltk.corpus.gutenberg.raw("chesterton-ball.txt") , nltk.corpus.gutenberg.raw("melville-moby_dick.txt") ,
          nltk.corpus.gutenberg.raw("edgeworth-parents.txt") , nltk.corpus.gutenberg.raw("bryant-stories.txt") ,
          nltk.corpus.gutenberg.raw("whitman-leaves.txt") ]
books = [nltk.corpus.gutenberg.raw("austen-emma.txt").lower(), nltk.corpus.gutenberg.raw("milton-paradise.txt").lower(),
         nltk.corpus.gutenberg.raw("chesterton-ball.txt").lower() , nltk.corpus.gutenberg.raw("melville-moby_dick.txt").lower(),
         nltk.corpus.gutenberg.raw("edgeworth-parents.txt").lower() , nltk.corpus.gutenberg.raw("bryant-stories.txt").lower() , 
         nltk.corpus.gutenberg.raw("whitman-leaves.txt").lower() ]
#########################


#########################
#extract words from string without punctuation
def book_punc(i):
  books[i] = re.sub("["+string.punctuation+"]", "", str(books[i])).split()
#########################
  
  
#########################  
#removing stop words 
def book_stopwords(i):
  tokenized_words = books[i] 
  stop_words = stopwords.words("english")
  books[i]=[word for word in tokenized_words if word not in stop_words]
#########################


#########################  
#Stemmer 
def book_stemmer(i):
  ps = PorterStemmer()
  example_words = books[i]
  new = []
  for w in example_words:
   new.append(ps.stem(w))
  books[i]=new
#########################
  
  
#########################
#number of words in each document
def book_docWords(i):
  documentW=len(books[i]) // 200   #counting the number of words per documents ?
  return int(documentW)
#########################
  

#########################
#make 200 random numbers for as a pointer to each word of a document.
#The goal is start reading from the pointer as a random number 
def book_randNum (documentWo,i):
  documentW = documentWo
  value = []
  #Preventing possible errors if each section of a book had less than 150 words
  if (documentW > 80):
    tguess = documentW - 80
    for i in range (200): #should be  200
      tmp=randint(0,tguess)
      value.append(tmp)
  else:
    tguess = documentW
    for i in range (200): #should be 200
      tmp=randint(0,tguess)
      if (i == 199):  # should be 200 - 1
        tmp=tmp-documentW
      value.append(tmp)
  return value
#########################
  

#########################
#make a list of 200 sublist *  contains 150 words (for each book, obviously!)
def book_listWord (documentWs ,values , i):
  value = values
  documentW = documentWs
  
  dataset = []
  datasetF = []

  res=books[i]
  for i in range (200):  # after testing should be 200 ~ number of sample documents for each book
    point = (i * documentW) + value[i]
    for x in range (80): #should be 150 ~ number of words in each sample
      dataset.append(res[point])
      point += 1
      #print(dataset) #check each line
      if len(dataset) == 80 :  #should be 150 ~ appending words to the dataset
        datasetF.append((list(dataset)))
        dataset.clear()      
        
  return list (datasetF)
#########################


#########################
#MAKE DATA FRAME FROM FINAL DATASET (which a list containst 7 list , 200 sublist and each sublist contains 150 words) ~
#### this data is for the whole books 
def book_lable(finList):
  booknamelist = ["Emma", "Paradise", "Ball", "Moby", "Parents" , "Stories" ,  "Leaves"]
  authornamelist = ["Austun", "Milton", "Chesterton", "Melville", "Edgeworth" , "Briant" , "Whitman"]
  column_names = ["BookName" , "AuthorName" , "Content"]
  df2 = pd.DataFrame(columns = column_names)

  for j in range (7): #always be 7 (7 list)
    for k in range (200):  #should be 200 after tests
      datas = { 
      "BookName": booknamelist[j] , "AuthorName": authornamelist[j] , "Content": (finList[j][k:k+1 ])
      #from finlist[j][k] to finlist[j][k+1]
         }
      df = pd.DataFrame.from_dict(datas)
      frames = [df]
      result = pd.concat(frames)
      df2 = df2.append(result)

  return df2
#########################


#########################
#The main function
bookGlist = []
bookFlist = []
values = []
column_names = ["BookName" , "AuthorName" , "Content"]
dataFrame_final = pd.DataFrame(columns = column_names)
for i in range(7):
  book_punc(i)
  book_stopwords(i)
  book_stemmer(i)
  documentW = book_docWords(i)
  values = book_randNum (documentW , i)
  bookGlist = book_listWord(documentW , values , i)
  bookFlist.append(bookGlist)
#########################
  
  
######################### 
print("DONE :: End of books process")
#########################


#########################
dataFrame_final = book_lable(bookFlist)  #labling books
dataFrame_final = dataFrame_final.reset_index() #make index - for the shuffling
dataFrame_finalshuff = dataFrame_final.reindex(np.random.permutation(dataFrame_final.index))  #shuffle dataframe
#########################


#########################
#Joint for dataFrame Content
dataFrame = pd.DataFrame(index=range(1400), columns=["Content", "AuthorName"])
for i in range(1400):
    dataFrame.at[i,"Content"] = " ".join(dataFrame_final.at[i, "Content"])
    dataFrame.at[i,"AuthorName"] = dataFrame_final.at[i,"AuthorName"]
    dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index)) 
#########################

##############################

print("\n\n","::::::>>>>>>> NAIVE <<<<<<<::::::")
#Naive + Vectorizer
X_train, X_test, y_train, y_test = train_test_split(dataFrame["Content"], dataFrame["AuthorName"],
                                              test_size=0.2,     random_state = 1)
count_vect = CountVectorizer()
fiter = count_vect.fit(X_train)
X_train_counts = fiter.transform(X_train)
tfidf_transformer = TfidfTransformer()
fiter2 = tfidf_transformer.fit(X_train_counts)
X_train_tfidf = fiter2.transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(">>> Naive Score:")
print(clf.score(fiter2.transform(fiter.transform(X_test)),y_test))    
    
    
######################    
"""import numpy as np
import sklearn.cluster
import distance

#words = "YOUR WORDS HERE".split(" ") #Replace this line
words = np.asarray(dataFrame['Content']) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))  """  
    
####################


"Clustering"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#X, y = make_blobs(n_samples=1400, centers=4, cluster_std=0.60, random_state=0)
#plt.scatter(X[:,0], X[:,1])
X=X_train_counts
y=dataFrame['AuthorName']

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203"""

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()