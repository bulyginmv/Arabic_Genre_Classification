# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:22:20 2019

@author: Misha
"""

import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
arabic_corpus = open('ARcorpusALL.txt', encoding='utf-8-sig')
test_corpus=open('test_X.txt', encoding='utf-8-sig')
test_corpus=test_corpus.read()
test_corpus=test_corpus.split('text*')
arabic_corpus=arabic_corpus.read()
arabic_corpus=re.split(r'تكست إد',arabic_corpus)
arabic_corpus.pop(0)
count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3))
X_train_counts = count_vect.fit_transform(arabic_corpus)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
genre_scores = open('annotations.txt','r',encoding='utf-8')
genre_scores = genre_scores.read()
genre_scores = genre_scores.split('\n')
y_train=[]
for score in genre_scores:
    y_train.append(score.split('\t'))
y_train=np.array(y_train)
X_new_counts = count_vect.transform(test_corpus)
X_new_tfidf = tf_transformer.transform(X_new_counts)
results=[]
clf = svm.LinearSVC()

for i in range (18):
    clf.fit(X_train_tf, y_train[:,i])
    results.append(clf.predict(X_new_tfidf))
results_final=np.array(results)
results_final=results_final.transpose()
k = open('resultsvmtr.txt','tw', encoding='utf-8')
for x in results_final:
    k.write('\t'.join([str(y) for y in x]))
    k.write('\n')
k.close()