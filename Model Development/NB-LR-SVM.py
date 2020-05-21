#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:21:34 2019

@author: WZE
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:20:30 2019

@author: WZE
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:17:19 2019

@author: WZE
"""
import pickle 
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import confusion_matrix, f1_score, classification_report

import math

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma','needn', 'shan','shouldn', 'won']
ps = PorterStemmer()
train = pd.read_csv('train.csv')   

test = pd.read_csv('test.csv')


#Data preprocessing
def reviewCleaning(review): 
    review = str(review)
    clean = review.lower()
    clean = re.sub('[^a-zA-Z?!]', ' ',clean).split()
    lemmatizer = WordNetLemmatizer()
    result = []
    for i in range(0, len(clean)):
        if clean[i] not in stopwords:
            result.append(lemmatizer.lemmatize(clean[i]))
    return (" ".join(result))


cleanedTrain = []
for i in range(len(train["Statement"])):
    ##if (reviewCleaning(train["Phrase"][i])) != "" :
    cleanedTrain.append(reviewCleaning(train["Statement"][i]))
train["Cleaned_Phrase"] = cleanedTrain
train.drop(['Statement'],axis=1,inplace=True)
#print(cleanedTrain)
cleanedTest = []
for i in range(len(test["Statement"])):
    cleanedTest.append(reviewCleaning(test["Statement"][i]))
test["Cleaned_Phrase"] = cleanedTest
test_text = test["Statement"]
test.drop(['Statement'],axis=1,inplace=True)


#Bag-of-word or TF-IDF

cv = TfidfVectorizer(max_features = 5000)
#cv = CountVectorizer(max_features = 5000)


#Train-Test Split

cleanedAll = cleanedTrain[:]
cleanedAll.extend(cleanedTest)
X_all = cv.fit_transform(cleanedAll)
X_train = X_all[:len(cleanedTrain)]
X_test = X_all[len(cleanedTrain):]
y_train = train["Label"]
y_test = test["Label"]
#print(y_train)
#print(y_test)

'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x__train, y, test_size = 0.30, random_state = 0)
'''

# Evaluation: Accuracy, Percision, Recall, F-measure
def f_measure(sys_l, ans_l):
    d = {}
    for i in range(len(sys_l)):
        sys = sys_l[i]
        ans = ans_l[i]
        l = d.get(ans,[0,0,0])
        if ans == sys:
            for j in range(len(l)):
                l[j] += 1;
            d[ans] = l
        else:
            l[1] += 1
            d[ans] = l
            l = d.get(sys,[0,0,0])
            l[2] += 1
            d[sys] = l
            
    f_total = 0
    for k,v in d.items():
        if v[0]==0:
            print(0)
            continue
        recall = v[0]/v[1]
        precision = v[0]/v[2]
        f_measure = 2/((1/recall) + (1/precision))

        print(precision)
        print(recall)
        print(f_measure)
        f_total += f_measure
    evalu = f_total/len(d.keys())
    print(evalu)


#Models NB, LR, SVM
'''
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
'''
from sklearn import svm
classifier = svm.SVC()
classifier.fit(X_train, y_train)


#Save model
filename = 'SVM.sav'
pickle.dump(classifier, open(filename, 'wb'))


#Get test Results
y_pred = classifier.predict(X_test)
num = len(y_test)
same = 0
pred_result = []
for i in range(len(y_test)):
    pred_result.append(str(y_pred[i]).upper())
    if y_test[i] == str(y_pred[i]).upper():

        same += 1
print("Acc: ",same/num)
f_measure(pred_result, y_test)

print("f1 ",f1_score(y_test,pred_result, average=None))



