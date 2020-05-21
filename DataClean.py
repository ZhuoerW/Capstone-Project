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

def cleaned():
    cleanedTrain = []
    for i in range(len(train["Statement"])):
        ##if (reviewCleaning(train["Phrase"][i])) != "" :
        cleanedTrain.append(reviewCleaning(train["Statement"][i]))
    train["Cleaned_Phrase"] = cleanedTrain
    #train.drop(['Statement'],axis=1,inplace=True)
    cleanedTest = []
    for i in range(len(test["Statement"])):
        cleanedTest.append(reviewCleaning(test["Statement"][i]))
    test["Cleaned_Phrase"] = cleanedTest
    test_text = test["Statement"]
    #test.drop(['Statement'],axis=1,inplace=True)


    cv = TfidfVectorizer(max_features = 5000)
    cleanedAll = cleanedTrain[:]
    cleanedAll.extend(cleanedTest)

    return cleanedAll


