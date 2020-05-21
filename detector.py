from keras.models import load_model
from keras.layers import Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import math
import csv
import DataClean
from flask import Flask, render_template, request, session, url_for, redirect
import os
import sys
import search 
from sklearn.preprocessing import LabelEncoder


#run the search program and diagnosis program
def pred_result():
    vocabulary_size = 400000
    time_step=300
    embedding_size=100

    # fit the input information with the trained model
    predset=pd.read_csv('check.csv')
    texts=[]
    train = pd.read_csv('train.csv')

    texts=predset['text'].astype(str)
    text_ml = texts
    from DataPrep.Clean_Texts import clean_text
    texts=texts.map(lambda x: clean_text(x))

    tokenizer_pred=Tokenizer(num_words=vocabulary_size)
    tokenizer_pred.fit_on_texts(texts)
    encoded_pred=tokenizer_pred.texts_to_sequences(texts)
    #print(encoded_docs)
    vocab_size_pred = len(tokenizer_pred.word_index) + 1

    X_pred = sequence.pad_sequences(encoded_pred, maxlen=time_step, padding='post')

    #Load models 
    model1 = load_model('Model_CNN.h5')
    model2 = pickle.load(open("NB.sav", 'rb'))
    model3 = pickle.load(open("Logistic.sav", 'rb'))
    model4 = pickle.load(open("SVM.sav", 'rb'))
    y_pred_1=model1.predict(X_pred)
    cv = TfidfVectorizer(max_features = 5000)
    cleaned = DataClean.cleaned()
    X_all = cv.fit_transform(cleaned) 
    arr_texts = cv.transform(text_ml)
    model2 = pickle.load(open("NB.sav", 'rb'))
    y_pred_2=model2.predict(arr_texts)
    y_pred_3=model3.predict(arr_texts)
    y_pred_4=model4.predict(arr_texts)
    count = []

    #count prediction result and get the reliability index
    if math.ceil(y_pred_1[0][0]) == 1:
    	count.append(1)
    if y_pred_2[0] == 1:
    	count.append(1)
    if y_pred_3[0] == 1:
    	count.append(1)
    if y_pred_4[0] == 1:
    	count.append(1)
    return count

#route for form post and send back diagnosis results
app = Flask(__name__)
@app.route('/check', methods=['GET','POST'])
def check():
    data = {}
    type = request.form['type']
    #url = os.system('python search.py '+content)
    if type == "url":
        url = request.form['content']
        data["content"] = url
        content = search.getUrlContent(url).title
        print("here is the content", content)
    else: 
        content = request.form['content']
        data["content"] = content
    csvfile = open("check.csv","w") 
    writer = csv.writer(csvfile)
    writer.writerow(["text"])
    writer.writerow([content])
    csvfile.close()

    result  = search.crawler(content)
    index = pred_result()
    data["index"] = index
    data["result"] = result
    if data:
         return render_template('index.html',data=data)

#route for the index page
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run()
