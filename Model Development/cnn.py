from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.layers import LSTM,Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras import optimizers
from keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder

dataVal_Fake_Real=pd.read_csv('train.csv')

texts=[]
texts=dataVal_Fake_Real['Statement']
label=dataVal_Fake_Real['Label']


from DataPrep.Clean_Texts import clean_text
X=texts.map(lambda x: clean_text(str(x)))

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))


training_size=int(0.8*X.shape[0])
X_train=X[:training_size]
y_train=y[:training_size]
X_test=X[training_size:]
y_test=y[training_size:]


#Max no of Vocab
vocabulary_size = 400000

time_step=300
embedding_size=100

#Tokenizing texts
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences_train= tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(sequences_train, maxlen=time_step,padding='post')


vocab_size=len(tokenizer.word_index)+1

#Reading Glove
f = open('glove.6B.100d.txt',encoding='utf-8')
embeddings={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
f.close()


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape[0],embedding_matrix.shape[1])


sequences_test= tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(sequences_test, maxlen=time_step,padding='post')



# Convolution
filter_length = 3
nb_filter = 128


#word embedding
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=time_step,
                    weights=[embedding_matrix],trainable=False))

#add the convolutional layer
model.add(Conv1D(filters=nb_filter,
                        kernel_size=filter_length,
                        activation='relu'))

#use the 1D MaxPooling for Pooling layer 
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())


model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

#model trainning
model.fit(X_train,y_train,batch_size=64,epochs=10)

#save the model
print("Saving Model...")
model_name = 'Model_CNN.h5'
model.save(model_name)

score=model.evaluate(X_test,y_test,verbose=1)
print('acc: '+str(score[1]))

#show test report
from sklearn.metrics import precision_recall_fscore_support,classification_report
y_pred=model.predict_classes(X_test)
print('Classification report:\n',classification_report(y_test,y_pred))
'''
model = load_model('Model_CNN_FR_2.h5')
model.name='Model_CNN_FR_2.h5'
'''

from sklearn.metrics import precision_recall_fscore_support, classification_report
y_pred=model.predict_classes(X_test)
print('Classification Report: '+classification_report(y_test, y_pred))
