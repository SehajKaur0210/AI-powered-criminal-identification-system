import random
import json
import tensorflow as tf
import numpy as np
import keras
import nltk
import pickle

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intend_file = json.loads(open(r'C:\Users\sehaj\OneDrive\Desktop\my\intents.json').read())

words =[]
classes =[]
documents =[]
ignoreLetters = ['.',',','!','?']

for intd in intend_file['intents']:
    for pattern in intd['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList,intd['tag']))
        if intd['tag'] not in classes :
            classes.append(intd['tag'])

#print(documents)     

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreLetters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

#print(len(words),'unique',words)
#print(len(documents),'documents')
#print(len(classes),classes)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
o_empty = [0]* len(classes)

for doc in documents :
    bag=[]
    w_pattern = doc[0]
    w_pattern = [lemmatizer.lemmatize(w.lower())for w in w_pattern]
    for w in words:
        bag.append(1) if w in w_pattern else bag.append(0)

    o_row = list(o_empty)
    o_row[classes.index(doc[1])]=1
    training.append(bag+ o_row)

random.shuffle(training)
training= np.array(training)
#print(training)

trainx = training[:,:len(words)]
trainy = training[:,len(words):]

#print(trainx,trainy)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape = (len(trainx[0]),),activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainy[0]),activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01 , momentum=0.9 ,nesterov =True)
model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist = model.fit(np.array(trainx),np.array(trainy),batch_size=5, epochs =200,verbose =1)
model.save('chatbott.keras',hist)
print('done')