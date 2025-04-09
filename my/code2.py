import json
import numpy as np
import pickle
import random
import nltk

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import warnings
import logging
import tensorflow as tf

# Suppress all TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO logs from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations logs

# Suppress Keras UserWarnings (input_shape/input_dim)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Set TensorFlow logger level to ERROR (suppress warnings and info)
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore", message=".*compiled metrics.*")
warnings.filterwarnings("ignore", message=".*Error in loading the saved optimizer state.*")
intd = json.loads(open(r'D:\batch\django\group_project\my\intents.json').read())
words = pickle.load(open(r'D:\batch\django\group_project\my\words.pkl','rb'))
classes = pickle.load(open(r'D:\batch\django\group_project\my\classes.pkl','rb'))
model = load_model(r'D:\batch\django\group_project\my\chatbott.keras')
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
lemmatizer = WordNetLemmatizer()



def clean_up_sentence (sentence):
    sentence_word=nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(w.lower()) for w in sentence_word]
    return sentence_word

def bag_words(sentence):
    bag_sentence = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in bag_sentence:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return(np.array(bag))

def predict_class(sentence):
    bow = bag_words(sentence)
    res = model.predict(np.array([bow]))[0]
    EROOR_THRESHOLD =0.25
    result= [[i,r] for i,r in enumerate(res) if r> EROOR_THRESHOLD ]
    result.sort(key=lambda x:x[1], reverse=True)
    result_list = []
    for r in result:
        result_list.append({"intent":classes[r[0]],'probability':str(r[1])})

    return result_list

def get_rep(intent_list,intend_j):
    tag = intent_list[0]['intent']
    list_of_intend = intend_j['intents']
    for i in list_of_intend:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result        

print('good to go')
while True:
    msg = input('enter your msg:  ')
    int = predict_class(msg)
    result = get_rep(int,intd)
    print('bot : ', result)


