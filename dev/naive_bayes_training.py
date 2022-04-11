from __future__ import division,print_function, absolute_import
from sklearn.datasets import fetch_20newsgroups #built-in dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import pickle
import pandas as pd
from scipy.io import wavfile as wav
from scipy.fftpack import fft

#Defining model and training it
categories = ["Cleaning vacum machine","Cleaning dishes","Listen music","Watching TV","Sleep","General acticvity"] 
folder_data= "./data/"
folder_audio= "'./data/audio/"

def fetch_train_dataset(categories):
    for file in os.listdir(folder_audio):
        frequencies = ["freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10", "category"] 
        rate, data = wav.read(file)
        fft_out = fft(data)
        name= os.path.basename(file)
        name.replace('.wav', '')
        
        #data is dataframe?

    return data

def bag_of_words(categories):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(fetch_train_dataset(categories).data)
    pickle.dump(count_vect.vocabulary_, open("behaviour.pickle", 'wb'))
    return X_train_counts

def tf_idf(categories):
    tf_transformer = TfidfTransformer()
    return (tf_transformer,tf_transformer.fit_transform(bag_of_words(categories)))

def model(categories):
    clf = MultinomialNB().fit(tf_idf(categories)[1], fetch_train_dataset(categories).target)
    return clf

model = model(categories)
pickle.dump(model,open("model.pickle", 'wb'))
print("Training Finished!")
#Training Finished Here