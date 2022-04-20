from __future__ import division,print_function, absolute_import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import pickle
import pandas as pd
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import tensorflow as tf

class training:

    #Defining model and training it
    activities= {}
    folder_data= "./data/"
    folder_audio= "./data/audio/"   
    predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude"]
    data_out= pd.DataFrame(columns= predictors) 

    counter= 0
    sampler= []
    sample_rate= 1000
    sample_slice= sample_rate
    freq_band= 15
    freq_interested= int(sample_slice/freq_band)

    counter_train= 0
    train_activity_level= 60000
    activity= ''
      
    def fetch_train_dataset_from_wav(self):
        data_res= 44100
        freq_band= 15
        freq_interested= int((data_res/2)/freq_band)
        predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude","activity"]     
        train_data= pd.DataFrame(columns= predictors)

        for file in os.listdir(self.folder_audio):
            rate, data = wav.read(self.folder_audio + file)
            x= data[:, 0]

            i= data_res
            slice_x= []
        
            while (i< len (x)):
                slice_x.append(i-1)
                i+= data_res

            name= os.path.basename(file)
            name= name.replace('.wav', '')
            num = len(self.activities)
            self.activities[num]= name
            row_activ= [num+1]
        
            start_index= 0
            data_fft= []
            
            for index in slice_x:
                        
                temp_data= x [start_index:index]
                row_ampl= [np.mean(np.abs(temp_data), axis= 0)]
                temp_data_fft = np.abs(fft(temp_data).real)                
       
                i= freq_interested
                j= 1
                slice_y= []
                start_index_2= 1
                data_fft= [temp_data_fft[0]]
        
                while (i< len (temp_data_fft) and j <= freq_band):
                    slice_y.append(i-1)
                    i+= freq_interested
                    j+= 1 

                for index_2 in slice_y:
                    temp_mean= temp_data_fft [start_index_2:index_2]
                    data_fft.append(np.mean(temp_mean, axis= 0))
                    start_index_2= index_2
            
                data_fft.append(row_ampl[0])

                data_fft.append(row_activ[0])
    
                data_fft= pd.DataFrame(data_fft).T
                data_fft.columns= predictors
            
                train_data= train_data.append(data_fft, ignore_index=True)

                start_index= index
        
        return train_data

    def store_data_train(self, msg):

        if (self.counter_train>= self.train_activity_level or self.counter_train== 0):
            self.activity = input('Insert the name of THE activity for labeling or input \'x\' for exit.\n')
            self.counter_train= 0
            if (self.activity== 'x'):
                quit()  
            self.counter_train+= 1

        if (self.counter< (self.sample_slice)):
            self.counter+= 1
            self.sampler.append(msg)
        else:
            row_ampl= [np.mean(np.abs(self.sampler), axis= 0)]
            temp_data_fft = np.abs(fft(self.sampler).real)
            data_fft= [temp_data_fft[0]]
               
            i= self.freq_interested
            j= 1
            slice_y= []
            start_index= 1
        
            while (i< len (temp_data_fft) and j <= self.freq_band):
                slice_y.append(i-1)
                i+= self.freq_interested
                j+= 1 

            for index in slice_y:
                temp_mean= temp_data_fft [start_index:index]
                data_fft.append(np.mean(temp_mean, axis= 0))
                start_index= index
            
            data_fft.append(row_ampl[0])
            self.sampler.clear()
            self.counter= 0

            data_fft= pd.DataFrame(data_fft).T
            data_fft.columns= self.predictors
            data_fft ["activity"]= self.activity          
            self.data_out= self.data_out.append(data_fft, ignore_index=True)

            #Storage max 10 min e.g. 600 rows of 1 second each
        if (self.data_out.shape[0]> 600):
            self.data_out = self.data_out.iloc[1: , :]

        pickle.dump(self.data_out,open(self.folder_data + "data.pickle", 'wb'))
        return 

    def fetch_train_dataset_from_phidget(self):
        train_data= pickle.load(open(self.folder_data + "data.pickle", 'rb'))   
        return train_data

    def create_model(self, data):
        # build model
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # tf.keras.layers.Flatten(input_shape(11,100)),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=10)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        print('\nTest accuracy:', test_acc)
        pickle.dump(model,open(self.folder_data + "model.pickle", 'wb'))
        return model
