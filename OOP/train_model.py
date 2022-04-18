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

    def fetch_train_dataset_from_wav(self):
        data_res= 44100
        freq_band= 15
        freq_interested= int((data_res/2)/freq_band)
        predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude","activity"]
     
        data_out= pd.DataFrame(columns= predictors)

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
            
                data_fft.append(row_ampl)

                data_fft.append(row_activ)
    
                data_fft= pd.DataFrame(data_fft).T
                data_fft.columns= predictors
            
                data_out= data_out.append(data_fft, ignore_index=True)

                start_index= index
         
        return data_out

    def fetch_train_dataset_from_phidget(self, data, activity, lenght):
        # Select here how many rows do you need "data.iloc[0:<How_many_row do you want>]" (e.g. 1 second is 1 row, max 600 rows)  
        data= data [0:lenght]
        data ["activity"]= activity     
        return data

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
