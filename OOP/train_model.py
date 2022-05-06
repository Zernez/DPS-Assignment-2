from __future__ import division,print_function, absolute_import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import pickle
import pandas as pd
from scipy.io import wavfile as wav
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import numpy as np
import joblib
from predict_model import predict
import matplotlib.pyplot as plt

class training:

    #Defining model and training it
    activities= {}
    folder_data= "./data/"
    folder_audio= "./data/audio/"   
    predictors_audio = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5", "average_amplitude"]
    predictors_shim = ["x", "y", "z"]
    # Set data type!
    data_type= "sound"
    if (data_type== "shimmer"): 
        data_out= pd.DataFrame(columns= predictors_shim) 
    elif (data_type== "sound"): 
        data_out= pd.DataFrame(columns= predictors_audio) 

    counter= 0
    sampler= []
    sample_rate= 1000
    duration= 1
    N= sample_rate*duration
    sample_slice= sample_rate
    freq_band= 10
    freq_interested= int(sample_slice/freq_band)

    counter_train= 0
    train_activity_level= 55000
    train_activity_level_shim= 200   
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

    def store_data_train(self, msg, data_type):

        if (data_type== "shimmer"):
            
            if (self.counter_train>= self.train_activity_level_shim or self.counter_train== 0):
                self.activity = input('Insert the name of THE activity for labeling or input \'x\' for exit or \'del\' for exit and delete train data\n')
                self.counter_train= 0
                if (self.activity== 'x'):
                    self.counter_train= 0
                    quit()
                if (self.activity== 'del'):
                    empty= pickle.load(open(self.folder_data + "shimmer.pickle", 'rb'))
                    empty= pd.DataFrame()  
                    pickle.dump(empty,open(self.folder_data + "shimmer.pickle", 'wb'))
                    self.counter_train= 0                
                    quit()    
            
            self.counter_train+= 1

            data_acc= []
            temp= ""
            msg= str(msg)
            msg = msg.replace("b", '')
            msg = msg.replace("'", '')
            count= len(msg)

            for char in msg:               
                temp+= char
                
                if (" " in char):
                    data_acc.append(int(temp))  
                    temp= ""              
                
                count-=1                    
                
                if count== 0:
                    data_acc.append(int(temp))
                    temp= ""
                    continue

            data_acc= pd.DataFrame(data_acc).T
            data_acc.columns= self.predictors_shim           
            data_acc ["activity"]= self.activity

            self.data_out= self.data_out.append(data_acc, ignore_index=True)            

            if (self.data_out.shape[0]> 20000):
                self.data_out = self.data_out.iloc[1: , :]

            pickle.dump(self.data_out,open(self.folder_data + "shimmer.pickle", 'wb'))

        elif (data_type== "sound"):
            if (self.counter_train>= self.train_activity_level or self.counter_train== 0):
                self.activity = input('Insert the name of THE activity for labeling or input \'x\' for exit or \'del\' for exit and delete train data\n')
                self.counter_train= 0
                if (self.activity== 'x'):
                    self.counter_train= 0
                    quit()
                if (self.activity== 'del'):
                    empty= pickle.load(open(self.folder_data + "definitive_audio.pickle", 'rb'))
                    empty= pd.DataFrame()  
                    pickle.dump(empty,open(self.folder_data + "definitive_audio.pickle", 'wb'))   
                    self.counter_train= 0             
                    quit()    
            
            self.counter_train+= 1

            if (self.sample_slice> self.counter):
                self.counter+= 1
                self.sampler.append(msg)
            else:
                row_ampl= [np.mean(self.sampler, axis= 0)]
                temp_data_fft = np.abs(rfft(self.sampler))
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
                data_fft.columns= self.predictors_audio
                data_fft ["activity"]= self.activity
                if (self.counter_train> 30000 and self.counter_train< 51000):
                     print ("Capturing!")     
                     self.data_out= self.data_out.append(data_fft, ignore_index=True)

                #Storage max 10 min e.g. 600 rows of 1 second each
                if (self.data_out.shape[0]> 200):
                    self.data_out = self.data_out.iloc[1: , :]

                pickle.dump(self.data_out,open(self.folder_data + "definitive_audio.pickle", 'wb'))
            
#            data= pickle.load(open(self.folder_data + "data.pickle", 'rb'))
#            print (data)
        return 

    def fetch_train_dataset_from_pickle(self):
        train_data= pickle.load(open(self.folder_data + "definitive_audio.pickle", 'rb'))
        # train_data = train_data.reset_index()

        for index,row in train_data.iterrows():  
            num= len (self.activities)
            name= row["activity"]
            if name not in self.activities.values():
                self.activities [num]= name

        inv_map = {v: k for k, v in self.activities.items()}

        for index, row in train_data.iterrows():
           train_data.at[index, 'activity'] = inv_map [row["activity"]]
        print(self.activities)
        pickle.dump(self.activities,open(self.folder_data + "activity.pickle", 'wb'))
        return train_data

    def fetch_shimmer_data_from_pickle(self):
        shimmer_data = pickle.load(open(self.folder_data + "shimmer.pickle", 'rb'))
        shimmer_data = shimmer_data.reset_index()
        for index,row in shimmer_data.iterrows():  
            num= len (self.activities)
            name= row["activity"]
            if name not in self.activities.values():
                self.activities [num]= name

        inv_map = {v: k for k, v in self.activities.items()}
        for index, row in shimmer_data.iterrows():
           shimmer_data.at[index, 'activity'] = inv_map [row["activity"]]
        return shimmer_data

    def combine_data(self):
        sound_data = self.fetch_train_dataset_from_pickle()
        shimmer_data = self.fetch_shimmer_data_from_pickle()
        return sound_data

    def save_model(self, model, filepath):
        pickle.dump(model, open('./data/model.pickle', 'wb'))


    def create_model(self):
        df = self.fetch_train_dataset_from_pickle()
        # df = df.drop(df[df.activity == 6].index)
        
        #df.loc[df['activity'] == 7, 'activity'] = 6
        #print(tr_data)
        

#        df= df.drop("freq_2", axis=1)
#        df= df.drop("freq_1", axis=1)         
#        df = df.drop(df[df.activity == 7].index)
#        df = df.drop(df[df.activity == 5].index)
#        df = df.drop(df[df.activity == 4].index)

        print(df)

        df.loc[df['activity'] == 6, 'activity'] = 4
#        df.loc[df['activity'] == 5, 'activity'] = 3
        #1 build model
        model = tf.keras.Sequential([
        #tf.keras.layers.Flatten(input_shape=(21,100)),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(8)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
#        df = self.fetch_train_dataset_from_pickle()

        # print(df)

        tra_data = np.asarray(df.drop(['activity'], axis=1))
        tra_label = np.asarray(df['activity'])

        print(tra_data)
        print(tra_label)

        (tra_data, tra_label) = tra_data.astype('float32'), tra_label.astype(int)
        train_data, test_data, train_labels, test_labels = train_test_split(tra_data, tra_label, test_size=0.2)

        model.fit(train_data, train_labels, epochs=100)

        test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

        # print(model.predict())
        print('\nTest accuracy:', test_acc)

        pickle.dump(model, open('./data/model.pickle', 'wb'))

        pr = predict()
        categories = self.activities
        prediction = pr.predict_model(model, test_data, categories)

        print("Confusion Matrix:")
        print(confusion_matrix([categories[k] for k in test_labels], prediction, labels=list(categories.values())))
        pickle.dump(model, open('./data/model.pickle', 'wb'))

        return model