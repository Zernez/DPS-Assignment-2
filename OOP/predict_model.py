import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.fft import fft
import time
import joblib
import time
import os
# tf.compat.v1.disable_eager_execution()
class predict:

    #To hard-code activities here   
    activities= {}
    folder_data= "./data/"    
    predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude"]
    data_out= pd.DataFrame(columns= predictors) 

    counter= 0
    sampler= []
    sample_rate= 1000
    sample_slice= sample_rate
    freq_band= 15
    freq_interested= int(sample_slice/freq_band)
    model = None
    def load_model(self):
        self.model= pickle.load(open(self.folder_data + "model.pickle", 'rb'))
        return self.model   

    def predict_model(self, model, test_data, categories):
        # print(test_data)
        probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
        test_data = np.asarray(test_data).astype('float32')
        predictions = probability_model.predict(test_data)
        predicted_cat = []
        print(predictions)
        for pred in predictions:
            lab = np.argmax(pred)
            print(pred,lab)
            predicted_cat.append(categories[lab])

        pre = max(set(predicted_cat),key=predicted_cat.count)
        # pre = predicted_cat[4]
        # print(pre)
        return pre

    def store_data_prediction(self, msg, data_type):  
        
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
            sd = data_fft.reset_index(drop=True)
            shimmer_data= pickle.load(open(self.folder_data + "shimmer.pickle", 'rb'))
            sh = shimmer_data.tail(1).reset_index(drop=True)
            data_fft_shim= pd.concat([sd, sh], axis=1)
            # print(data_fft_shim)                   
            self.data_out= self.data_out.append(data_fft, ignore_index=True)

            #Storage max 10 min e.g. 600 rows of 1 second each
            if (self.data_out.shape[0]> 10):
                self.data_out = self.data_out.iloc[1: , :] 
            print(self.data_out)
            pickle.dump(self.data_out,open(self.folder_data + "real_time.pickle", 'wb'))
        return

    def real_time_pred(self):

        #Loading model
        print("Loading pre-trained model")
        loaded_model = self.load_model()
       
        # Select here how many rows do you need "predict_data [0:<How_many_row do you want>] (e.g. 1 second is 1 row, max 600 rows)
        
        print(time.ctime(os.path.getmtime(self.folder_data+"real_time.pickle")),' : ', end='')
        predict_data= pickle.load(open(self.folder_data + "real_time.pickle", 'rb')).tail(3) 
        # predict_data= predict_data.tail(60)
        # print(predict_data)
        self.activities = pickle.load(open(self.folder_data + "activity.pickle", 'rb'))
        prediction= self.predict_model(loaded_model, predict_data, self.activities)
        # prediction = 1
        return prediction

