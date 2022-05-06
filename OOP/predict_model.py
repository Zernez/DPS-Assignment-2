import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.fft import rfft, rfftfreq
import time
import joblib
import time
import os
import os.path
from timeit import Timer
# tf.compat.v1.disable_eager_execution()

class predict:

    #To hard-code activities here   
    activities= {}
    folder_data= "./data/"    
    predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","average_amplitude"]
    predictors_shim= ["x", "y", "z"]
    pred_shim= ""
    trigger_shim_treshold= 500
    data_out= pd.DataFrame(columns= predictors) 


    counter= 0
    sampler= []
    sample_rate= 1000
    duration= 1
    N= sample_rate*duration
    sample_slice= sample_rate
    freq_band= 10
    freq_interested= int(sample_slice/freq_band)
    model = None
    immutable= False
    timing= Timer()

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
        for pred in predictions:
            lab = np.argmax(pred)
            predicted_cat.append(categories[lab])

        pre = max(set(predicted_cat),key=predicted_cat.count)

        return predicted_cat

    def store_data_prediction(self, msg, data_type):  
        
        if (data_type== "sound"):
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

                data_fft.columns= self.predictors
#                sd = data_fft.reset_index(drop=True)
#                shimmer_data= pickle.load(open(self.folder_data + "shimmer.pickle", 'rb'))
#                sh = shimmer_data.tail(1).reset_index(drop=True)
#                data_fft_shim= pd.concat([sd, sh], axis=1)
                # print(data_fft_shim)                   
                self.data_out= self.data_out.append(data_fft, ignore_index=True)

                #Storage max 10 min e.g. 600 rows of 1 second each
                if (self.data_out.shape[0]> 10):
                    self.data_out = self.data_out.iloc[1: , :] 
                print(self.data_out)
                pickle.dump(self.data_out,open(self.folder_data + "real_time.pickle", 'wb'))
        
        elif (data_type== "shimmer"):
            data_acc= []
            temp= ""
            msg= str(msg.pyaload)
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

            data_out= pickle.load(open(self.folder_data + "shimmer.pickle", 'rb'))

            data_out= data_out.append(data_acc, ignore_index=True)

            #Storage max 10 min e.g. 600 rows of 1 second each
            if (data_out.shape[0]> 10):
                data_out = data_out.iloc[1: , :]

            pickle.dump(data_out,open(self.folder_data + "shimmer.pickle", 'wb'))

            if (self.pred_shim== "Going Down"):
                self.timing.stop()                 
                self.timing.cancel()                

            if (data_out.shape[0]< 2):
                self.pred_shim= "Steady"
            else:
                if (abs(data_out.loc[-1, "z"] - data_out.loc[-2, "z"])> self.trigger_shim_treshold):
                    self.pred_shim= "Going Down"
                    self.timing.start()
                elif (abs(data_out.loc[-1, "x"] - data_out.loc[-2, "x"])> self.trigger_shim_treshold or abs(data_out.loc[-1, "y"] - data_out.loc[-2, "y"])> self.trigger_shim_treshold):

                    if (os.path.isfile(self.folder_data + "prediction_result.pickle")):
                        activity_predicted= pickle.load(open(self.folder_data + "prediction_result.pickle", 'rb'))                                         
                        if (activity_predicted== "Help_request"):
                            self.pred_shim= "Alarm"
                            self.immutable= True
                        else:
                            self.pred_shim= "Going Down"                            
                    else:
                        self.pred_shim= "Going Down"                    
                else:
                    self.pred_shim= "Steady"                    
            
            pickle.dump(data_out,open(self.folder_data + "shimmer_prediction.pickle", 'wb'))

        else:
            print ("No valid data")

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
