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

class training:

    #Defining model and training it
    activities= {}
    folder_data= "./data/"
    folder_audio= "./data/audio/"
    data_res= 44100
    freq_band= 15
    freq_interested= int((data_res/2)/freq_band)


    def fetch_train_dataset(self, activities):

        predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude","activity"]    
        data_out= pd.DataFrame(columns= predictors)

        for file in os.listdir(self.folder_audio):
            rate, data = wav.read(self.folder_audio + file)
            x= data[:, 0]

            i= self.data_res
            slice_x= []
        
            while (i< len (x)):
                slice_x.append(i-1)
                i+= self.data_res

            name= os.path.basename(file)
            name= name.replace('.wav', '')
            num = len(activities)
            activities[num]= name
            row_activ= [num+1]
        
            start_index= 0
            data_fft= []
            
            for index in slice_x:
                        
                temp_data= x [start_index:index]
                row_ampl= [np.mean(temp_data, axis= 0)]
                temp_data_fft = np.abs(fft(temp_data).real)                
       
                i= self.freq_interested
                j= 1
                slice_y= []
                start_index_2= 1
                data_fft= [temp_data_fft[0]]
        
                while (i< len (temp_data_fft) and j <= self.freq_band):
                    slice_y.append(i-1)
                    i+= self.freq_interested
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

    def model(activities):
        model = Training_MachineLearning_Algo()
        return model

    model = model(activities)
    pickle.dump(model,open(folder_data + "model.pickle", 'wb'))
    print("Training Finished!")