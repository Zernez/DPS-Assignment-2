from __future__ import division,print_function, absolute_import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import pickle
import pandas as pd
from scipy.io import wavfile as wav
from scipy.fftpack import fft

class training:

    #Defining model and training it
    activities= {}
    folder_data= "./data/"
    folder_audio= "'./data/audio/"
    data_res= 100


    def fetch_train_dataset(self, activities):
        for file in os.listdir(self.folder_audio):
            predictors = ["freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","average_amplitude","activity"] 
            rate, data = wav.read(file)
            name= os.path.basename(file)
            name.replace('.wav', '')

            start_index= 0
            data_out= pd.Dataframe(columns=predictors)
            for rowIndex, row in data.iterrows():
                
                if (rowIndex% self.data_res== 0):
                    temp_data= data [start_index:rowIndex]
                    temp_data["average_amplitude"]= temp_data.mean(axis= 1)
                    temp_data= pd.DataFrame(fft(temp_data), columns=predictors)
                    num = len(self.activities)
                    self.activities[num]= name
                    temp_data["activity"]= num       

                    start_index= rowIndex
        return data_out

    def model(activities):
        model = Training_MachineLearning_Algo()
        return model

    model = model(activities)
    pickle.dump(model,open("model.pickle", 'wb'))
    print("Training Finished!")