from predict_model import predict
from train_model import training
from predict_model import predict
import pandas as pd
import pickle
import numpy as np

prediction_obj= predict()
train_obj= training()
folder_data= "./data/"

train_wav= False

# "train_wav= False" when you want to train the model with phidget data so use "sound222_FFT_only_training.py" for produce data
if train_wav== False:
#    training_data= train_obj.fetch_train_dataset_from_pickle()
#   training_data= pickle.load(open(folder_data + "data.pickle", 'rb'))
    training_data= pickle.load(open(folder_data + "shimmer.pickle", 'rb'))
else:
   training_data= train_obj.fetch_train_dataset_from_wav()

pd.set_option('display.max_rows', None)

print (training_data)

#model= train_obj.create_model(training_data)

# Add activities "activities=[<all_activities>]" accordingly to "sound222_subscriber.py" settings
#activities= pickle.load(open(folder_data + "activity.pickle", 'rb'))

#prediction= prediction_obj.predict_model(model, training_data, activities)

