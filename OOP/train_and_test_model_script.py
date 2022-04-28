from predict_model import predict
from train_model import training
from predict_model import predict
import pandas as pd
import pickle

prediction_obj= predict()
train_obj= training()
folder_data= "./data/"

train_wav= True

# "train_wav= False" when you want to train the model with phidget data so use "sound222_FFT_only_training.py" for produce data
if train_wav== False:

    training_data= train_obj.fetch_train_dataset_from_pickle()

else:
   training_data= train_obj.fetch_train_dataset_from_wav()

model= train_obj.create_model(training_data)

# Add activities "activities=[<all_activities>]" accordingly to "sound222_subscriber.py" settings
activities= pickle.load(open(folder_data + "activity.pickle", 'rb'))

prediction= prediction_obj.predict_model(model, training_data, activities)

