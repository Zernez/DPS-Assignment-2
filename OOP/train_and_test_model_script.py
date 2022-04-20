from predict_model import predict
from train_model import training
from predict_model import predict
import pandas as pd
import pickle

prediction_obj= predict()
train_obj= training()
folder_data= "./data/"

predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude","activity"]
training_data= pd.DataFrame(columns= predictors)

# Add activities "activities=[<all_activities>]" accordingly to "sound222_publisher.py" settings
activities= ["Computering","Vacuum_cleaning","Cooking"]
# Add activities "number_of_activities=[<number_of_activities>]" accordingly to "sound222_FFT_publisher.py" settings
quantity_of_activities= 3

train_wav= True

# "train_wav= False" when you want to train the model with phidget data so use "sound222_FFT_only_training.py" for produce data
if train_wav== False:

    data= pickle.load(open(folder_data + "data.pickle", 'rb')) 

    start_index= 0
    # Select here "quantity= <how_many_rows>" how many rows do you need (e.g. 1 second is 1 row, max 600 rows) accordingly to "sound222_publisher.py" settings
    quantity= 60
    index= quantity

    for activity in activities:
        training_data= training_data.append(train_obj.fetch_train_dataset_from_phidget(data[start_index:index], activity), ignore_index=True)
        start_index= index
        index+= quantity
else:
   training_data= train_obj.fetch_train_dataset_from_wav()

model= train_obj.create_model(training_data)

prediction= predict.predict_model(model, test_data, activities)