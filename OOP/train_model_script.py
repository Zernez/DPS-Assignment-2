from predict_model import predict
from train_model import training
import pandas as pd

result_obj= predict()
training_obj= training()

predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude","activity"]
training_data= pd.DataFrame(columns= predictors)

train_wav= True

# "train_wav= False" when you want to train the model with phidget data so use "sound222_FFT_only_training.py" for produce data
if train_wav== False:


    data= result_obj.data_out
    # Add activities "activities=[<all_activities>]" accordingly to "sound222_FFT_only_training.py" settings
    activities= ["Computering","Vacuum_cleaning","Cooking"]

    start_index= 0
    # Select here "quantity= <how_many_rows>" how many rows do you need (e.g. 1 second is 1 row, max 600 rows) accordingly to "sound222_FFT_only_training.py" settings
    quantity= 60
    index= quantity

    for activity in activities:
        training_data= training_data.append(training_obj.fetch_train_dataset_from_phidget(data[start_index:index], activity), ignore_index=True)
        start_index= index
        index+= quantity
        
else:
   training_data= training_obj.fetch_train_dataset_from_wav()



model= training_obj.create_model(training_data)