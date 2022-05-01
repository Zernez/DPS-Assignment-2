import paho.mqtt.client as mqtt
from predict_model import predict 
from train_model import training 
import pandas as pd
import pickle

# Select True if you want to train a model
training_selector= True
folder_data= "./data/"
data_type= "shimmer"
predictors_shim= ["x", "y", "z"]

prediction_obj= predict()
train_obj= training()
data_out= pd.DataFrame()

# The callback for when the client receives a CONNACK response from the server.
    
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("shimmer")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):

    if (training_selector== True):
        train_obj.store_data_train(msg.payload, data_type)
    else:
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
        data_acc.columns= predictors_shim 

        data_out= pickle.load(open(folder_data + "shimmer.pickle", 'rb')) 

        data_out= data_out.append(data_acc, ignore_index=True)            

        #Storage max 10 min e.g. 600 rows of 1 second each
        if (data_out.shape[0]> 600):
            data_out = data_out.iloc[1: , :]

        pickle.dump(data_out,open(folder_data + "shimmer.pickle", 'wb'))

#    print(f"topic = {msg.topic}, payload = {msg.payload}")

client = mqtt.Client()
    # Client callback that is called when the client successfully connects to the broker.
client.on_connect = on_connect
    # Client callback that is called when a message within the subscribed topics is published.
client.on_message = on_message

client.connect("84.238.56.157", 1883, 60)

    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a manual interface.
client.loop_forever()