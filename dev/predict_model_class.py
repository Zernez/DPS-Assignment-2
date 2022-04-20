import paho.mqtt.client as mqtt
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

class predict:

    #To hard-code activities here   
    activities= ["Computering","Vacuum_cleaning","Cooking"]    
    folder_data= "./data/"    
    client = mqtt.Client()
    predictors = ["freq_0","freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","freq_11","freq_12","freq_13","freq_14","freq_15","average_amplitude"]
    data_out= pd.DataFrame(columns= predictors)   

    def load_model(self):
        model= pickle.load(open(self.folder_data + "model.pickle", 'rb'))
        return model   

    def predict_model(self, model, test_data, categories):
        probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
        predictions = probability_model.predict(test_data)

        predicted_labels = []
        predicted_cat = []
        for pred in predictions:
            lab = np.argmax(pred)
            predicted_labels.append(lab)
            predicted_cat.append(categories[lab])

        return predicted_cat  

    # The callback for when the client receives a CONNACK response from the server.
    
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("test")

# The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        data_fft= pd.DataFrame(msg).T
        data_fft.columns= self.predictors          
        self.data_out= self.data_out.append(data_fft, ignore_index=True)

        #Storage max 10 min e.g. 600 rows of 1 second each
        if (self.data_out.shape[0]> 600):
            self.data_out = self.data_out.iloc[1: , :]           

        #Loading model
        print("Loading pre-trained model")
        loaded_model = self.load_model()
       
        # Select here how many rows do you need "self.data_out.iloc[0:<How_many_row do you want>] (e.g. 1 second is 1 row, max 600 rows)    
        predict_data= self.data_out [0:60]   

        self.predict_model(loaded_model, predict_data, self.activities)

        print(f"topic = {msg.topic}, payload = {msg.payload}")

    # Client callback that is called when the client successfully connects to the broker.
    client.on_connect = on_connect
    # Client callback that is called when a message within the subscribed topics is published.
    client.on_message = on_message

    client.connect("84.238.56.157", 1883, 60)

    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a manual interface.
    client.loop_forever()