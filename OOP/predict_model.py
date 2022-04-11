import paho.mqtt.client as mqtt
import pickle
import pandas as pd
from train_model import training

class predict:
   
    def __init__(self):
        model= training()
        activities= model.activities
    
    client = mqtt.Client()

    def load_model(self):
        model= model_to_load()
        return model   

    def predict_model(self, data, model):
        predict= model_to_run()
        return predict    
    
    # The callback for when the client receives a CONNACK response from the server.
    
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("test")

# The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        predictors = ["freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10","average_amplitude","activity"]
        frequencies= ["freq_1","freq_2","freq_3","freq_4","freq_5","freq_6","freq_7","freq_8","freq_9","freq_10"]
        data= msg.payload
        data_out= pd.Dataframe(columns=predictors)
        
        for rowIndex, row in data.iterrows():           
            temp_data= pd.DataFrame(row, columns=frequencies)
            temp_data["average_amplitude"]= temp_data.mean(axis= 1)
            data_out= data_out.append(temp_data, ignore_index=True)

        #Loading model
        print("Loading pre-trained model")
        loaded_model = self.load_model()        

        self.predict_model(data_out, loaded_model)

        print(f"topic = {msg.topic}, payload = {msg.payload}")

    # Client callback that is called when the client successfully connects to the broker.
    client.on_connect = on_connect
    # Client callback that is called when a message within the subscribed topics is published.
    client.on_message = on_message

    client.connect("84.238.56.157", 1883, 60)

    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a manual interface.
    client.loop_forever()