import paho.mqtt.client as mqtt
import pickle

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("test")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print("Starting ML predictions.")
    X_new_counts = count_vect.transform([msg])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = load_model.predict(X_new_tfidf)
    print(msg+" => "+fetch_train_dataset(categories).target_names[predicted[0]])

    print(f"topic = {msg.topic}, payload = {msg.payload}")

client = mqtt.Client()

#Loading model and vocab
print("Loading pre-trained model")
vocabulary_to_load = pickle.load(open("behaviour.pickle", 'rb'))
count_vect = CountVectorizer(vocabulary=vocabulary_to_load)
load_model = pickle.load(open("model.pickle", 'rb'))
count_vect._validate_vocabulary()
tfidf_transformer = tf_idf(categories)[0]

# Client callback that is called when the client successfully connects to the broker.
client.on_connect = on_connect
# Client callback that is called when a message within the subscribed topics is published.
client.on_message = on_message

client.connect("84.238.56.157", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a manual interface.
client.loop_forever()
