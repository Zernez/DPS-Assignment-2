from time import sleep
import paho.mqtt.client as mqtt

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

# The callback for when a PUBLISH message is received from the server.
def on_publish(client, userdata, message_id):
    print(f"message with ID {message_id} published")

client = mqtt.Client()
# Client callback that is called when the client successfully connects to the broker.
client.on_connect = on_connect
# Client callback that is called when the client successfully publishes to the broker.
client.on_publish = on_publish

# Connect to the MQTT broker running in the localhost.
client.connect("localhost", 1883, 60)
message_counter = 0

# The client will publish a message to the broker every 3 seconds.
while True:
    client.publish("test", f"test message {message_counter}")
    message_counter += 1
    sleep(3)