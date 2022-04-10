from Phidget22.PhidgetException import *
from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import traceback
import time
import paho.mqtt.client as mqtt
from time import sleep

#Declare any event handlers here. These will be called every time the associated event occurs.

def onVoltageRatioChange(self, voltageRatio):
	client.publish("test", f"test message {voltageRatio}")	
	print("VoltageRatio: " + str(voltageRatio))

def onAttach(self):
	print("Attach!")

def onDetach(self):
	print("Detach!")

def main():
	try:
		#Create your Phidget channels
		voltageRatioInput0 = VoltageRatioInput()

		#Set addressing parameters to specify which channel to open (if any)
		voltageRatioInput0.setDeviceSerialNumber(137422)

		#Assign any event handlers you need before calling open so that no events are missed.
		voltageRatioInput0.setOnVoltageRatioChangeHandler(onVoltageRatioChange)
		voltageRatioInput0.setOnAttachHandler(onAttach)
		voltageRatioInput0.setOnDetachHandler(onDetach)

		#Open your Phidgets and wait for attachment
		voltageRatioInput0.openWaitForAttachment(5000)
		
		dataInterval = voltageRatioInput0.getDataInterval()
		print("DataInterval: " + str(dataInterval))
		
		voltageRatioInput0.setDataInterval(10)
		dataInterval = voltageRatioInput0.getDataInterval()
		print("DataInterval: " + str(dataInterval))

		#Do stuff with your Phidgets here or in your event handlers.
		
		# The callback for when the client receives a CONNACK response from the server.
		def on_connect(client, userdata, flags, rc):
			print("Connected with result code "+str(rc))

		# The callback for when a PUBLISH message is received from the server.
		def on_publish(client, userdata, message_id):
			print(f"message with ID {message_id} published")
		
		global client 
		client = mqtt.Client()
		# Client callback that is called when the client successfully connects to the broker.
		client.on_connect = on_connect
		# Client callback that is called when the client successfully publishes to the broker.
		client.on_publish = on_publish

		# Connect to the MQTT broker running in the localhost.
		client.connect("localhost", 1883, 60)		
		
#		message_counter = 0

		# The client will publish a message to the broker every 3 seconds.
#		while True:
#			client.publish("test", f"test message {message_counter}")
#			message_counter += 1
#			sleep(1)

		try:			
			input("Press Enter to Stop\n")
		except (Exception, KeyboardInterrupt):
			pass

		#Close your Phidgets once the program is done.
		voltageRatioInput0.close()

	except PhidgetException as ex:
		#We will catch Phidget Exceptions here, and print the error informaiton.
		traceback.print_exc()
		print("")
		print("PhidgetException " + str(ex.code) + " (" + ex.description + "): " + ex.details)


main()
