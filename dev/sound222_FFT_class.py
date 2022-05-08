from Phidget22.PhidgetException import *
from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import traceback
import time
import paho.mqtt.client as mqtt
from time import sleep
from numpy.fft import fft
import numpy as np

#Declare any event handlers here. These will be called every time the associated event occurs.

class sounds:

	counter= 0
	sampler= []
	client = mqtt.Client()
	sample_rate= 1000
	sample_slice= sample_rate
	freq_band= 15
	freq_interested= int(sample_slice/freq_band)

	def onVoltageRatioChange(self, voltageRatio):
	
		if (self.counter< (self.sample_slice)):
			self.counter+= 1
			self.sampler.append(voltageRatio)
		else:
			row_ampl= [np.mean(np.abs(self.sampler), axis= 0)]
			temp_data_fft = np.abs(fft(self.sampler).real)
			data_fft= [temp_data_fft[0]]
               
			i= self.freq_interested
			j= 1
			slice_y= []
			start_index= 1
        
			while (i< len (temp_data_fft) and j <= self.freq_band):
				slice_y.append(i-1)
				i+= self.freq_interested
				j+= 1 

			for index in slice_y:
				temp_mean= temp_data_fft [start_index:index]
				data_fft.append(np.mean(temp_mean, axis= 0))
				start_index= index
            
			data_fft.append(row_ampl) 			

			self.client.publish("test", data_fft)
			self.sampler.clear()
			self.counter= 0
			
#		print("VoltageRatio: " + str(voltageRatio))

	def onAttach(self):
		print("Attach!")

	def onDetach(self):
		print("Detach!")

	def main(self):
		try:
			#Create your Phidget channels
			voltageRatioInput0 = self.VoltageRatioInput()

			#Set addressing parameters to specify which channel to open (if any)
			voltageRatioInput0.setDeviceSerialNumber(137422)

			#Assign any event handlers you need before calling open so that no events are missed.
			voltageRatioInput0.setOnVoltageRatioChangeHandler(self.onVoltageRatioChange)
			voltageRatioInput0.setOnAttachHandler(self.onAttach)
			voltageRatioInput0.setOnDetachHandler(self.onDetach)

			#Open your Phidgets and wait for attachment
			voltageRatioInput0.openWaitForAttachment(5000)
		
			dataInterval = voltageRatioInput0.getDataInterval()
			print("DataInterval: " + str(dataInterval))
		
			voltageRatioInput0.setDataInterval(1)
			dataInterval = voltageRatioInput0.getDataInterval()
			print("DataInterval: " + str(dataInterval))

			#Do stuff with your Phidgets here or in your event handlers.
		
			# The callback for when the client receives a CONNACK response from the server.
			def on_connect(client, userdata, flags, rc):
				print("Connected with result code "+str(rc))

			# The callback for when a PUBLISH message is received from the server.
			def on_publish(client, userdata, message_id):
				print(f"message with ID {message_id} published")

			# Client callback that is called when the client successfully connects to the broker.
			self.client.on_connect = on_connect
			# Client callback that is called when the client successfully publishes to the broker.
			self.client.on_publish = on_publish

			# Connect to the MQTT broker running in the localhost.
			self.client.connect("localhost", 1883, 60)		

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
