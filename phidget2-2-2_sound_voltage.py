from Phidget22.PhidgetException import *
from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import traceback
import time

#Declare any event handlers here. These will be called every time the associated event occurs.

def onVoltageRatioChange(self, voltageRatio):
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

		#Do stuff with your Phidgets here or in your event handlers.

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
