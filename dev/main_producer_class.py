import traceback
import time
import paho.mqtt.client as mqtt
from time import sleep
from numpy.fft import fft
from sound222_FFT_class import sounds

input= sounds()

input.main()