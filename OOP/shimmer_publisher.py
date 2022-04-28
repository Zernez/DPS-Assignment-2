import time
import paho.mqtt.client as mqtt
from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType


def handler(pkt: DataPacket) -> None:
    cur_value = pkt[EChannelType.INTERNAL_ADC_13]
    client.publish("shimmer", cur_value)	
    print(f'Received new data point: {cur_value}')


if __name__ == '__main__':
    serial = Serial('/dev/rfcomm42', DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)

    global client

    client = mqtt.Client()
		# Client callback that is called when the client successfully connects to the broker.
    client.on_connect = on_connect
		# Client callback that is called when the client successfully publishes to the broker.
    client.on_publish = on_publish

		# Connect to the MQTT broker running in the localhost.
    client.connect("localhost", 1883, 60)

    shim_dev.initialize()

    dev_name = shim_dev.get_device_name()
    print(f'My name is: {dev_name}')

    shim_dev.add_stream_callback(handler)

    shim_dev.start_streaming()
    time.sleep(5.0)
    shim_dev.stop_streaming()

    shim_dev.shutdown()