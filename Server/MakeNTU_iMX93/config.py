import os

MODEL_PATH = 'yolov8n-pose_int8_vela.tflite'
UART_PORT = '/dev/ttyLP1'
BAUD_RATE = 115200
# Lowered threshold slightly to accommodate scaled confidence
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
IMG_SIZE = 320
DISPLAY_SIZE = 640
DISPLAY_SCALE = DISPLAY_SIZE / IMG_SIZE
C270_FOV = 60
HOST_IP = '0.0.0.0'
PORT = 9999
DISCORD_WEBHOOK_URL=os.getenv("DISCORD_WEBHOOK_URL")