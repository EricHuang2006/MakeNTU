import cv2
import socket
import struct
import numpy as np

# ==========================================
# Configuration
# ==========================================
# Replace with your i.MX 93's actual IP address
# You can find it by typing 'ifconfig' on the i.MX 93 terminal
# IMX93_IP = '192.168.0.73' 
IMX93_IP = '192.168.0.73' 
PORT = 9999

# ==========================================
# Network Initialization
# ==========================================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to i.MX 93 at {IMX93_IP}:{PORT}...")

try:
    client_socket.connect((IMX93_IP, PORT))
    print("Connected! Receiving video stream...")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

data = b""
# 'Q' specifies an 8-byte unsigned long long (must match the server)
payload_size = struct.calcsize("Q") 

# ==========================================
# Main Display Loop
# ==========================================
try:
    while True:
        # Retrieve the message size (first 8 bytes)
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data += packet

        if not data: break
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Retrieve the image data based on the extracted size
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Decode JPEG bytes back into a numpy array / OpenCV frame
        img_np = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("MakeNTU - i.MX 93 Face Tracking Live View", frame)

        # Press 'q' on the keyboard to exit the stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream closed by user.")
finally:
    client_socket.close()
    cv2.destroyAllWindows()
