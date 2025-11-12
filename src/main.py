import cv2
import yaml
import RPi.GPIO as GPIO
import time
import numpy as np

# -----------------------------
# Load configuration
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SOLENOID_PIN = config["solenoid_pin"]
EMERGENCY_PIN = config["emergency_pin"]
LED_PIN = config["status_led_pin"]
HSV_LOWER = np.array(config["hsv_lower"])
HSV_UPPER = np.array(config["hsv_upper"])
PULSE_MS = config["pulse_ms"]
COOLDOWN_S = config["cooldown_s"]

# -----------------------------
# Setup GPIO
# -----------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(SOLENOID_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(EMERGENCY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# -----------------------------
# Setup USB webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = first USB webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Full HD width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Full HD height

last_shot_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera frame detected")
            time.sleep(0.5)
            continue

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for green color
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        green_pixels = cv2.countNonZero(mask)

        # Fire solenoid if green detected and emergency not pressed
        if green_pixels > 500 and GPIO.input(EMERGENCY_PIN):
