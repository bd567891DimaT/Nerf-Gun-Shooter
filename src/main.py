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
# Setup camera
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_shot_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera frame")
            time.sleep(0.5)
            continue

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for green color
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        green_pixels = cv2.countNonZero(mask)

        # If enough green is detected and emergency not pressed
        if green_pixels > 500 and GPIO.input(EMERGENCY_PIN):
            now = time.time()
            if now - last_shot_time >= COOLDOWN_S:
                print("Green detected! Firing solenoid.")
                GPIO.output(SOLENOID_PIN, GPIO.HIGH)
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(PULSE_MS / 1000)
                GPIO.output(SOLENOID_PIN, GPIO.LOW)
                GPIO.output(LED_PIN, GPIO.LOW)
                last_shot_time = now

        # Optional: show mask window
        cv2.imshow("Green Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
