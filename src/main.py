#!/usr/bin/env python3
"""
src/main.py
Basic Nerf shooter for Raspberry Pi (Python 3)
- Uses OpenCV to detect a colored marker on a spinning target.
- Estimates angular velocity, predicts when to fire.
- Fires solenoid by toggling a GPIO that drives a MOSFET gate.
- Enforces safety: max_pulse_ms (<=500) and min_cooldown_s (>=0.5).
- Handles camera disconnects and emergency stop button.
- Logs shot events to a log file for judges.
"""

import time
import math
import collections
import logging
import os
import sys
import yaml
import cv2

# Prefer RPi.GPIO on Raspberry Pi
try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None

# ---------- Utility: load config ----------
ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
if not os.path.exists(CONFIG_PATH):
    print("Missing config.yaml in project root. Exiting.")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ---------- Config shortcuts ----------
WIDTH = int(cfg.get("width", 640))
HEIGHT = int(cfg.get("height", 480))
CAM_IDX = int(cfg.get("camera_index", 0))
HSV_LOWER = tuple(cfg.get("hsv_lower", [5,150,150]))
HSV_UPPER = tuple(cfg.get("hsv_upper", [15,255,255]))
MIN_CONTOUR_AREA = int(cfg.get("min_contour_area", 120))
DESIRED_ANGLE_DEG = float(cfg.get("desired_angle_deg", 0.0))
PROCESS_LATENCY = float(cfg.get("processing_latency_s", 0.05))
FIRE_WINDOW = float(cfg.get("fire_window_s", 0.06))

SOL_PIN = int(cfg.get("solenoid_gpio_bcm", 17))
EMG_PIN = int(cfg.get("emergency_gpio_bcm", 27))
LED_PIN = int(cfg.get("status_led_gpio_bcm", 22))

MAX_PULSE_MS = int(cfg.get("max_pulse_ms", 500))
DEFAULT_PULSE_MS = int(cfg.get("default_pulse_ms", 150))
MIN_COOLDOWN = float(cfg.get("min_cooldown_s", 0.5))

RECONNECT_INTERVAL = float(cfg.get("reconnect_interval_s", 1.0))
HISTORY_LEN = int(cfg.get("history_len", 8))

LOG_FILE = cfg.get("log_file", "shots.log")

# ---------- Logging ----------
logging.basicConfig(filename=os.path.join(ROOT, LOG_FILE),
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("nerf_shooter")
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

# ---------- GPIO Setup ----------
if GPIO is None:
    logger.error("RPi.GPIO not available. Install RPi.GPIO on the Pi before running.")
    sys.exit(1)

GPIO.setmode(GPIO.BCM)
GPIO.setup(SOL_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(EMG_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# ---------- Simple angular estimator ----------
class AngularEstimator:
    def __init__(self, history_len=8, center=(WIDTH/2.0, HEIGHT/2.0)):
        self.history = collections.deque(maxlen=history_len)  # (t, angle)
        self.center = center

    @staticmethod
    def _unwrap(angles):
        unwrapped = [angles[0]]
        for a in angles[1:]:
            prev = unwrapped[-1]
            while a - prev > math.pi:
                a -= 2*math.pi
            while a - prev < -math.pi:
                a += 2*math.pi
            unwrapped.append(a)
        return unwrapped

    def add_sample(self, t, cx, cy):
        angle = math.atan2(cy - self.center[1], cx - self.center[0])
        self.history.append((t, angle))

    def estimate_omega(self):
        if len(self.history) < 3:
            return None
        times = [t for t,a in self.history]
        angles = [a for t,a in self.history]
        unwrapped = self._unwrap(angles)
        n = len(unwrapped)
        t0 = times[0]
        xs = [times[i] - t0 for i in range(n)]
        ys = [unwrapped[i] for i in range(n)]
        mean_x = sum(xs)/n
        mean_y = sum(ys)/n
        num = sum((xs[i]-mean_x)*(ys[i]-mean_y) for i in range(n))
        den = sum((xs[i]-mean_x)**2 for i in range(n))
        if den == 0:
            return None
        slope = num/den
        return slope  # rad/s

# ---------- Detection ----------
def detect_color_marker(frame):
    """Return (cx, cy) or None and mask for debugging."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = HSV_LOWER
    upper = HSV_UPPER
    mask = cv2.inRange(hsv, lower, upper)
    # clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None, mask
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return (cx, cy), mask

# ---------- Shooter class ----------
class Shooter:
    def __init__(self):
        self.last_shot = 0.0

    def emergency_pressed(self):
        return GPIO.input(EMG_PIN) == GPIO.HIGH

    def can_fire(self):
        return (time.time() - self.last_shot) >= MIN_COOLDOWN

    def set_status_led(self, on):
        GPIO.output(LED_PIN, GPIO.HIGH if on else GPIO.LOW)

    def fire(self, pulse_ms=None):
        if pulse_ms is None:
            pulse_ms = DEFAULT_PULSE_MS
        pulse_ms = min(int(pulse_ms), MAX_PULSE_MS)
        if self.emergency_pressed():
            logger.warning("Emergency pressed - shot blocked")
            return False
        if not self.can_fire():
            logger.debug("Cooldown active - shot blocked")
            return False
        logger.info("Firing solenoid for %d ms", pulse_ms)
        GPIO.output(SOL_PIN, GPIO.HIGH)
        time.sleep(pulse_ms / 1000.0)
        GPIO.output(SOL_PIN, GPIO.LOW)
        self.last_shot = time.time()
        logger.info("Shot recorded")
        return True

    def cleanup(self):
        GPIO.output(SOL_PIN, GPIO.LOW)
        GPIO.output(LED_PIN, GPIO.LOW)

# ---------- Main loop ----------
def open_camera():
    cap = cv2.VideoCapture(CAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, cfg.get("fps",30))
    return cap if cap.isOpened() else None

def main():
    cap = open_camera()
    last_reconnect = time.time()
    shooter = Shooter()
    shooter.set_status_led(True)
    estimator = AngularEstimator(history_len=HISTORY_LEN, center=(WIDTH/2.0, HEIGHT/2.0))
    desired_angle = math.radians(DESIRED_ANGLE_DEG)

    try:
        while True:
            # emergency handling
            if shooter.emergency_pressed():
                shooter.set_status_led(False)
                logger.warning("Emergency pressed - pausing firing until released")
                while shooter.emergency_pressed():
                    # try to reconnect camera while waiting
                    if cap is None or not cap.isOpened():
                        now = time.time()
                        if now - last_reconnect >= RECONNECT_INTERVAL:
                            logger.info("Trying camera reconnect while emergency held")
                            cap = open_camera()
                            last_reconnect = now
                    time.sleep(0.1)
                shooter.set_status_led(True)
                logger.info("Emergency released - resuming")

            # camera reconnect logic
            if cap is None or not cap.isOpened():
                now = time.time()
                if now - last_reconnect >= RECONNECT_INTERVAL:
                    logger.info("Attempting camera reconnect")
                    cap = open_camera()
                    last_reconnect = now
                shooter.set_status_led(False)
                time.sleep(0.05)
                continue
            else:
                shooter.set_status_led(True)

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Camera read failed, closing and will reconnect")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                shooter.set_status_led(False)
                time.sleep(0.05)
                continue

            pos, mask = detect_color_marker(frame)
            tnow = time.time()
            if pos is None:
                # no detection this frame
                time.sleep(0.001)
                continue

            cx, cy = pos
            estimator.add_sample(tnow, cx, cy)
            omega = estimator.estimate_omega()
            if omega is None:
                continue

            # current angle and diff to desired
            current_angle = math.atan2(cy - HEIGHT/2.0, cx - WIDTH/2.0)
            angle_diff = desired_angle - current_angle
            # wrap to -pi..pi
            while angle_diff > math.pi:
                angle_diff -= 2*math.pi
            while angle_diff < -math.pi:
                angle_diff += 2*math.pi

            if abs(omega) < 1e-4:
                continue
            time_to_reach = angle_diff / omega
            period = 2*math.pi / abs(omega)
            while time_to_reach < 0:
                time_to_reach += period

            planned_fire_in = time_to_reach - PROCESS_LATENCY

            if 0 <= planned_fire_in <= FIRE_WINDOW:
                fired = shooter.fire(DEFAULT_PULSE_MS)
                if fired:
                    logger.info("Shot fired at t=%.3f (omega=%.3f rad/s, time_to_reach=%.3f)",
                                time.time(), omega, time_to_reach)
                else:
                    logger.info("Shot attempt blocked")

            # small sleep to limit CPU usage
            time.sleep(max(0.001, 1.0 / max(30, cfg.get("fps",30))))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt - exiting")
    finally:
        logger.info("Cleaning up")
        try:
            shooter.cleanup()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        GPIO.cleanup()

if __name__ == "__main__":
    main()
