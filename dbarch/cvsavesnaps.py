# savesnaps.py
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

SNAPS_FOLDER = "snaps"
os.makedirs(SNAPS_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

MOVEMENT_THRESHOLD = 120        # px movement for significant change
MIN_TIME_GAP = 6                # seconds between snaps
last_snap_time = 0

MAX_HISTORY = 5
recent_snap_positions = []      # last 3–5 snapshots

# Buffer to stabilize movement detection
movement_buffer = {}            # label → count of consecutive large movements
BUFFER_REQUIRED = 3             # require 3 consecutive detections


def is_significant_change(history_tracks, current_tracks):
    global movement_buffer

    for snap in history_tracks:
        for label, (x, y) in current_tracks.items():

            # NEW object detected
            if label not in snap:
                print(f"🆕 New object appeared: {label}")
                return True

            px, py = snap[label]

            dx = abs(x - px)
            dy = abs(y - py)

            moved = dx > MOVEMENT_THRESHOLD or dy > MOVEMENT_THRESHOLD

            # Smooth movement detection
            if moved:
                movement_buffer[label] = movement_buffer.get(label, 0) + 1
            else:
                movement_buffer[label] = 0

            if movement_buffer[label] >= BUFFER_REQUIRED:
                print(f"🔄 {label} moved significantly.")
                movement_buffer[label] = 0
                return True

    return False


def should_save_snap(current_tracks):
    global last_snap_time, recent_snap_positions

    now = datetime.now().timestamp()

    if now - last_snap_time < MIN_TIME_GAP:
        return False

    if len(recent_snap_positions) == 0:
        print("📸 First snapshot saved.")
        return True

    if is_significant_change(recent_snap_positions, current_tracks):
        return True

    return False


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, conf=0.65, verbose=False)[0]

    current_tracks = {}  # label -> center position

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]

            x1, y1, x2, y2 = box.xyxy[0]
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            current_tracks[label] = (x_center, y_center)

    if should_save_snap(current_tracks):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(SNAPS_FOLDER, f"snap_{timestamp}.png")
        cv2.imwrite(filepath, frame)
        print(f"📸 Snap saved: {filepath}")

        # Store history
        recent_snap_positions.append(current_tracks.copy())
        if len(recent_snap_positions) > MAX_HISTORY:
            recent_snap_positions.pop(0)

        last_snap_time = datetime.now().timestamp()

    cv2.imshow("Dynamic Snap Capture", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
