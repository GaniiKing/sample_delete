import cv2
import numpy as np
import os
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

# ---------------------------------------------
# Load YOLO Model
# ---------------------------------------------
model = YOLO("yolov8n.pt")  # change to yolov8s/m/l for better accuracy

# ---------------------------------------------
# Background subtractor
# ---------------------------------------------
fgbg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=False)

OBJECT_AREA_THRESHOLD = 8000
SCENE_DIFF_THRESHOLD = 250000
SIMILARITY_THRESHOLD = 0.40  

SNAPS_FOLDER = "snaps"
os.makedirs(SNAPS_FOLDER, exist_ok=True)

# Track previously seen objects
previous_labels = set()

def is_unique_snap(new_img):
    """Check if new image is unique using SSIM similarity."""
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    for file in os.listdir(SNAPS_FOLDER):
        if not file.endswith(".png"):
            continue

        existing = cv2.imread(os.path.join(SNAPS_FOLDER, file))
        existing_gray = cv2.cvtColor(existing, cv2.COLOR_BGR2GRAY)

        # Resize for fair comparison
        existing_gray = cv2.resize(existing_gray, (new_gray.shape[1], new_gray.shape[0]))

        score, _ = ssim(new_gray, existing_gray, full=True)
        if score > SIMILARITY_THRESHOLD:
            return False  # Similar -> skip

    return True


cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # ---------------------------------------------------
    # 1️⃣ YOLO OBJECT DETECTION
    # ---------------------------------------------------
    results = model.predict(frame, conf=0.55, verbose=False)[0]

    current_labels = set()
    new_object_detected = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        current_labels.add(label)

        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Detect NEW object that was NOT there in previous frame
    newly_added = current_labels - previous_labels
    if len(newly_added) > 0:
        new_object_detected = True

    previous_labels = current_labels.copy()

    # ---------------------------------------------------
    # 2️⃣ Optional: Background Movement Detection
    # ---------------------------------------------------
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 7)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_motion_detected = any(cv2.contourArea(cnt) > OBJECT_AREA_THRESHOLD for cnt in contours)

    # ---------------------------------------------------
    # 3️⃣ Optional: SCENE CHANGE DETECTION
    # ---------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    diff_score = np.sum(diff)
    prev_gray = gray

    scene_changed = diff_score > SCENE_DIFF_THRESHOLD

    # ---------------------------------------------------
    # 4️⃣ SNAP LOGIC — SAVE ONLY WHEN NEW OBJECT APPEARS
    # ---------------------------------------------------
    if new_object_detected:  # <<<< ONLY here
        if is_unique_snap(frame):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SNAPS_FOLDER, f"snap_{timestamp}.png")
            cv2.imwrite(path, frame)
            cv2.putText(original, "📸 NEW OBJECT SNAP SAVED!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
        else:
            cv2.putText(original, "SKIPPED DUPLICATE", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Smart Snapshot System (YOLO Enhanced)", original)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
