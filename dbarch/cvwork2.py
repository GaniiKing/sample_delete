import cv2
import numpy as np
import os
import json
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import base64
import ollama

# ---------------------------------------------
# YOLO & folders
# ---------------------------------------------
model = YOLO("yolov8n.pt")
fgbg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=False)

OBJECT_AREA_THRESHOLD = 8000
SIMILARITY_THRESHOLD = 0.40  

SNAPS_FOLDER = "snaps"
os.makedirs(SNAPS_FOLDER, exist_ok=True)
METADATA_FILE = os.path.join(SNAPS_FOLDER, "metadata.json")
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "w") as f:
        json.dump([], f, indent=4)

previous_labels = set()

# ---------------------------------------------
# LLaMA Vision (standalone function)
# ---------------------------------------------
def describe_with_llama(image_path):
    """Run in parallel; generate description using LLaMA 3.2-Vision."""
    with open(image_path, "rb") as img:
        b64_image = base64.b64encode(img.read()).decode("utf-8")

    prompt = "Describe everything important in this image in 2–3 lines."

    try:
        response = ollama.chat(
            model="llava-llama3",
            messages=[{"role": "user", "content": prompt, "images": [b64_image]}]
        )
        description = response["message"]["content"]
    except Exception as e:
        description = f"Error generating description: {e}"

    # Append metadata
    with open(METADATA_FILE, "r+") as f:
        data = json.load(f)
        data.append({
            "image": image_path,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "description": description
        })
        f.seek(0)
        json.dump(data, f, indent=4)
    print(f"✅ Description saved for {image_path}")


# ---------------------------------------------
# Check uniqueness (SSIM)
# ---------------------------------------------
def is_unique_snap(new_img):
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    for file in os.listdir(SNAPS_FOLDER):
        if not file.endswith(".png"):
            continue
        existing = cv2.imread(os.path.join(SNAPS_FOLDER, file))
        existing_gray = cv2.cvtColor(existing, cv2.COLOR_BGR2GRAY)
        existing_gray = cv2.resize(existing_gray, (new_gray.shape[1], new_gray.shape[0]))
        score, _ = ssim(new_gray, existing_gray, full=True)
        if score > SIMILARITY_THRESHOLD:
            return False
    return True


# ---------------------------------------------
# Camera loop
# ---------------------------------------------
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Thread pool for parallel description
executor = ThreadPoolExecutor(max_workers=2)  # adjust depending on your GPU/CPU

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # YOLO detection
    results = model.predict(frame, conf=0.55, verbose=False)[0]
    current_labels = {results.names[int(box.cls[0])] for box in results.boxes}
    newly_added = current_labels - previous_labels
    new_object_detected = len(newly_added) > 0
    previous_labels = current_labels.copy()

    # SNAP logic
    if new_object_detected:
        if is_unique_snap(frame):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(SNAPS_FOLDER, f"snap_{timestamp}.png")
            cv2.imwrite(snap_path, frame)
            cv2.putText(original, "📸 SNAP SAVED!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
            # Send to LLaMA in parallel
            executor.submit(describe_with_llama, snap_path)
        else:
            cv2.putText(original, "SKIPPED DUPLICATE", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    # Show feed
    cv2.imshow("Smart Snapshot System + Parallel LLaMA", original)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
