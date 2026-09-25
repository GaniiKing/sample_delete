# intelligent_perception.py
# Telugu-focused voice capture + efficient transcription; snapshot only after valid speech.

import os
import queue
import threading
import time
from datetime import datetime

import cv2
import speech_recognition as sr
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# =========================================================
# CONFIG
# =========================================================

CAMERA_INDEX = 0
# Google Speech API: Telugu (India). Works well for spoken Telugu when online.
SPEECH_LANGUAGE = "te-IN"

OUTPUT_DIR = "perception_out"
SNAP_SUBDIR = "snaps"
TRANSCRIPT_LOG = "telugu_transcripts.txt"

# Listen: wait for speech, cap phrase length to keep latency reasonable.
LISTEN_TIMEOUT_S = 30
PHRASE_TIME_LIMIT_S = 20

# Optional: ignore very short noise bursts (seconds of audio data).
MIN_AUDIO_DURATION_S = 0.35
YOLO_MODEL = "yolov8n.pt"
OBJECT_CONFIDENCE = 0.45


# =========================================================
# SHARED STATE (main thread + speech thread)
# =========================================================

frame_lock = threading.Lock()
latest_frame = None  # last BGR frame from camera; updated only in main loop

speech_queue = queue.Queue()
_yolo_model = None
_hog_people_detector = None


def _ensure_output_dirs():
    base = os.path.join(os.path.dirname(__file__) or ".", OUTPUT_DIR)
    snaps = os.path.join(base, SNAP_SUBDIR)
    os.makedirs(snaps, exist_ok=True)
    return base, snaps


def _audio_duration_s(audio: sr.AudioData) -> float:
    if audio.sample_rate <= 0 or audio.sample_width <= 0:
        return 0.0
    num_samples = len(audio.frame_data) // audio.sample_width
    return num_samples / float(audio.sample_rate)


def _get_yolo_model():
    global _yolo_model
    if YOLO is None:
        return None
    if _yolo_model is None:
        try:
            _yolo_model = YOLO(YOLO_MODEL)
        except Exception:
            _yolo_model = None
    return _yolo_model


def _get_hog_people_detector():
    global _hog_people_detector
    if _hog_people_detector is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _hog_people_detector = hog
    return _hog_people_detector


def detect_people_and_objects(frame):
    if frame is None:
        return [], []

    objects = []
    people = []

    model = _get_yolo_model()
    if model is not None:
        try:
            result = model(frame, conf=OBJECT_CONFIDENCE, verbose=False)[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = str(result.names[cls])
                    objects.append(label)
                    if label == "person":
                        people.append("person")
        except Exception:
            pass

    # Fallback people detector if YOLO unavailable or didn't detect people.
    if not people:
        try:
            hog = _get_hog_people_detector()
            boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for _ in boxes:
                people.append("person")
            if boxes is not None and len(boxes) > 0:
                objects.extend(["person"] * len(boxes))
        except Exception:
            pass

    return people, objects


def speech_worker():
    recognizer = sr.Recognizer()
    # Slightly less aggressive energy threshold after ambient calibration.
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)

    print("Speech worker ready (Telugu / te-IN). Speak after calibration.")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT_S,
                    phrase_time_limit=PHRASE_TIME_LIMIT_S,
                )
        except sr.WaitTimeoutError:
            # No speech — no snapshot, no enqueue.
            continue
        except Exception as e:
            print("Mic/listen error:", e)
            time.sleep(0.5)
            continue

        if _audio_duration_s(audio) < MIN_AUDIO_DURATION_S:
            continue

        try:
            text = recognizer.recognize_google(audio, language=SPEECH_LANGUAGE)
        except sr.UnknownValueError:
            print("(Could not understand audio — no snapshot)")
            continue
        except sr.RequestError as e:
            print("Recognition service error:", e)
            continue
        except Exception as e:
            print("Transcription error:", e)
            continue

        text = (text or "").strip()
        if not text:
            continue

        # Valid transcription only from here — safe to tie a snapshot to this utterance.
        with frame_lock:
            frame_copy = None if latest_frame is None else latest_frame.copy()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, snaps_dir = _ensure_output_dirs()

        snap_path = None
        if frame_copy is not None:
            snap_path = os.path.join(snaps_dir, f"snap_{stamp}.jpg")
            cv2.imwrite(snap_path, frame_copy)

        line = f"{stamp}\t{text}"
        if snap_path:
            line += f"\t{snap_path}"
        line += "\n"

        log_path = os.path.join(base, TRANSCRIPT_LOG)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

        people, objects = detect_people_and_objects(frame_copy)
        speech_queue.put(
            {
                "text": text,
                "snap": snap_path,
                "stamp": stamp,
                "people": people,
                "objects": objects,
            }
        )
        print(f"Transcript ({SPEECH_LANGUAGE}): {text}")
        if snap_path:
            print(f"Snapshot: {snap_path}")


def main():
    global latest_frame

    base, _ = _ensure_output_dirs()
    print(f"Output folder: {base}")

    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera index {CAMERA_INDEX}")
        return

    print("Camera running. Snapshots are saved only after a successful Telugu transcription.")
    print("Press ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with frame_lock:
                latest_frame = frame

            cv2.imshow("Telugu voice + snapshot (valid speech only)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
