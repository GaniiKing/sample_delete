"""
Unified entry: Telugu voice + optional camera (precep/intelligent_perception.py)
pipelined into selfcreate/trial.py (LLM + DB orchestration).
"""

from __future__ import annotations
import numpy
import os
import queue
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfcreate"))
sys.path.insert(0, str(ROOT / "precep"))

import cv2  # noqa: E402

import intelligent_perception as perception  # noqa: E402
import trial  # noqa: E402


def _packet_from_perception_item(item: dict) -> dict:
    text = (item.get("text") or "").strip()
    snap = item.get("snap")
    if snap and not os.path.isfile(snap):
        snap = None

    faces: list[dict] = []
    errors: list[str] = []
    if snap:
        frame = cv2.imread(snap)
        if frame is None:
            errors.append("snapshot_unreadable")
        else:
            try:
                detected = trial._identify_faces(frame)
                faces = [
                    {"face_id": f["face_id"], "name": f["name"], "confidence": f["confidence"]}
                    for f in detected
                ]
            except Exception as err:
                errors.append(f"face_detect: {err}")

    has_audio = bool(text)
    has_video = bool(snap)
    if has_audio and has_video:
        mode = "both"
    elif has_audio:
        mode = "audio"
    elif has_video:
        mode = "video"
    else:
        mode = "text"

    return {
        "mode": mode,
        "audio_text": text,
        "video_snapshot_path": snap,
        "video_change_score": 0.0,
        "video_significant_change": bool(snap),
        "faces": faces,
        "errors": errors,
        "final_user_text": text,
    }


def _process_queue_item(item: dict) -> None:
    if not (item.get("text") or "").strip():
        return
    packet = _packet_from_perception_item(item)
    trial.ask_llm(packet)


def _drain_speech_queue() -> None:
    while True:
        try:
            item = perception.speech_queue.get_nowait()
        except queue.Empty:
            break
        _process_queue_item(item)


def _run_voice_only() -> None:
    print("Voice-only mode (no camera). Speak in Telugu; Ctrl+C to exit.")
    try:
        while True:
            item = perception.speech_queue.get()
            _process_queue_item(item)
    except KeyboardInterrupt:
        print("\nSession closed.")


def main() -> None:
    base, _ = perception._ensure_output_dirs()
    print(f"Perception logs/snaps: {base}")
    trial.initialize_schema()

    threading.Thread(target=perception.speech_worker, daemon=True).start()

    cap = cv2.VideoCapture(perception.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera index {perception.CAMERA_INDEX}")
        _run_voice_only()
        return

    print("Camera running. Speak in Telugu; each valid utterance is sent to trial. Press ESC to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            with perception.frame_lock:
                perception.latest_frame = frame
            _drain_speech_queue()
            cv2.imshow("Voice + trial pipeline (ESC = quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
