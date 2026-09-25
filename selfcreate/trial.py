import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from dbconnect import run_query

# Keep OpenCV backend logs quiet on systems without a camera.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import face_recognition  # type: ignore
except ImportError:
    face_recognition = None

try:
    import speech_recognition as sr  # type: ignore
except ImportError:
    sr = None

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "db_schema.json"
FACE_STORE_PATH = BASE_DIR / "face_memory.json"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
PERSONALITY_PATH = BASE_DIR.parent / "dividedArch" / "personality.txt"
API_URL = os.getenv("SELFCREATE_LLM_URL", "https://myai.ganiisunkara.workers.dev")
API_TOKEN = os.getenv("SELFCREATE_LLM_TOKEN", "Bearer 12345678")
MAX_HISTORY = int(os.getenv("SELFCREATE_MAX_HISTORY", "30"))
REQUEST_TIMEOUT = int(os.getenv("SELFCREATE_LLM_TIMEOUT", "30"))
VIDEO_CHANGE_THRESHOLD = float(os.getenv("SELFCREATE_VIDEO_CHANGE_THRESHOLD", "14.0"))
MAX_AUDIO_WINDOW_SECONDS = int(os.getenv("SELFCREATE_MAX_AUDIO_WINDOW_SECONDS", "35"))
ENABLE_DYNAMIC_VIDEO = os.getenv("SELFCREATE_ENABLE_DYNAMIC_VIDEO", "0").strip() == "1"
_camera_available_cache: bool | None = None

chat_history: deque[dict[str, str]] = deque(maxlen=MAX_HISTORY)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _load_schema() -> dict[str, Any]:
    return _load_json(SCHEMA_PATH, {})


def _normalize_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("response")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"response": payload}
    return {}


def _clean_response_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            maybe = json.loads(cleaned)
            if isinstance(maybe, dict) and isinstance(maybe.get("response"), str):
                return maybe["response"].strip()
        except json.JSONDecodeError:
            pass
    return cleaned


def _load_personality() -> str:
    if PERSONALITY_PATH.exists():
        return PERSONALITY_PATH.read_text(encoding="utf-8")
    return "direct, practical, concise"


def _llm_call(prompt: str, system_prompt: str, history: list[dict[str, str]]) -> dict[str, Any]:
    headers = {"Authorization": API_TOKEN, "Content-Type": "application/json"}
    request_body = {"prompt": prompt, "systemPrompt": system_prompt, "history": history}
    response = requests.post(API_URL, headers=headers, json=request_body, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_safe_read_query(query: str) -> bool:
    if not isinstance(query, str):
        return False
    q = query.strip()
    if not q:
        return False
    if ";" in q.rstrip(";"):
        return False
    first = q.split(None, 1)[0].lower()
    if first not in {"select", "with"}:
        return False
    forbidden = r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b"
    return re.search(forbidden, q, flags=re.IGNORECASE) is None


def _looks_like_information_query(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    if not lowered:
        return False
    if "?" in lowered:
        return True
    return lowered.startswith(("what", "who", "where", "when", "which", "how", "show", "list", "find", "tell me"))


def _clean_choice(value: Any, allowed: set[str], fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    candidate = value.strip().upper()
    return candidate if candidate in allowed else fallback


def _coerce_expire_time(value: Any) -> int | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _memory_heuristic(user_input: str, payload: dict[str, Any], media: dict[str, Any]) -> tuple[bool, str, float]:
    importance = _clean_choice(payload.get("importance"), {"CONSIDER", "NOT_CONSIDER"}, "NOT_CONSIDER")
    memory_type = _clean_choice(payload.get("memory_type"), {"TEMPORARY", "SHORT_TERM", "LONG_TERM"}, "TEMPORARY")
    personal = bool(
        re.search(
            r"\b(my name is|i am|i live in|my (email|phone|dob|birthday|goal|profession|hobby)|remember this|don.t forget)\b",
            user_input,
            flags=re.IGNORECASE,
        )
    )
    has_named_face = any(face.get("name") not in (None, "unknown") for face in media.get("faces", []))

    if importance == "CONSIDER" and memory_type in {"SHORT_TERM", "LONG_TERM"}:
        return True, "llm_recommended_storage", 0.92
    if personal:
        return True, "personal_fact_detected", 0.8
    if has_named_face:
        return True, "named_face_detected", 0.85
    if len(user_input.split()) <= 3 and not media.get("video_snapshot_path"):
        return False, "low_signal_short_input", 0.68
    return False, "fallback_not_considered", 0.62


def _face_id_from_encoding(encoding: list[float]) -> str:
    rounded = ",".join(str(round(v, 5)) for v in encoding[:32])
    return hashlib.sha256(rounded.encode("utf-8")).hexdigest()[:24]


def _load_face_store() -> dict[str, Any]:
    return _load_json(FACE_STORE_PATH, {"faces": []})


def _save_face_store(store: dict[str, Any]) -> None:
    _save_json(FACE_STORE_PATH, store)


def _capture_frame_from_webcam() -> tuple[Any, str | None]:
    global _camera_available_cache
    if cv2 is None:
        return None, "OpenCV not installed (`pip install opencv-python`)."
    if _camera_available_cache is False:
        return None, "Webcam unavailable."

    # Reduce noisy OpenCV logs when hardware is unavailable.
    try:
        cv2.setLogLevel(2)
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _camera_available_cache = False
        return None, "Unable to access webcam."
    _camera_available_cache = True
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, "Unable to read webcam frame."
    return frame, None


def _save_frame(frame: Any, suffix: str) -> str | None:
    if cv2 is None:
        return None
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{suffix}.jpg"
    path = SNAPSHOT_DIR / filename
    ok = cv2.imwrite(str(path), frame)
    return str(path) if ok else None


def _frame_change_score(frame_a: Any, frame_b: Any) -> float:
    if cv2 is None or frame_a is None or frame_b is None:
        return 0.0
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(diff.mean())


def _identify_faces(frame: Any) -> list[dict[str, Any]]:
    if face_recognition is None or frame is None:
        return []

    store = _load_face_store()
    known_encodings = []
    known_names = []
    for item in store.get("faces", []):
        encoding = item.get("encoding")
        name = item.get("name", "unknown")
        if isinstance(encoding, list) and encoding:
            known_encodings.append(encoding)
            known_names.append(name)

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected = []
    for encoding in face_encodings:
        encoded_list = [float(v) for v in encoding.tolist()]
        generated_face_id = _face_id_from_encoding(encoded_list)
        assigned_name = "unknown"
        confidence = 0.0
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
            distances = face_recognition.face_distance(known_encodings, encoding)
            if len(distances) > 0:
                best_idx = int(distances.argmin())
                confidence = float(max(0.0, 1.0 - distances[best_idx]))
                if matches[best_idx]:
                    assigned_name = known_names[best_idx]
        detected.append(
            {
                "face_id": generated_face_id,
                "name": assigned_name,
                "confidence": round(confidence, 4),
                "encoding": encoded_list,
            }
        )
    return detected


def _remember_face_name(face_id: str, name: str, encoding: list[float] | None = None) -> str | None:
    store = _load_face_store()
    for item in store.get("faces", []):
        if item.get("face_id") == face_id:
            item["name"] = name
            if encoding:
                item["encoding"] = encoding
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_face_store(store)
            return "updated"

    new_item = {
        "face_id": face_id,
        "name": name,
        "encoding": encoding or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    store.setdefault("faces", []).append(new_item)
    _save_face_store(store)
    return "inserted"


def _audio_to_text() -> tuple[str, str | None]:
    if sr is None:
        fallback = input("AUDIO_FALLBACK_TEXT: ").strip()
        return fallback, None

    recognizer = sr.Recognizer()
    try:
        transcripts: list[str] = []
        silence_streak = 0
        speech_detected = False
        start_time = time.time()

        with sr.Microphone() as source:
            print("Listening continuously... speak naturally.")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while (time.time() - start_time) < MAX_AUDIO_WINDOW_SECONDS:
                try:
                    audio_data = recognizer.listen(source, timeout=2, phrase_time_limit=8)
                except sr.WaitTimeoutError:
                    silence_streak += 1
                    if speech_detected and silence_streak >= 2:
                        break
                    if not speech_detected and silence_streak >= 4:
                        break
                    continue

                silence_streak = 0
                try:
                    chunk = recognizer.recognize_google(audio_data).strip()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as err:
                    return "", f"speech_service_error: {err}"

                if chunk:
                    transcripts.append(chunk)
                    speech_detected = True

        final_transcript = " ".join(transcripts).strip()
        return final_transcript, None
    except Exception as err:
        return "", f"audio_capture_failed: {err}"


def collect_dynamic_input(text_input: str = "") -> dict[str, Any]:
    media_info: dict[str, Any] = {
        "mode": "dynamic",
        "audio_text": "",
        "video_snapshot_path": None,
        "video_change_score": 0.0,
        "video_significant_change": False,
        "faces": [],
        "errors": [],
    }

    # 1) Always try continuous audio capture first.
    audio_text, audio_error = _audio_to_text()
    media_info["audio_text"] = audio_text
    if audio_error:
        media_info["errors"].append(audio_error)

    # 2) Dynamic video is optional; disabled by default to avoid webcam probe noise.
    #    Enable with SELFCREATE_ENABLE_DYNAMIC_VIDEO=1.
    baseline_frame = None
    if ENABLE_DYNAMIC_VIDEO and cv2 is not None:
        baseline_frame, baseline_error = _capture_frame_from_webcam()
        if baseline_error and baseline_error != "Webcam unavailable.":
            media_info["errors"].append(baseline_error)

        current_frame, frame_error = _capture_frame_from_webcam()
        if frame_error:
            if frame_error != "Webcam unavailable.":
                media_info["errors"].append(frame_error)
        else:
            if baseline_frame is not None:
                score = _frame_change_score(baseline_frame, current_frame)
                media_info["video_change_score"] = round(score, 4)
                media_info["video_significant_change"] = score >= VIDEO_CHANGE_THRESHOLD
                should_capture = media_info["video_significant_change"]
            else:
                should_capture = bool(audio_text.strip()) is False

            if should_capture:
                media_info["video_snapshot_path"] = _save_frame(current_frame, "webcam")
                faces = _identify_faces(current_frame)
                media_info["faces"] = [{"face_id": f["face_id"], "name": f["name"], "confidence": f["confidence"]} for f in faces]

                unnamed_faces = [f for f in faces if f["name"] == "unknown"]
                for face in unnamed_faces:
                    user_label = input(f"Detected unknown face {face['face_id']}. Enter name (or blank to skip): ").strip()
                    if user_label:
                        _remember_face_name(face["face_id"], user_label, encoding=face.get("encoding"))
                        face["name"] = user_label
                media_info["faces"] = [{"face_id": f["face_id"], "name": f["name"], "confidence": f["confidence"]} for f in faces]

    final_text = text_input.strip()
    if media_info["audio_text"]:
        final_text = media_info["audio_text"]
    if not final_text:
        final_text = input("No text detected. Enter message: ").strip()

    has_audio = bool(media_info["audio_text"].strip())
    has_video = bool(media_info["video_snapshot_path"])
    if has_audio and has_video:
        media_info["mode"] = "both"
    elif has_audio:
        media_info["mode"] = "audio"
    elif has_video:
        media_info["mode"] = "video"
    else:
        media_info["mode"] = "text"

    media_info["final_user_text"] = final_text
    return media_info


def _build_read_query_from_schema(user_input: str, schema: dict[str, Any], history: list[dict[str, str]], media: dict[str, Any]) -> str | None:
    prompt = f"""
You generate read-only PostgreSQL query from schema and multimodal context.
Return ONLY JSON: {{"DB_QUERY":"SELECT ..."}} or {{"DB_QUERY":null}}

SCHEMA: {json.dumps(schema, ensure_ascii=True)}
HISTORY: {json.dumps(history[-8:], ensure_ascii=True)}
MEDIA: {json.dumps(media, ensure_ascii=True)}
USER_INPUT: {user_input}

Rules:
- Only SELECT or WITH query.
- No write operations.
- No placeholders (? or :1).
"""
    try:
        raw = _llm_call(user_input, prompt, history[-8:])
        payload = _normalize_payload(raw)
        db_query = payload.get("DB_QUERY")
        if isinstance(db_query, str) and _is_safe_read_query(db_query):
            return db_query.strip()
    except Exception:
        return None
    return None


def _build_memory_insert_sql(user_input: str, response_text: str, memory_type: str, expire_time: int | None, confidence: float) -> str:
    expire_expr = "NULL"
    if memory_type == "SHORT_TERM" and expire_time is not None:
        expire_expr = f"(NOW() + INTERVAL '{expire_time} seconds')"
    sql = f"""
INSERT INTO personal_memory
(category, memory_key, memory_value, source_text, confidence, importance_score, memory_type, expires_at, tags, created_at, updated_at)
VALUES
('profile', 'user_statement', {_sql_literal(response_text)}, {_sql_literal(user_input)}, {round(confidence, 4)}, {round(confidence, 4)}, {_sql_literal(memory_type)}, {expire_expr}, '["selfcreate","multimodal"]'::jsonb, NOW(), NOW());
""".strip()
    return sql


def _build_face_memory_sql(faces: list[dict[str, Any]], snapshot_path: str | None) -> str | None:
    named_faces = [f for f in faces if f.get("name") and f.get("name") != "unknown"]
    if not named_faces:
        return None

    values = []
    for face in named_faces:
        face_id = _sql_literal(str(face["face_id"]))
        name = _sql_literal(str(face["name"]))
        confidence = round(float(face.get("confidence", 0.0)), 4)
        snap = "NULL" if not snapshot_path else _sql_literal(snapshot_path)
        values.append(f"({face_id}, {name}, {confidence}, {snap}, NOW(), NOW())")

    return (
        "INSERT INTO face_memory (face_id, person_name, confidence, last_snapshot_path, created_at, updated_at)\n"
        + "VALUES\n"
        + ",\n".join(values)
        + "\nON CONFLICT (face_id) DO UPDATE SET person_name = EXCLUDED.person_name, confidence = EXCLUDED.confidence, last_snapshot_path = EXCLUDED.last_snapshot_path, updated_at = NOW();"
    )


def generate_context_output(history: list[dict[str, str]], db_query: str, db_result: Any, user_input: str, media: dict[str, Any]) -> str:
    context_prompt = f"""
Answer user using DB_RESULT, USER_INPUT and multimodal context.
Return only JSON: {{"response":"string"}}

USER_INPUT: {user_input}
DB_QUERY: {db_query}
DB_RESULT: {json.dumps(db_result, ensure_ascii=True)}
MEDIA: {json.dumps(media, ensure_ascii=True)}
HISTORY: {json.dumps(history[-12:], ensure_ascii=True)}
"""
    try:
        raw = _llm_call(user_input, context_prompt, history[-12:])
        payload = _normalize_payload(raw)
        text = _clean_response_text(payload.get("response"))
        if text:
            return text
    except Exception:
        pass
    return "I checked the available database context but could not produce a reliable answer right now."


def initialize_schema() -> dict[str, Any]:
    schema = _load_schema()
    executed = []
    for sql in schema.get("bootstrap_sql", []):
        run_query(sql, readonly_only=False)
        executed.append(sql[:80] + ("..." if len(sql) > 80 else ""))
    return {"initialized": True, "executed_count": len(executed)}


def ask_llm(multimodal_input: dict[str, Any]) -> dict[str, Any]:
    schema = _load_schema()
    personality = _load_personality()
    user_input = multimodal_input.get("final_user_text", "").strip()
    media = multimodal_input

    chat_history.append({"role": "user", "content": user_input})
    history_snapshot = list(chat_history)[-20:]

    planner_prompt = f"""
You are Divya multimodal memory orchestrator.
Decide if this input should be stored and optionally generate read query.
Use schema and multimodal context.
Never generate write query in DB_QUERY.

PERSONALITY: {personality}
SCHEMA: {json.dumps(schema, ensure_ascii=True)}
MEDIA: {json.dumps(media, ensure_ascii=True)}

Return ONLY JSON:
{{
  "importance": "CONSIDER|NOT_CONSIDER",
  "memory_type": "TEMPORARY|SHORT_TERM|LONG_TERM",
  "expire_time": number|null,
  "response": "string",
  "DB_QUERY": "SELECT ...|WITH ...|null"
}}
"""

    try:
        raw = _llm_call(user_input, planner_prompt, history_snapshot)
        payload = _normalize_payload(raw)
    except Exception as err:
        payload = {"response": f"LLM error: {err}"}

    response_text = _clean_response_text(payload.get("response"))
    if not response_text:
        response_text = "Input received. I will keep improving with each interaction."

    should_store, reason, confidence = _memory_heuristic(user_input, payload, media)
    memory_type = _clean_choice(payload.get("memory_type"), {"TEMPORARY", "SHORT_TERM", "LONG_TERM"}, "TEMPORARY")
    expire_time = _coerce_expire_time(payload.get("expire_time"))
    if not should_store:
        memory_type = "TEMPORARY"
        expire_time = 0

    db_query = payload.get("DB_QUERY")
    if (not isinstance(db_query, str) or not _is_safe_read_query(db_query)) and _looks_like_information_query(user_input):
        db_query = _build_read_query_from_schema(user_input, schema, list(chat_history), media)

    db_result = None
    if isinstance(db_query, str) and _is_safe_read_query(db_query):
        db_result = run_query(db_query, readonly_only=True)
        response_text = generate_context_output(list(chat_history), db_query, db_result, user_input, media)

    memory_write_query = None
    if should_store:
        memory_write_query = _build_memory_insert_sql(user_input, response_text, memory_type, expire_time, confidence)

    face_memory_write_query = _build_face_memory_sql(media.get("faces", []), media.get("video_snapshot_path"))

    output = {
        "meta": {
            "engine": "selfcreate-multimodal-v4",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "latency_ms": int((time.perf_counter() % 1) * 1000),
        },
        "input_layer": {
            "mode": media.get("mode"),
            "audio_text": media.get("audio_text"),
            "video_snapshot_path": media.get("video_snapshot_path"),
            "video_change_score": media.get("video_change_score"),
            "video_significant_change": media.get("video_significant_change"),
            "faces": media.get("faces", []),
            "errors": media.get("errors", []),
        },
        "decision": {
            "should_store": should_store,
            "reason": reason,
            "confidence": confidence,
            "memory_type": memory_type,
            "expire_time_seconds": expire_time,
        },
        "db": {
            "read_query": db_query if isinstance(db_query, str) else None,
            "read_result": db_result,
            "memory_write_query": memory_write_query,
            "face_memory_write_query": face_memory_write_query,
        },
        "assistant": {"response": response_text},
    }

    chat_history.append({"role": "assistant", "content": response_text})
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return output


def main() -> None:
    initialize_schema()
    print("Selfcreate multimodal engine ready. Dynamic capture enabled.")
    while True:
        user_trigger = input("INPUT (type message or press Enter for dynamic audio/video; quit to exit): ").strip()
        if user_trigger.lower() in {"exit", "quit"}:
            print("Session closed.")
            break
        input_packet = collect_dynamic_input(user_trigger)
        ask_llm(input_packet)


if __name__ == "__main__":
    main()