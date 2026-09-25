import ollama 
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

modell = "llama3:8b"
ENABLE_TTS = os.getenv("TEMPORARYUSE_ENABLE_TTS", "1").strip() == "1"
ENABLE_SIGNAL_SERVER = os.getenv("TEMPORARYUSE_ENABLE_SIGNAL_SERVER", "1").strip() == "1"

personality = None
tts_queue = queue.Queue()

_signal_lock = threading.Lock()
_signal_state = {
    "seq": 0,
    "path": None,
    "updated_at": None,
}


def _publish_audio_signal(audio_path: Path) -> None:
    """Update state served by Flask GET /signal after a new MP3 is written."""
    resolved = str(audio_path.resolve())
    with _signal_lock:
        _signal_state["seq"] = _signal_state["seq"] + 1
        _signal_state["path"] = resolved
        _signal_state["updated_at"] = time.time()
        seq = _signal_state["seq"]
        updated_at = _signal_state["updated_at"]

    webhook = os.getenv("TEMPORARYUSE_SIGNAL_POST_URL", "").strip()
    if webhook:
        try:
            import requests

            requests.post(
                webhook,
                json={"path": resolved, "seq": seq, "updated_at": updated_at},
                timeout=5,
            )
        except Exception as exc:
            print("Signal webhook POST failed:", exc)


def _create_signal_app():
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    cors_origin = os.getenv("TEMPORARYUSE_SIGNAL_CORS", "*").strip() or "*"

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = cors_origin
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/signal", methods=["GET", "OPTIONS"])
    def signal():
        if request.method == "OPTIONS":
            return ("", 204)
        with _signal_lock:
            payload = {
                "ok": True,
                "audio_ready": _signal_state["path"] is not None,
                "path": _signal_state["path"],
            }
        return jsonify(payload)

    return app


def _run_signal_server():
    host = os.getenv("TEMPORARYUSE_SIGNAL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("TEMPORARYUSE_SIGNAL_PORT", "5055"))
    app = _create_signal_app()
    print(f"Signal server: http://{host}:{port}/signal")
    app.run(host=host, port=port, threaded=True, use_reloader=False)
def clean_history(history):
    cleaned = []
    for msg in history:
        content = msg["content"]
        
        # If content is dict → extract string
        if isinstance(content, dict):
            content = content.get("response", str(content))
        
        cleaned.append({
            "role": msg["role"],
            "content": str(content)
        })
    return cleaned
with open("temporaryuse/personality.txt", "r") as f:
    personality = f.read()
print("Personality:", personality)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "precep"))
import intelligent_perception as perception


def _tts_worker():
    """Write queued text to MP3 via Edge TTS (`speak.py`); frontend can play the file."""
    from speak import get_audio_output_path, save_tts_mp3_sync

    out_path = get_audio_output_path()
    while True:
        text = tts_queue.get()
        if text is None:
            return
        try:
            print("Synthesizing TTS to:", out_path)
            save_tts_mp3_sync(text, out_path)
            print("Saved audio:", out_path)
            _publish_audio_signal(out_path)
        except Exception as exc:
            print("TTS error:", exc)


def speak_text(text):
    if not ENABLE_TTS:
        return
    clean = (text or "").strip()
    if not clean:
        return
    tts_queue.put(clean)

def generate_response(user_input, chat_history, scene_context=None):
    scene_context = scene_context or {"people": [], "objects": []}
    people = scene_context.get("people", [])
    objects = scene_context.get("objects", [])
    context_text = (
        f"\nDETECTED_PEOPLE: {json.dumps(people, ensure_ascii=False)}"
        f"\nDETECTED_OBJECTS: {json.dumps(objects, ensure_ascii=False)}"
    )
    enriched_input = f"{user_input}{context_text}"

    chat_history.append({"role": "user", "content": enriched_input})

    SYSTEM_PROMPT = f"""
    You are Divya so respond as Divya.Your name is DIVYA
    You are given with chat history , user_input Your task is to generate a context-aware response based on the chat history, user input.
    The response text should be generated based on the below personalities:(each ranging from 0-1)
    {personality}
    Have your response language as telugu only.
    Output format is strictly in JSON and should follow the below format and rules strictly:
    {{
        "response": string
    }}
    Note : No outer json should be there in the output.
    Note : No extra text should be there in the output.
    Note : No explanation should be there in the output.
    Note : No markdown should be there in the output.
    Note : No backticks should be there in the output.
    """
    import requests

    url = "https://myai.ganiisunkara.workers.dev"

    headers = {
        "Authorization": "Bearer 12345678",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": enriched_input,
        "systemPrompt": SYSTEM_PROMPT,
        "history": clean_history(chat_history[:-1])  # Exclude the current user input from history
    }

    response = requests.post(url, headers=headers, json=data)
    res = json.loads(response.text)
    # print("LLM RAW RESPONSE:", res["DB_QUERY"])
    print("LLM RAW RESPONSE:", res)
    outer = res.get("response",None)
    if outer:
        print("Reponse from LLM:", outer)
        speak_text(outer)
    chat_history.append({"role": "assistant", "content": outer})


def _drain_speech_queue(chat_history):
    while True:
        try:
            item = perception.speech_queue.get_nowait()
        except queue.Empty:
            break

        text = (item.get("text") or "").strip()
        if not text:
            continue
        scene = {
            "people": item.get("people", []),
            "objects": item.get("objects", []),
            "snap": item.get("snap"),
        }
        print(f"Voice input (te-IN): {text}")
        if scene["people"] or scene["objects"]:
            print("Scene:", json.dumps(scene, ensure_ascii=False))
        generate_response(text, chat_history, scene_context=scene)

if __name__ == "__main__":
    chat_history = []
    if ENABLE_SIGNAL_SERVER:
        threading.Thread(target=_run_signal_server, daemon=True).start()
    if ENABLE_TTS:
        threading.Thread(target=_tts_worker, daemon=True).start()
    threading.Thread(target=perception.speech_worker, daemon=True).start()

    print("Voice-only mode enabled. Camera is disabled until requested. Ctrl+C to exit.")
    try:
        while True:
            _drain_speech_queue(chat_history)
    except KeyboardInterrupt:
        print("\nSession closed.")