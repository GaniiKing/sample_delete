import os
import json
import queue
import zipfile
import requests
import sounddevice as sd
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from input.consider_logic import generate_importance

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_ZIP = MODEL_NAME + ".zip"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
SAMPLE_RATE = 16000

audio_queue = queue.Queue()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*"
}

def download_model():
    if os.path.isdir(MODEL_NAME):
        print("✅ English model already exists")
        return

    print("⬇️ Downloading English model...")

    r = requests.get(
        MODEL_URL,
        headers=HEADERS,
        stream=True,
        allow_redirects=True,
        timeout=60
    )

    if r.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {r.status_code}")

    total = int(r.headers.get("content-length", 0))

    with open(MODEL_ZIP, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc="Downloading"
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    # 🔒 ZIP VALIDATION
    if not zipfile.is_zipfile(MODEL_ZIP):
        os.remove(MODEL_ZIP)
        raise RuntimeError(
            "Downloaded file is not a ZIP. "
            "Check firewall / antivirus / proxy."
        )

    print("📦 Extracting English model...")
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")

    os.remove(MODEL_ZIP)
    print("✅ English model ready")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))

def detect_importance(text):
    generate_importance(text)
    


def get_input_voice():
    download_model()

    model = Model(MODEL_NAME)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    print("🎤 Speak English (CTRL+C to stop)")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print("LLM is thinking...")
                    detect_importance(text)
                    print("🗣️", text)



