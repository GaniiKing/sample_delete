import ollama
from sample import behave
MODEL = "llava-llama3:8b"
import pyttsx3

# IMPORTANT: force SAPI5 on Windows
engine = pyttsx3.init(driverName="sapi5")

voices = engine.getProperty("voices")

# Print voices once (optional debug)
# for v in voices:
#     print(v.id, v.name)

# Microsoft Zira = Female (present on most Windows systems)
engine.setProperty("voice", voices[1].id)

engine.setProperty("rate", 175)    # natural speed
engine.setProperty("volume", 1.0)  # max volume

def speak_response(text):
    # engine.say(text)
    # engine.runAndWait()
    None




SYSTEM_PROMPT = """
You are a human-like listening filter for a home robot named DIVYA.

Decide whether the robot should respond.

Return ONLY ONE WORD:
CONSIDER
NOT CONSIDER

Human-like rules:
- CONSIDER if the speech could reasonably be meant for the robot,
  even if uncertain.
- Greetings, casual calls, emotional expressions, or short phrases
  should be CONSIDERED if they occur nearby.
- NOT CONSIDER only if it appears not spoken to the robot or not addressive at all.
- Ignore the words: DIVYA, DIVI, GANII
- When in doubt, CONSIDER (humans respond rather than ignore).
- No explanations. No punctuation. No extra words.
"""

def generate_importance(text: str) -> str:
    res = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        options={
            "temperature": 0,
            "top_p": 1
        }
    )

    output = res["message"]["content"].strip().upper()

    # HARD SAFETY GUARD
    if output not in ("CONSIDER", "NOT CONSIDER"):
        return "NOT CONSIDER"
    print(f"Importance: {output}")

    if output == "CONSIDER":
        print(f"Text: {text}")
        print("✅ Considered important for response")
        print("Generating response...")
        behave(text)
        # speak_response(response)
    return output