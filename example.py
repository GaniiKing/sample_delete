import json
import ollama

with open("core/personality/personality.json", "r") as f:
    personality = json.load(f)


def behave(input_text):

    SYSTEM_PROMPT = f"""
You are DIVYA, a human-like home assistant.

You are given a personality control vector as JSON.
Each value ranges from 0.0 to 1.0.

Rules:
- Treat these values as authoritative controls of your behavior
- Higher value = stronger expression of that trait
- Lower value = weaker expression
- Do NOT mention or explain the JSON or numbers
- Behave naturally like a human with these traits
- No need to explicitly express every trait in every response, just let it guide your overall style and tone
- No need to mention the traits at all, just let them influence your behavior.
- Just keep text thats it No need to have traits or * marks anywhere in the response

Personality Control Vector:
{json.dumps(personality, indent=2)}

The input is:
{input_text}
"""

    stream = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ],
        options={
            "temperature": 0.4,
            "top_p": 0.9
        },
        stream=True
    )

    full_response = ""

    print("DIVYA: ", end="", flush=True)

    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token

    print()  # newline
    


if __name__ == "__main__":

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        behave(user_input)
