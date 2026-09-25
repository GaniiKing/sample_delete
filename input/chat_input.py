import ollama
import json

# 🔹 Temporary memory (chat history)
chat_history = []

def chat_with_ollama(user_input: str):
    global chat_history

    # 🔹 Build history string (last few messages only)
    history_text = ""
    for msg in chat_history[-6:]:   # limit to last 6 for efficiency
        history_text += f'{msg["role"].upper()}: {msg["content"]}\n'

    prompt = f"""
You are an Input Processing Module of an AI system. 
Your job is to: 
1. Analyze user input 
2. Decide if it should be stored as memory 
3. Classify the type of memory 
4. Generate a short response Think of processing the input as if you are a human deciding what to remember and how to respond in a conversation. 

Some of the characcteristics of your processing should be: 
"responsiveness": 0.1, 
"consistency": 0.0, 
"context_memory_weight": 0.9,
 ------------------------------------- 
 USER INPUT: "{user_input}"
   ------------------------------------- 
   DECISION RULES: 
   1. IMPORTANCE: - "CONSIDER" → if input contains personal info, preferences, facts, events, or useful conversational context - "NOT_CONSIDER" → if input is generic, repetitive, or not useful for future recall 
   2. MEMORY TYPE: - "TEMPORARY" → casual conversation, greetings, short-lived context - "SHORT_TERM" → useful for limited time (hours) - "LONG_TERM" → important personal facts, relationships, identity, preferences 
   3. EXPIRE TIME: - Only for SHORT_TERM - Give value in HOURS (e.g., 2, 6, 24) - Otherwise null 
   4. RESPONSE: - If CONSIDER → generate a natural short reply - If NOT_CONSIDER → return null 
   5. CLUSTER: - Assign a category if useful Examples: "Personal Info", "Emotion", "Task", "Preference", "General", "Greeting" - Else null
     ------------------------------------- 
     STRICT OUTPUT RULES: 
     - Return ONLY valid JSON 
     - NO explanation 
     - NO extra text 
     - Follow schema 
     EXACTLY 
     ------------------------------------- 
     OUTPUT FORMAT: {{ "importance": "CONSIDER" or "NOT_CONSIDER", "memory_type": "TEMPORARY" or "SHORT_TERM" or "LONG_TERM", "expire_time": number or null, "response": string or null, "cluster": string or null }}
"""

    try:
        stream = ollama.chat(
            model="llama3:8b",
            messages=[
                *chat_history,
                {"role": "user", "content": prompt}],
            options={
                "temperature": 0.1
            },
            stream=True
        )

        full_response = ""
        print("RESPONSE:", end=" ", flush=True)

        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token

        print()

        # 🔹 Parse JSON
        # 🔹 Parse JSON
        parsed = json.loads(full_response)

        # ✅ ALWAYS STORE (this is the fix)
        chat_history.append({"role": "user", "content": user_input})

        if parsed["response"]:
            chat_history.append({
                "role": "assistant",
                "content": parsed["response"]
            })

        # 🔹 Routing
        if parsed["memory_type"] != "TEMPORARY":
            print("DB REQUIRED")
        print("=" * 30)
        print(chat_history)

    except Exception as e:
        print("Error:", e)


# 🔹 LOOP
while True:
    usr = input("Enter chat input: ")
    chat_with_ollama(usr)