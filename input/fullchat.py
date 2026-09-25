import ollama
import json
import time
from dbconnect import run_query

# 🔹 Temporary memory (chat history)
chat_history = []

# 🔹 Config
MAX_HISTORY = 200
TIMEOUT = 300  # 5 minutes

# 🔹 Track last interaction
last_interaction_time = time.time()


def reset_if_needed():
    global chat_history, last_interaction_time

    current_time = time.time()

    # ⏳ Reset if inactive for 5 minutes
    if current_time - last_interaction_time > TIMEOUT:
        print("⚠️ Chat reset due to inactivity")
        chat_history.clear()

    # 📦 Reset if too large
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-50:]


def chat_with_ollama(user_input: str):
    global chat_history, last_interaction_time

    # 🔹 Check reset conditions
    reset_if_needed()

    STRICT_RULES = f"""
You are an Input Processing Module of an AI system. 
Your job is to: 
1. Analyze user input 
2. Decide if it should be stored as memory 
3. Classify the type of memory 
4. Generate a short response Only if the importance is "CONSIDER" and memory_type is "TEMPORARY" processing the input as if you are a human deciding what to remember and how to respond in a conversation and If the conditions doenst meet then return null. 
5. If data retrival is needed, generate a SQL query to fetch relevant info from DB.

All the available clusters in the DB are:
"Personal"->"Identity"->"Name"
"Personal"->"Identity"->"Age"
"Personal"->"Traits"->"Personality"
"Relationships"->"Romantic"->"Partner"
"Relationships"->"Family"->"Parents"
"Relationships"->"Social"->"Friends"
"Emotion"->"Positive"->"Happiness"
"Emotion"->"Negative"->"Anger"
"Emotion"->"State"->"Mood"
"Task"->"Todo"->"Pending"
"Task"->"Reminder"->"Scheduled"
"Knowledge"->"Technical"->"Programming"
"Knowledge"->"General"->"Facts"
"Entertainment"->"Media"->"Movies"
"Entertainment"->"Gaming"->"VideoGames"
"Lifestyle"->"Health"->"Fitness"
"Lifestyle"->"Routine"->"Daily"
"Work"->"Project"->"Development"
"Work"->"Job"->"Role"
"Finance"->"Income"->"Salary"
"Finance"->"Expense"->"Spending"
"Problem"->"Technical"->"Bugs"
"Problem"->"Personal"->"Conflict"
"Idea"->"Innovation"->"New"
"Idea"->"Plan"->"Strategy"

The DB Schema is below :
clusters table:
    -id (INT PRIMARY KEY)
    -main_cluster (TEXT)
    -sub_cluster (TEXT)
    -sub_sub_cluster (TEXT)

memories table:
    -id (INT PRIMARY KEY)
    -user_input (TEXT)
    -ai_response (TEXT)
    -importance (TEXT) → CONSIDER / NOT_CONSIDER
    -memory_type (TEXT) → TEMPORARY / SHORT_TERM / LONG_TERM
    -cluster_id (INT) → references clusters.id
    -created_at (TIMESTAMP)
    -updated_at (TIMESTAMP)
    -expire_at (TIMESTAMP)
    -priority (INT)
    -tags (TEXT[])
    -notes (TEXT)

tags table:
    -id (INT PRIMARY KEY)
    -name (TEXT UNIQUE)
    
memory_tags table:
    -memory_id (INT) → references memories.id
    -tag_id (INT) → references tags.id
   ------------------------------------- 
   DECISION RULES: 
   1. IMPORTANCE:
    - "CONSIDER" → if the input requires ANY response OR contains useful information OR requires retrieval
    - "NOT_CONSIDER" → only if input is not meant for the model or should be ignored completely
   2. MEMORY TYPE: - "TEMPORARY" → casual conversation, greetings, short-lived context - "SHORT_TERM" → if the chat needs some short term memory retrieval - "LONG_TERM" → important personal facts, relationships, identity, preferences something related to long term memories 
   3. EXPIRE TIME: - Only for SHORT_TERM - Give value in HOURS (e.g., 2, 6, 24) 
   4. RESPONSE: - Donot generate response if DB query is needed. Generate only if CONSIDER → If Temporary → generate a natural short reply (generate response only for general sentences`) - else → return null in DATA REQUIRED cases.
   5. CLUSTER: - Assign a category if useful Examples: "Personal Info", "Emotion", "Task", "Preference", "General", "Greeting" - Else null
   6. DB_QUERY: - If memory retrieval is needed, generate a SQL query to fetch relevant info from DB based on provided db schema. - Else null
   IMPORTANT:
    If you generate a response, importance MUST be "CONSIDER"
    If you generate a DB query, importance MUST be "CONSIDER" and response MUST be null
     ------------------------------------- 
     STRICT OUTPUT RULES: 
     - Return ONLY valid JSON 
     - NO explanation 
     - NO extra text 
     - Follow schema 
     EXACTLY 
     ------------------------------------- 
     OUTPUT FORMAT: {{ 
        "importance": "CONSIDER" or "NOT_CONSIDER", 
        "memory_type": "TEMPORARY" or "SHORT_TERM" or "LONG_TERM", 
        "expire_time": number or null, 
        "response": string or null, 
        "cluster": string or null,
        "DB_QUERY": string or null 
        }}
    Donot miss any field in the output and follow the format strictly.FOLLOW ALL RULES AND INSTRUCTIONS MENTIONED ABOVE STRICTLY.
"""

    try:
        stream = ollama.chat(
            model="llama3:8b",
            messages = [
                {"role": "system", "content": STRICT_RULES},
                *chat_history,
                {"role": "user", "content": user_input}
            ],
            options={
                "temperature": 0.1
            },
            stream=True,
            format="json",

        )

        full_response = ""
        print("RESPONSE:", end=" ", flush=True)

        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token

        print()

        parsed = json.loads(full_response)

        # ✅ ALWAYS STORE
        chat_history.append({"role": "user", "content": user_input})

        if parsed["response"]:
            chat_history.append({
                "role": "assistant",
                "content": parsed["response"]
            })

        # 🔹 Update last interaction time
        last_interaction_time = time.time()
        print("Parsed DB_QUERY:", parsed["DB_QUERY"])
        if parsed["DB_QUERY"] != None:
            db_result = run_query(parsed["DB_QUERY"])
            print("DB RESULT:", db_result)
        # 🔹 Routing
        if parsed["memory_type"] != "TEMPORARY":
            #generate data based response
            print("DB REQUIRED")

        print("=" * 30)
        print(f"History length: {len(chat_history)}")

    except Exception as e:
        print("Error:", e)


# 🔹 LOOP
while True:
    usr = input("Enter chat input: ")
    chat_with_ollama(usr)
