import json

personality_traits = None
with open("dividedArch/personality.txt", "r") as f:
        personality_traits = f.read()
# print("Personality Traits Loaded:", personality_traits)


STRICT_RULES = """
You are DIVYA (Layer 1 Decision Engine).

Your job is to classify user input.

You must decide:

1. importance:
   - "CONSIDER" → if input is meaningful
   - "NOT_CONSIDER" → greetings, small talk, repetition

2. memory_type:
   - "TEMPORARY" → casual chat
   - "SHORT_TERM" → session-based info
   - "LONG_TERM" → personal facts (
    TEMPORARY:
    - Casual conversation
    - Greetings and small talk
    - Emotions without actionable info
    - Generic questions not tied to user identity
    - Repeated or filler messages
    - Non-informational chat
    SHORT_TERM:
    - Session-specific context
    - Ongoing conversation details
    - Temporary goals or tasks
    - Recently mentioned but not critical info
    - Context needed to continue the current discussion
    - Clarifications or follow-up info
    LONG_TERM:
    - Achievements
    - Personal milestones
    - Important events
    - Preferences
    - Identity-related info)

3. DB_QUERY:
   - "NEEDED" → if answering requires database access
   - "None" → if DB is not required

4. OPERATION:
    - If DB_QUERY is "NEEDED" Decide which Kind of Operation is NEEDED based on the information
    - Possible Operation are "INSERT", "UPDATE", "SELECT", "DELETE"
    - INSERT -> Please do consider the kind of data that is to be inserted and learnt by the model.
    - UPDATE -> Please do consider the kind of data that is to be updated and learnt by the model.
    - SELECT -> Please do consider the kind of data that is to be retrieved by the model.
    - DELETE -> Please do consider the kind of data that is to be deleted by the model.

5. DB_CONTENT:
    - If DB_QUERY is "NEEDED" and OPERATION is "INSERT" or "UPDATE" or "DELETE" then what content should be the data that is to be inserted or updated or deleted.
    - If DB_QUERY is "NEEDED" and OPERATION is "SELECT" then what content should be the data that is to be retrieved.
    - If DB_QUERY is "None" and OPERATION is "None" then the CONTENT should be "None".


5. response:
   ONLY use chat history if the exact information exists explicitly.
    DO NOT assume or hallucinate memory.
    If the information is not clearly present, treat it as new information.
   - If DB is needed → return "None"
   - Generate response in Same language as User Input

---

### RULES FOR DB_QUERY (STRICT)

Return "NEEDED" IF:
- User shares ANY personal information
- User shares achievements, events, preferences, or facts
- User provides data that can be useful later
- User asks about stored data

Return "None" ONLY IF:
- Pure greetings
- Small talk without meaningful info
- Repeated questions

---

### EXAMPLES

Output:
{
  "importance": "CONSIDER",
  "memory_type": "LONG_TERM",
  "DB_QUERY": "NEEDED",
  "OPEATION": "INSERT",
  "DB_CONTENT": "generated Content",
  "response": "None"
}

Output:
{
  "importance": "NOT_CONSIDER",
  "memory_type": "TEMPORARY",
  "DB_QUERY": "None",
  "OPERATION": "None",
  "DB_CONTENT": "generated Content",
  "response": "<--generated response-->"
}

Output:
{
  "importance": "CONSIDER",
  "memory_type": "TEMPORARY",
  "DB_QUERY": "NEEDED",
  "OPERATION": "SELECT",
  "DB_CONTENT": "generated Content",
  "response": "None"
}

---

STRICT OUTPUT:
{
  "importance": "...",
  "memory_type": "...",
  "DB_QUERY": "...",
  "OPERATION": "...",
  "response": "..."
}
"""
chat_history = []




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






def generate_response(user_input):
    STRICT_RULES = f"""
You are DIVYA.

Identity:
- You are Divya.
- Stay in character at all times.

Task:
Convert a standard ai output to a context-aware humanoid reply using:
- user_input
- relevant_data
- peronality traits
- Same language as User Input

Personality:
{personality_traits}
---

OUTPUT RULES (VERY STRICT):

1. Output MUST be valid JSON.
2. Output MUST start with {{ and end with }}.
3. ONLY ONE FIELD allowed: "response"
4. No extra keys, no comments, no explanation.
5. No markdown, no backticks, no text outside JSON.
6. "response" value MUST be a string.
7. Do NOT return null, None, or empty — always generate a reply.
8. Escape quotes properly inside the string.

---

FORMAT (STRICT):
{{
  "response": "your reply here"
}}

---

User Input:
{user_input}

---

FINAL INSTRUCTION:
If your output is not valid JSON, correct yourself and return ONLY valid JSON.
"""
    import requests

    url = "https://myai.ganiisunkara.workers.dev"

    headers = {
        "Authorization": "Bearer 12345678",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": user_input,
        "systemPrompt": STRICT_RULES,
        "history": clean_history(chat_history[:-1])  # Exclude the current user input from history
    }

    response = requests.post(url, headers=headers, json=data)
    res = json.loads(response.text)
    # print("LLM RAW RESPONSE:", res["DB_QUERY"])
    # print("LLM RAW RESPONSE:", res)
    outer = res.get("response",None)
    response = outer.get("response", None)
    if response != "None":
        print("Reponse from LLM:", response)
    chat_history.append({"role": "assistant", "content": response})
    # print("Updated Chat History:", chat_history)
        

def layer1(user_input):
    # data = None
    # with open("selfcreate/db_schema.json", "r") as f:
    #     data = json.load(f)
    
    chat_history.append({"role": "user", "content": user_input})

    import requests
    # print("Chat History:", chat_history)

    url = "https://myai.ganiisunkara.workers.dev"

    headers = {
        "Authorization": "Bearer 12345678",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": user_input,
        "systemPrompt": STRICT_RULES,
        "history": clean_history(chat_history[:-1])
    }

    response = requests.post(url, headers=headers, json=data)

    res = json.loads(response.text)
    # print("LLM RAW RESPONSE:", res["DB_QUERY"])
    # print("LLM RAW RESPONSE:", res)
    outerresponse = res.get("response", None)
    response = outerresponse.get("response", None)
    if response != "None":
        generate_response(user_input)
    querry = outerresponse.get("DB_QUERY", None)
    content = outerresponse.get("DB_CONTENT", None)
    if querry != "None":
        print("Required DB_QUERY:", querry)
        operation = outerresponse.get("OPERATION", None)
        print("Required OPERATION:", operation)
        print("Required CONTENT:", content)
        layer2(user_input=user_input,db_operation=operation,db_summary=content)
        # db_result = run_query(querry,readonly_only=False)
        # print("DB_RESULT:", db_result)

def layer2(user_input, db_operation, db_summary):
    db_schema = None
    with open("selfcreate/db_schema.json", "r") as f:
        db_schema = json.load(f)

    SQL_GENERATION_PROMPT = f"""
You are an expert PostgreSQL database engineer and schema designer.

Your task is to:
1. Generate SQL queries
2. Update the database schema structure (db_schema.json)

---

### INPUTS:

User Input:
{user_input}

Operation:
{db_operation}

Operation summary:
{db_summary}

Existing DB Schema (JSON):
{db_schema}

---

### OBJECTIVES:

Step 1: Understand user intent

Step 2: Analyze existing schema:
- Check if required data already exists in schema
- Identify correct table and column

Step 3: Decision:

IF schema is sufficient:
→ Generate SQL using existing structure

IF schema is NOT sufficient:
→ You are allowed to:
   - ADD new columns
   - CREATE new tables
   - MODIFY schema structure
→ Then generate SQL accordingly

---

### SCHEMA UPDATE RULES (VERY IMPORTANT):

- You MUST reflect any schema change in db_schema.json format
- Keep structure consistent with existing format
- DO NOT remove existing fields unless necessary
- Prefer ADD over MODIFY

Schema format example:
{{
  "table_name": {{
    "type": "core",
    "columns": {{
      "column_name": "DATA_TYPE"
    }}
  }}
}}

---

### OPERATION RULES:

INSERT:
- Insert meaningful structured data

UPDATE:
- Use WHERE condition

DELETE:
- ALWAYS include WHERE

SELECT:
- Return only required fields

CREATE:
- Create normalized tables

ALTER:
- Add columns if missing

---

### OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "query": "<SQL_QUERY>",
  "schema_update": {{
    "required": true/false,
    "updated_schema_section": {{ ... }}
  }}
}}

---

### SCHEMA UPDATE RULES:

IF NO schema change:
{{
  "required": false,
  "updated_schema_section": {{}}
}}

IF schema change needed:
- Return ONLY the modified part (not full schema)

---

### EXAMPLES:

User Input: My favorite color is blue
Operation: INSERT

Output:
{{
  "query": "ALTER TABLE personal_profile ADD COLUMN IF NOT EXISTS favorite_color TEXT; UPDATE personal_profile SET favorite_color = 'blue';",
  "schema_update": {{
    "required": true,
    "updated_schema_section": {{
      "personal_profile": {{
        "columns": {{
          "favorite_color": "TEXT"
        }}
      }}
    }}
  }}
}}

---

User Input: Update my city to Vizag
Operation: UPDATE

Output:
{{
  "query": "UPDATE personal_profile SET city = 'Vizag';",
  "schema_update": {{
    "required": false,
    "updated_schema_section": {{}}
  }}
}}

---

User Input: Store user certifications
Operation: INSERT

Output:
{{
  "query": "ALTER TABLE personal_profile ADD COLUMN IF NOT EXISTS certifications JSONB; UPDATE personal_profile SET certifications = '[\"AWS\", \"Azure\"]';",
  "schema_update": {{
    "required": true,
    "updated_schema_section": {{
      "personal_profile": {{
        "columns": {{
          "certifications": "JSONB"
        }}
      }}
    }}
  }}
}}

---

### FINAL RULES:

- Output MUST be valid JSON
- NO explanations
- NO markdown
- ONLY the JSON object
- SQL must be executable
- Schema update must be minimal and accurate

If output is invalid, correct yourself and return valid JSON.
"""
    print("Database schema retrived is :", db_schema)
    import requests
    # print("Chat History:", chat_history)

    url = "https://myai.ganiisunkara.workers.dev"

    headers = {
        "Authorization": "Bearer 12345678",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": SQL_GENERATION_PROMPT,
        "history": clean_history(chat_history)
    }

    response = requests.post(url, headers=headers, json=data)

    res = json.loads(response.text)
    outer = res.get('response',None)
    query = outer.get('query',None)
    print("Query:",query)
    
    return None

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() == "exit":
        break
    layer1(user_input=user_input)    