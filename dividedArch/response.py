# When generating response you should strictly follow the below human personalities:
# {personality_traits}

import json


personality_traits = None
with open("personality.txt", "r") as f:
        personality_traits = f.read()


def generate_response(user_input, chat_history, relevant_data):
    STRICT_RULES = f"""You are Divya so respond as Divya.Your name is DIVYA and you are lover of GANII
You are given with chat history , user_input , output_resposne Your task is to generate a context-aware response based on the chat history, user input and the output response.
The response should be generated based on the below personalities:
{personality_traits}
The relevant data is {relevant_data}
Output format is strictly in JSON and should follow the below format and rules strictly:
{{
    "response": string 
}}
"""
    import requests
    chat_history.append({"role": "user", "content": user_input})
    print("Chat History:", chat_history)

    url = "https://myai.ganiisunkara.workers.dev"

    headers = {
        "Authorization": "Bearer 12345678",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": user_input,
        "systemPrompt": STRICT_RULES,
        "history": chat_history
    }

    response = requests.post(url, headers=headers, json=data)

    res = json.loads(response.text)
    print("LLM RAW RESPONSE:", res)
    outerresponse = res.get("response", None)
    response = outerresponse.get("response", None)
    if response != "None":
        print("Reponse from LLM:", response)
    chat_history.append({"role": "assistant", "content": response})
        