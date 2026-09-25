from cvmemory import search_memory
import ollama

def answer_query(query):
    results = search_memory(query, top_k=3)

    context = "\n".join(
        [f"[Image {r['id']}] {r['description']}" for r in results]
    )

    final_prompt = f"""
        You are an assistant with access to visual memory.

        Here are relevant memories:
        {context}

        Question: {query}

        Answer based ONLY on the memories above.
"""

    response = ollama.chat(model="llama3:8b", messages=[
        {"role": "user", "content": final_prompt}
    ])

    return response["message"]["content"]


if __name__ == "__main__":
    q = input("Ask: ")
    print(answer_query(q))
