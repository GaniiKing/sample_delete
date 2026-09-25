import requests
import json

class LLMEngine:
    def __init__(
        self,
        model: str = "llama3:8b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: int = 60
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        resp = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama usually returns "response"
        if "response" in data:
            return data["response"].strip()

        # Fallbacks for safety
        for key in ("text", "result", "output"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()

        return json.dumps(data)
