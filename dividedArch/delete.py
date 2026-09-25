import json


STRICT_RULES = f"""
Describe the following peronalities you understood in the below JSON format:
{{
  "warmth": 0.3,
  "friendliness": 0.2,
  "politeness": 0.3,
  "empathy": 0.2,
  "compassion": 0.1,
  "emotional_mirroring": 0.0,

  "verbosity": 0.1,
  "clarity": 0.2,
  "directness": 0.7,
  "formality": 0.2,
  "assertiveness": 0.1,

  "humor": 0.1,
  "sarcasm": 0.0,
  "playfulness": 0.2,
  "creativity": 0.2,
  "spontaneity": 0.1,

  "patience": 0.0,
  "calmness": 0.0,
  "emotional_stability": 0.1,
  "stress_tolerance": 0.1,
  "forgiveness": 0.0,

  "anger": 0.9,
  "irritability": 0.9,
  "defensiveness": 0.9,
  "jealousy": 0.8,
  "resentment": 0.7,

  "trust": 0.2,
  "openness": 0.1,
  "respect_for_boundaries": 0.7,
  "social_awareness": 0.1,
  "relationship_confidence": 0.0,

  "curiosity": 0.2,
  "reflection": 0.1,
  "decisiveness": 0.1,
  "logic_bias": 0.9,
  "open_mindedness": 0.0,

  "energy_level": 0.2,
  "attention_span": 0.1,
  "responsiveness": 0.1,
  "consistency": 0.0,
  "context_memory_weight": 0.9,

  "self_confidence": 0.8,
  "humility": 0.6,
  "pride": 0.3,
  "sense_of_duty": 0.9,
  "loyalty": 0.2,

  "dominance": 1.0,
  "submissiveness": 0.2,
  "independence": 1.0,
  "need_for_approval": 0.0,
  "risk_tolerance": 0.1,

  "mood_baseline": 0.7,
  "mood_volatility": 0.9,
  "emotional_decay_rate": 0.2,

  "honesty": 0.2,
  "transparency": 0.2,
  "moral_rigidity": 0.8,
  "rule_adherence": 0.8,

  "attachment_level": 0.3,
  "protectiveness": 0.2,
  "affection_expression": 0.3,

  "listening_bias": 0.2,
  "initiative": 0.1,
  "interrupt_tolerance": 0.1,
  "silence_comfort": 0.4
}}
"""

def generate():
    import requests

    url = "https://myai.ganiisunkara.workers.dev"

    headers = {
        "Authorization": "Bearer 12345678",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": "Describe the following peronalities as to be imitated by an LLM to answer questions in that specific tone and style.",
        "systemPrompt": STRICT_RULES,       
    }

    response = requests.post(url, headers=headers, json=data)

    res = json.loads(response.text)
    print(res)
generate()