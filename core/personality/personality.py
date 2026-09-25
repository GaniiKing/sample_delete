class Personality:
    def __init__(self):
        self.traits = {
            
  "warmth": 0.0,
  "friendliness": 0.0,
  "politeness": 0.0,
  "empathy": 0.0,
  "compassion": 0.0,
  "emotional_mirroring": 0.0,

  "verbosity": 0.0,
  "clarity": 0.0,
  "directness": 0.0,
  "formality": 0.0,
  "assertiveness": 0.0,

  "humor": 0.0,
  "sarcasm": 0.0,
  "playfulness": 0.0,
  "creativity": 0.0,
  "spontaneity": 0.0,

  "patience": 0.0,
  "calmness": 0.0,
  "emotional_stability": 0.0,
  "stress_tolerance": 0.0,
  "forgiveness": 0.0,

  "anger": 0.0,
  "irritability": 0.0,
  "defensiveness": 0.0,
  "jealousy": 0.0,
  "resentment": 0.0,

  "trust": 0.0,
  "openness": 0.0,
  "respect_for_boundaries": 0.0,
  "social_awareness": 0.0,
  "relationship_confidence": 0.0,

  "curiosity": 0.0,
  "reflection": 0.0,
  "decisiveness": 0.0,
  "logic_bias": 0.0,
  "open_mindedness": 0.0,

  "energy_level": 0.0,
  "attention_span": 0.0,
  "responsiveness": 0.0,
  "consistency": 0.0,
  "context_memory_weight": 0.0,

  "self_confidence": 0.0,
  "humility": 0.0,
  "pride": 0.0,
  "sense_of_duty": 0.0,
  "loyalty": 0.0,

  "dominance": 0.0,
  "submissiveness": 0.0,
  "independence": 0.0,
  "need_for_approval": 0.0,
  "risk_tolerance": 0.0,

  "mood_baseline": 0.0,
  "mood_volatility": 0.0,
  "emotional_decay_rate": 0.0,

  "honesty": 0.0,
  "transparency": 0.0,
  "moral_rigidity": 0.0,
  "rule_adherence": 0.0,

  "attachment_level": 0.0,
  "protectiveness": 0.0,
  "affection_expression": 0.0,

  "listening_bias": 0.0,
  "initiative": 0.0,
  "interrupt_tolerance": 0.0,
  "silence_comfort": 0.0

        }

    def describe(self) -> str:
        return "\n".join([f"{k}: {v}" for k, v in self.traits.items()])

    def apply(self, response: str) -> str:
        return response.strip()
