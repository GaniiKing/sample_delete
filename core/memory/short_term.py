from collections import deque

class ShortTermMemory:
    def __init__(self, max_turns=10):
        self.buffer = deque(maxlen=max_turns)

    def add(self, role, content):
        self.buffer.append(f"{role}: {content}")

    def get_context(self):
        return "\n".join(self.buffer)
