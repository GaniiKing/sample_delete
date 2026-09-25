from core.llm.llm_engine import LLMEngine
from core.personality.personality import Personality

class DialogueManager:
    def __init__(self, stm, memory_manager):
        self.llm = LLMEngine()
        self.personality = Personality()
        self.stm = stm
        self.memory_manager = memory_manager

        with open("core/config/system_prompt.txt") as f:
            self.system_prompt = f.read()

    def respond(self, user_input: str) -> str:
        self.stm.add("User", user_input)

        memory_response = self.memory_manager.process_input(user_input)
        if memory_response:
            return memory_response

        prompt = f"""
{self.system_prompt}

Personality traits:
{self.personality.describe()}

{self.memory_manager.build_context()}

Conversation so far:
{self.stm.get_context()}

User: {user_input}
Divya:
"""

        raw_response = self.llm.generate(prompt)
        response = self.personality.apply(raw_response)

        self.stm.add("Divya", response)
        return response
