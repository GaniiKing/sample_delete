from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.memory.memory_manager import MemoryManager
from core.dialogue.dialogue_manager import DialogueManager

stm = ShortTermMemory()
ltm = LongTermMemory()
memory_manager = MemoryManager(stm, ltm)
dialogue = DialogueManager(stm, memory_manager)

print("Divya AI initialized. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    reply = dialogue.respond(user_input)
    print("Divya:", reply)
