class MemoryManager:
    def __init__(self, stm, ltm):
        self.stm = stm
        self.ltm = ltm

    def process_input(self, user_input):
        if user_input.lower().startswith("remember"):
            content = user_input.replace("remember", "").strip()
            self.ltm.store("fact", content)
            return "I'll remember that."

    def build_context(self):
        long_mem = self.ltm.retrieve_all()
        formatted_ltm = "\n".join([f"- {c}" for _, c in long_mem])
        return f"Known facts:\n{formatted_ltm}"
