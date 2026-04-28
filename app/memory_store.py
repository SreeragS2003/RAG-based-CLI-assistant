class Memory:
    def __init__(self, max_turns = 5):
        self.history = []
        self.max_turns = max_turns

    def add(self, user, assistant):
        self.history.append({
            "user": user,
            "assistant": assistant
        })
        if len(self.history) > self.max_turns:
            self.history.pop(0) #Remove the oldest entry if we exceed max turns to keep memory relevant and concise

    def get(self):
        return "\n".join(
            f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in self.history[-5:]
        )