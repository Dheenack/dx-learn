class EventStore:
    def __init__(self):
        self.events = []

    def log(self, event: dict):
        self.events.append(event)

    def fetch(self, run_id):
        return [e for e in self.events if e["run_id"] == run_id]
