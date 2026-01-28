from abc import ABC, abstractmethod

class DashboardPlugin(ABC):
    name: str

    @abstractmethod
    def fetch(self, run_id: str):
        pass
