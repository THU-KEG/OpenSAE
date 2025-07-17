from abc import ABC, abstractmethod

class AgentBase(ABC):
    """
    Abstract base class for all agents in the framework.
    """
    def __init__(self, settings: dict):
        self.settings = settings

    @abstractmethod
    def run(self, data: dict) -> dict:
        """
        Execute the agent logic on input data and return output data.
        """
        pass