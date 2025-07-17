from .agent_base import AgentBase

class Scheduler:
    """
    Orchestrates sequential execution of agents with shared state.
    """
    def __init__(self, agent_classes: list, settings: dict):
        # Instantiate each agent with its corresponding settings
        self.agents = [cls(settings[cls.__name__]) for cls in agent_classes]
        self.state = {}

    def dispatch(self, initial_input: dict) -> dict:
        self.state.update(initial_input)
        for agent in self.agents:
            output = agent.run(self.state)
            self.state.update(output)
        return self.state