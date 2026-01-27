import numpy as np

from BASEMENT.opinion import Opinion


class Agent:

    def __init__(
        self,
        initial_confidence: float = None,
        initial_trust: float = None,
    ):
        self.opinion = None
        self.confidence = initial_confidence
        self.trust = initial_trust

    def set_opinion(self, opinion: list | tuple | np.ndarray | Opinion):
        if isinstance(opinion, Opinion):
            self.opinion = opinion
        else:
            self.opinion = Opinion(opinion)

    def set_confidence(self, confidence: float):
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1.")
        self.confidence = confidence

    def set_trust(self, trust: float):
        if trust < 0 or trust > 1:
            raise ValueError("Trust must be between 0 and 1.")
        self.trust = trust


if __name__ == "__main__":
    agent = Agent(initial_confidence=0.8, initial_trust=0.5)
    agent.set_opinion([0.1, -0.2, 0.3])
    