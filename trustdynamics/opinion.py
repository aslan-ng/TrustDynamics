from pprint import pprint
import numpy as np
from trustdynamics.utils.cosine_similarity import cosine_similarity


class Opinion:

    def __init__(self, values: list | tuple | np.ndarray = []):
        self.values = list(values)

    def show(self):
        pprint(self.values)

    def randomize(
            self,
            dimensions: int,
            opinion_min: float = -1,
            opinion_max: float = 1,
            seed = None,
            seed_state = None
        ):
        if seed is not None:
            np.random.seed(seed)
        if seed_state is not None:
            np.random.set_state(seed_state)
        self.values = np.random.uniform(opinion_min, opinion_max, dimensions).tolist()
        return self.values
    
    def cosine_similarity(self, other_opinion) -> float:
        if isinstance(other_opinion, Opinion):
            other_opinion_values = other_opinion.values
        elif isinstance(other_opinion, list) or \
            isinstance(other_opinion, tuple) or \
            isinstance(other_opinion, np.ndarray):
            other_opinion_values = other_opinion
        return cosine_similarity(self.values, other_opinion_values)


if __name__ == "__main__":
    opinion = Opinion([0.2, -0.5, 0.8])
    opinion.show()