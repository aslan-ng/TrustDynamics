import numpy as np

#from trustevolution.model.functions import cosine_similarity


class FriedkinJohnsen:
    """
    Friedkin & Johnsen (1999) Social Influence Model
    Reference: Friedkin, Noah & Johnsen, Eugene. (1999). Social Influence Networks and Opinion Change. Advances in Group Processes. 16.
    """
    
    def update_step(
            opinion_initial: list,
            influence_weight_matrix: list,
            influence_resistance: float,
            opinions_previous: list = None,
        ):
        """
        Update opinion
        opinion_initial: initial opinion vector of the agent
        opinions_previous: all opinions of all agents in during the previous step
        influence_weight_matrix: a matrix showing influence of i on j
        influence_resistance: resistance of the agent towards being influenced by others
        """
        # Convert to numpy arrays
        opinion_initial = np.array(opinion_initial).reshape(-1, 1)
        if opinions_previous is None:
            opinions_previous = opinion_initial.copy()
        else:
            opinions_previous = np.array(opinions_previous).reshape(-1, 1)
        influence_weight_matrix = np.array(influence_weight_matrix)
        # Validation
        if opinion_initial.shape[0] != 1 or \
            opinion_initial.shape[1] != opinions_previous.shape[1]:
            raise ValueError("Shapes of initial opinion and previous opinions do not match.")
        if not (0 <= influence_resistance <= 1):
            raise ValueError("Influence resistance must be between 0 and 1.")
        if influence_weight_matrix.ndim != 2 or \
        influence_weight_matrix.shape[0] != influence_weight_matrix.shape[1] or \
        influence_weight_matrix.shape[0] != opinions_previous.shape[0]:
            raise ValueError("Influence weight matrix must be a square matrix of size (n, n), matching the number of agents.")
        # Update
        opinion = influence_resistance * opinion_initial + (1 - influence_resistance) * np.dot(influence_weight_matrix, opinions_previous)
        return opinion
    
    def update(
        opinions_initial,
        opinions_previous,
        influence_weight_matrix,
        influence_resistance: float,
        steps: int = 1,
        threshold: float = None,
    ):
        if threshold is None:
            for _ in range(steps):

                pass


if __name__ == "__main__":
    # Initialize normalized weights
    influence_resistance = [
        [0.9, 0.1],  # Influence weights human on self, human on computer
        [0.1, 0.9],  # Influence weights computer on human, computer on self
    ]