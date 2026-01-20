import numpy as np


class Influence:

    def __init__(self, influence_matrix: np.ndarray):
        """
        influence_matrix: A square matrix where each entry W[i][j] represents the influence weight of agent j on agent i.
        The matrix must satisfy the following conditions:
        1. Square and non-empty: The matrix must be square (same number of rows and columns) and cannot be empty.
        2. Nonnegative: All entries must be nonnegative, as they represent influence weights.
        3. Row normalization: Each row must sum to 1, ensuring that the influence weights for each agent sum to 1.
        """
        self.influence_matrix = self._validate_and_normalize(influence_matrix)

    def _validate_and_normalize(self, influence_matrix: np.ndarray) -> np.ndarray:
        W = np.asarray(influence_matrix, dtype=float)

        # 1. Square and non-empty
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("Influence matrix must be square.")
        if W.shape[0] == 0:
            raise ValueError("Influence matrix cannot be empty.")

        # 2. Nonnegative
        if np.any(W < 0):
            raise ValueError("Influence weights must be nonnegative.")

        # 3. Row normalization
        row_sums = W.sum(axis=1)
        if np.any(row_sums == 0):
            raise ValueError("Each row must have a positive sum to normalize.")

        W = W / row_sums[:, None]
        return W
    
    @property
    def num_agents(self) -> int:
        return self.influence_matrix.shape[0]

    def __str__(self):
        return str(self.influence_matrix)
    

if __name__ == "__main__":
    influence_matrix = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.4, 0.4]
        ]
    )
    influence = Influence(influence_matrix)
    print(influence)