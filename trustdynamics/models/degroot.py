"""
Reference:
DeGroot, M. H. (1974). Reaching a Consensus. Journal of the American Statistical Association, 69(345), 118–121. https://doi.org/10.2307/2285509
"""

from __future__ import annotations
import numpy as np
from trustdynamics.influence_matrix import Influence


class Degroot:

    def __init__(self, influence: Influence):
        self.num_agents = influence.num_agents
        self.influence = influence

    def update_opinions(self, opinions: list | np.ndarray) -> np.ndarray:
        """
        Perform one DeGroot update step.

        opinions:
          - numpy array of shape (N, d), or
          - list of length N where each element is a vector (length d)

        returns:
          - updated opinions as numpy array of shape (N, d)
        """
        X = np.asarray(opinions, dtype=float)

        # Accept either (N, d) or (N,) with vector-like elements.
        # Most users should pass (N, d).
        if X.ndim == 1:
            # Could be a list of vectors; convert to 2D
            X = np.vstack([np.asarray(v, dtype=float) for v in opinions])

        if X.ndim != 2:
            raise ValueError("Opinions must be a 2D array of shape (num_agents, opinion_dim).")

        if X.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected opinions for {self.num_agents} agents, got {X.shape[0]}."
            )

        W = self.influence.influence_matrix  # (N, N)
        X_next = W @ X  # (N, d)
        return X_next
    
    def run_steps(
            self,
            opinions: list | np.ndarray,
            steps: int | None = None,
            threashold: float = 1e-6,
            max_steps: int | None = 10_000
        ) -> dict:
        """
        Run a number of DeGroot steps starting from opinions.
        If steps is None, run until convergence. If steps is an integer, run for that many steps.
        In an case, if the opinions converge (change less than threashold), stop and return the final opinions.
        If max_steps is not None, stop after that many steps to prevent infinite loops.
        Convergence metric ('delta_inf'):
            delta_inf = max_{i,k} |X_next[i,k] - X[i,k]|
        """
        if threashold <= 0:
            raise ValueError("threashold must be positive.")  # keep your spelling, but consider renaming to threshold
        if steps is not None and steps < 0:
            raise ValueError("steps must be >= 0 or None.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")

        # Parse opinions into (N, d)
        X = np.asarray(opinions, dtype=float)
        if X.ndim == 1:
            X = np.vstack([np.asarray(v, dtype=float) for v in opinions])
        if X.ndim != 2:
            raise ValueError("Opinions must be a 2D array of shape (num_agents, opinion_dim).")
        if X.shape[0] != self.num_agents:
            raise ValueError(f"Expected opinions for {self.num_agents} agents, got {X.shape[0]}.")

        W = self.influence.influence_matrix  # (N, N)

        # Decide how many iterations we are allowed to run
        # - If steps is provided: run at most steps (and also respect max_steps if provided)
        # - If steps is None: run until convergence (bounded by max_steps if provided; otherwise use a safe default)
        if steps is None:
            iters_allowed = max_steps if max_steps is not None else 10_000
        else:
            iters_allowed = steps if max_steps is None else min(steps, max_steps)

        num_steps = 0
        converged = False
        delta_inf = float("inf")  # convergence metric from the most recent update

        for _ in range(iters_allowed):
            X_next = W @ X
            delta_inf = float(np.max(np.abs(X_next - X)))  # max componentwise change across all agents and dimensions
            X = X_next
            num_steps += 1

            if delta_inf <= threashold:
                converged = True
                break

        return {
            "final_opinions": X,
            "num_steps": num_steps,
            "converged": converged,
            "delta_inf": delta_inf,
            }
    

if __name__ == "__main__":
    influence_matrix = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.4, 0.4]
        ]
    )
    influence = Influence(influence_matrix)
    model = Degroot(influence)
    opinion = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    result = model.run_steps(opinions=opinion, steps=None)
    print(result)
