"""
Reference:
DeGroot, M. H. (1974). Reaching a Consensus. Journal of the American Statistical Association, 69(345), 118–121. https://doi.org/10.2307/2285509
"""


from __future__ import annotations
import pandas as pd


class Degroot:
    """
    DeGroot opinion dynamics model (scalar opinions only).

    Conventions:
    - influence matrix W is row-stochastic
    - W[i, j] = weight agent i assigns to agent j
    - opinions are a pd.Series indexed by agent id
    """

    def __init__(self, influence: pd.DataFrame):
        if not isinstance(influence, pd.DataFrame):
            raise TypeError("influence must be a pandas DataFrame")

        if not influence.index.equals(influence.columns):
            raise ValueError("Influence matrix index and columns must match (agent IDs).")

        self.influence = influence.copy()
        self.agents = influence.index
        self.num_agents = len(self.agents)

    def update_opinions(self, opinions: pd.Series) -> pd.Series:
        """
        Perform one DeGroot update step.

        Parameters
        ----------
        opinions : pd.Series
            Scalar opinion per agent (indexed by agent id)

        Returns
        -------
        pd.Series
            Updated opinions (indexed by agent id)
        """
        if not isinstance(opinions, pd.Series):
            raise TypeError("opinions must be a pandas Series")

        if not opinions.index.equals(self.agents):
            raise ValueError("Opinion index must match influence matrix agents.")

        # DeGroot update
        x_next = self.influence @ opinions

        # Preserve Series metadata
        x_next.name = opinions.name

        return x_next

    def run_steps(
        self,
        opinions: pd.Series,
        steps: int | None = None,
        threshold: float = 1e-6,
        max_steps: int = 10_000,
    ) -> dict:
        """
        Run DeGroot dynamics until convergence or for a fixed number of steps.

        Convergence metric:
            delta_inf = max_i |x_next[i] - x[i]|
        """
        if not isinstance(opinions, pd.Series):
            raise TypeError("opinions must be a pandas Series")

        if not opinions.index.equals(self.agents):
            raise ValueError("Opinion index must match influence matrix agents.")

        if threshold <= 0:
            raise ValueError("threshold must be positive.")

        if steps is not None and steps < 0:
            raise ValueError("steps must be >= 0 or None.")

        x = opinions.copy()
        num_steps = 0
        converged = False
        delta_inf = float("inf")

        iters_allowed = steps if steps is not None else max_steps

        for _ in range(iters_allowed):
            x_next = self.influence @ x
            delta_inf = float((x_next - x).abs().max())
            x = x_next
            num_steps += 1

            if steps is None and delta_inf <= threshold:
                converged = True
                break

        x.name = opinions.name

        return {
            "final_opinions": x,
            "num_steps": num_steps,
            "converged": converged,
            "delta_inf": delta_inf,
        }


if __name__ == "__main__":
    W = pd.DataFrame(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.4, 0.4],
        ],
        index=[0, 1, 2],
        columns=[0, 1, 2],
    )

    model = Degroot(W)

    opinions = pd.Series([0.1, 0.3, 0.5], index=W.index, name="opinions")
    result = model.run_steps(
        opinions,
        steps=1
    )

    print(result["final_opinions"])
