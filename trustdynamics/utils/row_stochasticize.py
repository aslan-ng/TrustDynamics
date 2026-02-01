import pandas as pd


def row_stochasticize(W: pd.DataFrame, self_weight_if_isolated: float = 1.0) -> pd.DataFrame:
    """
    Ensure W is row-stochastic. If a row sums to 0, assign self-weight.
    """
    W = W.copy().astype(float)
    row_sums = W.sum(axis=1)

    for i in W.index:
        if row_sums.loc[i] <= 0.0:
            W.loc[i, :] = 0.0
            W.loc[i, i] = float(self_weight_if_isolated)

    return W.div(W.sum(axis=1), axis=0)


if __name__ == "__main__":
    # Example: 3 agents
    W = pd.DataFrame(
        [
            [0.0, 1.0, 1.0],  # normal row
            [0.0, 0.0, 0.0],  # isolated agent
            [2.0, 1.0, 0.0],  # weighted row
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    W_hat = row_stochasticize(W)
    print(W_hat)
    print("Row sums:\n", W_hat.sum(axis=1))