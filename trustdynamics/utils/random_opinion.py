import numpy as np


def random_opinions(
        n: int = 1,
        dimensions: int = 1,
        opinion_min: float = -1,
        opinion_max: float = 1
    ):
    """
    Generates random opinion vectors for n players with specific number of topics.
    """
    if n < 1:
        raise ValueError("Number of players must be at least 1.")
    if dimensions < 1:
        raise ValueError("Number of topics must be at least 1.")
    if opinion_min >= opinion_max:
        raise ValueError("opinion_min must be less than opinion_max.")

    # Generate random values in range [0,1] and scale to [opinion_min, opinion_max]
    matrix = opinion_min + (opinion_max - opinion_min) * np.random.rand(n, dimensions)

    return matrix.tolist()  # Convert to list of lists


if __name__ == "__main__":

    from pprint import pprint

    opinions = random_opinions(n=2, dimensions=3)
    pprint(opinions)