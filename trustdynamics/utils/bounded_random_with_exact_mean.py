import numpy as np


def bounded_random_with_exact_mean(
    n: int,
    target_mean: float,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
    seed: np.random.Generator | int | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Generate n random values in [min_value, max_value] with (numerically) exact target mean.
    """

    if min_value >= max_value:
        raise ValueError("min_value must be < max_value.")

    if not (min_value <= target_mean <= max_value):
        raise ValueError("target_mean must be within [min_value, max_value].")

    # RNG handling
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    if n <= 0:
        return np.array([], dtype=float)

    # Initial draw
    x = rng.uniform(min_value, max_value, size=n)

    # Shift to target mean
    x = x - x.mean() + target_mean
    x = np.clip(x, min_value, max_value)

    target_sum = n * target_mean
    delta = target_sum - float(x.sum())

    for _ in range(max_iter):
        if abs(delta) < tol:
            break

        if delta > 0:
            movable = x < max_value
            if not np.any(movable):
                break
            room = max_value - x[movable]
            step = min(delta / room.size, room.max())
            inc = np.minimum(room, step)
            x[movable] += inc
            delta -= float(inc.sum())
        else:
            movable = x > min_value
            if not np.any(movable):
                break
            room = x[movable] - min_value
            step = min((-delta) / room.size, room.max())
            dec = np.minimum(room, step)
            x[movable] -= dec
            delta += float(dec.sum())

    # Final numerical cleanup
    x = np.clip(x, min_value, max_value)
    residual = target_sum - float(x.sum())

    if abs(residual) > tol:
        if residual > 0:
            idxs = np.where(x < max_value)[0]
            if idxs.size:
                i = idxs[0]
                x[i] = min(max_value, x[i] + residual)
        else:
            idxs = np.where(x > min_value)[0]
            if idxs.size:
                i = idxs[0]
                x[i] = max(min_value, x[i] + residual)

    return x


if __name__ == "__main__":
    samples = bounded_random_with_exact_mean(
        n=10,
        target_mean=0.3,
        min_value=-2.0,
        max_value=5.0,
        seed=42,
    )
    print(samples)
    print(samples.mean())